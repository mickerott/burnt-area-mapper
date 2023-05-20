import datetime
import re
import warnings
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import rasterio
import requests
import shapely
import xmltodict
from requests.exceptions import JSONDecodeError
from retry import retry

import geo

S2_L2A_BANDS = [
    "AOT",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B11",
    "B12",
    "B8A",
    "SCL",
    "WVP",
]
S2_L2A_RESOLUTIONS = {
    "AOT": "R10m",
    "B02": "R10m",
    "B03": "R10m",
    "B04": "R10m",
    "B08": "R10m",
    "TCI": "R10m",
    "WVP": "R10m",
    "B05": "R20m",
    "B06": "R20m",
    "B07": "R20m",
    "B11": "R20m",
    "B12": "R20m",
    "B8A": "R20m",
    "SCL": "R20m",
    "B01": "R60m",
    "B09": "R60m",
    "B10": "R60m",
}
S2_L2A_RESOLUTIONS_BY_RES = {
    "R10m": ["AOT", "B02", "B03", "B04", "B08", "TCI", "WVP"],
    "R20m": ["B05", "B06", "B07", "B11", "B12", "B8A", "SCL"],
    "R60m": ["B01", "B09", "B10"],
}
QUANTIFICATION_VALUE = 10000


class S2Query:
    def __init__(
        self,
        start_date,
        end_date,
        geojson=None,
        bounds=None,
        max_cc=100,
        min_dc=0,
        tile_id=None,
        endpoint=None,
        collection="sentinel-s2-l2a-cogs",
        items_limit=1000,
    ):
        self.endpoint = (
            "https://earth-search.aws.element84.com/v0/search"
            if endpoint is None
            else endpoint
        )
        self.collection = collection
        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime(
            "%Y-%m-%dT00:00:00Z"
        )  # note: scenes from that day are NOT included when passed like
        # this (i.e. must consider this in input parameter for end_date
        if geojson is None and bounds is not None:
            self.geojson = self.geojson_from_bounds(bounds)
        elif geojson is not None and bounds is None:
            self.geojson = geojson
        else:
            raise ValueError('Specify (only) one of: "geojson" or "bounds".')
        self.max_cc = max_cc
        self.tile_id = tile_id
        self.min_dc = min_dc
        self.urls_dict = None
        self.items_limit = items_limit

        if not self.endpoint.startswith("https://earth-search.aws.element84.com/"):
            if self.min_dc != 0:
                raise ValueError(
                    "Search parameter min_dc only supported by AWS STAC. Set it to 0 "
                    "(zero) when using other search endpoints."
                )

    def geojson_from_bounds(self, bounds):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [bounds[0], bounds[1]],
                                [bounds[0], bounds[3]],
                                [bounds[2], bounds[3]],
                                [bounds[2], bounds[1]],
                                [bounds[0], bounds[1]],
                            ]
                        ],
                    },
                }
            ],
        }
        return geojson

    def construct_payload_aws(self):
        payload = {
            "collections": [self.collection],
            "datetime": f"{self.start_date}/{self.end_date}",
            "query": {
                "eo:cloud_cover": {"lt": self.max_cc + 0.0001},
                "sentinel:data_coverage": {"gt": self.min_dc - 0.0001},
            },
            "intersects": self.geojson["features"][0]["geometry"],
            "limit": self.items_limit,
            "fields": {
                "include": [
                    "id",
                    "properties.datetime",
                    "properties.eo:cloud_cover",
                    "properties.sentinel:data_coverage",
                    "properties.sentinel:product_id",
                ],
            },
            "sortby": [
                {"field": "properties.datetime", "direction": "asc"},
            ],
        }

        if self.tile_id is not None:
            utm_zone = re.findall(r"\d+", self.tile_id)[0]
            latitude_band = re.findall("[a-zA-Z]+", self.tile_id)[0][0]
            grid_square = re.findall("[a-zA-Z]+", self.tile_id)[0][1:3]

            payload["query"]["sentinel:utm_zone"] = ({"eq": utm_zone},)
            payload["query"]["sentinel:latitude_band"] = ({"eq": latitude_band},)
            payload["query"]["sentinel:grid_square"] = {"eq": grid_square}

        return payload

    def validate_response(self):
        try:
            r = self.response.json()
            if "numberMatched" in r.keys():  # AWS response
                items_found = r["numberMatched"]
                items_returned = r["numberReturned"]
            elif "value" in r.keys():  # Creodias ODATA response
                items_found = (
                    len(r["value"]) - 1
                )  # can't find out total results. -1 should
                # provoke an error whenever items_returned == self.items_limit
                items_returned = len(r["value"])
            else:
                raise KeyError()
        except (AttributeError, KeyError):
            raise HTTPError(
                url=self.endpoint,
                code=self.response.status_code,
                msg=f"Issue when making request to {self.endpoint}: error code "
                f"{self.response.status_code}. {self.response.text}",
                hdrs=None,
                fp=None,
            )

        if items_found > items_returned:
            raise ValueError(
                f"The number of items that match the search criteria seems to be "
                f"smaller than what the search API returns ({items_returned} items). "
                f"Consider searching for a smaller time interval or AOI."
            )

    @retry(exceptions=(JSONDecodeError, HTTPError), tries=9, delay=2, backoff=5)
    def make_request(self):
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
            "Accept": "application/geo+json",
        }

        if self.endpoint.startswith("https://earth-search.aws.element84.com/"):
            self.payload = self.construct_payload_aws()
            self.response = requests.post(
                self.endpoint, headers=headers, json=self.payload
            )
        else:
            self.payload = self.construct_payload_creodias()
            self.response = requests.get(
                self.endpoint, headers=headers, params=self.payload
            )

        self.validate_response()

        return self.response


def search_window_from_isodate(date_isoformat):
    start_date = str(date_isoformat)
    end_date = str(date.fromisoformat(str(date_isoformat)) + timedelta(days=1))

    return (
        start_date,
        end_date,
    )


def search_and_select_product(start_date, end_date, bounds):
    # search data using earth-search STAC; select product with bigger overlap with bounds in case of
    # multiple products in time window

    # search for AOI
    s2_query = S2Query(
        start_date=start_date,
        end_date=end_date,
        bounds=bounds,
        collection="sentinel-s2-l2a",
    )
    s2_query.make_request()
    response_json = s2_query.response.json()

    # get product id(s) for features
    returned_features = response_json["features"]
    if len(returned_features) == 0:
        raise ValueError(
            f"No product for the given time window ({start_date}, {end_date})."
        )
    if len(returned_features) == 1:
        feat = returned_features[0]
    else:
        # take the product that has the larger overlap with bounds
        warnings.warn(
            f"The search returned more than 1 product for the given time window ("
            f"{start_date}, {end_date}). Only the product with the larger overlap with "
            f"the AOI will be used."
        )
        polygon = geo.polygon_from_bounds(bounds)

        overlap_area = []
        for feat in returned_features:
            scene_geometry = shapely.geometry.shape(feat["geometry"])
            overlap_area.append(polygon.intersection(scene_geometry).area)

        # choose product with bigger overlap
        feat = returned_features[np.argmax(np.array(overlap_area))]

    product_id = feat["properties"]["sentinel:product_id"]
    # e.g. S2B_MSIL2A_20230217T101029_N0509_R022_T32UPU_20230217T125054

    return product_id


def get_metadata_local(product_path):
    # Get image bands from metadata file on top level of product
    metadata_path = Path(product_path) / "MTD_MSIL2A.xml"
    with open(metadata_path, "r") as f:
        mtd_xml = f.read()

    return mtd_xml


def get_metadata_google(product_url):
    # Get image bands from metadata file on Google Cloud Storage
    metadata_url = f"{product_url}/MTD_MSIL2A.xml"
    mtd_response = requests.get(metadata_url)
    mtd_response.raise_for_status()
    mtd_xml = mtd_response.text

    return mtd_xml


def get_img_path_mtd(mtd_xml, band_name):
    mtd_dict = xmltodict.parse(mtd_xml)
    try:
        gran_list = mtd_dict["n1:Level-2A_User_Product"]["n1:General_Info"][
            "Product_Info"
        ]["Product_Organisation"]["Granule_List"]
        has_regular_structure = True
    except KeyError:
        try:
            gran_list = mtd_dict["n1:Level-2A_User_Product"]["n1:General_Info"][
                "L2A_Product_Info"
            ]["L2A_Product_Organisation"]["Granule_List"]
            has_regular_structure = False
        except KeyError:
            raise KeyError(f"Unexpected metadata structure: {mtd_dict}.")

    if len(gran_list) == 1:
        if has_regular_structure:
            fps = gran_list["Granule"]["IMAGE_FILE"]
        else:
            fps = gran_list["Granule"]["IMAGE_FILE_2A"]
        img_format = gran_list["Granule"]["@imageFormat"]
    elif len(gran_list) > 1:
        img_formats = [g["Granule"]["@imageFormat"] for g in gran_list]
        if len(set(img_formats)) == 1:
            img_format = img_formats[0]
        else:
            raise ValueError("conflicting image formats found in metadata")
        fps = []
        for gran_info in gran_list:
            if has_regular_structure:
                fps.extend(gran_info["Granule"]["IMAGE_FILE"])
            else:
                fps.extend(gran_info["Granule"]["IMAGE_FILE_2A"])
    else:
        raise ValueError(f"Unexpected metadata structure: {mtd_dict}.")

    band_options = [fp for fp in fps if f"_{band_name}_" in fp]

    idx_finest_res = 0  # first if none of the ifs apply
    if len(band_options) > 1:
        # fetch resolution
        resolutions = [float(b.split("/R")[1].split("m/")[0]) for b in band_options]

        if 10 in resolutions:
            idx_finest_res = int(np.where(np.array(resolutions) == 10)[0])
        elif 20 in resolutions:
            idx_finest_res = int(np.where(np.array(resolutions) == 20)[0])
        elif 60 in resolutions:
            idx_finest_res = int(np.where(np.array(resolutions) == 60)[0])
        else:
            raise ValueError(f"None of the resolutions work: {resolutions}")

    if img_format == "JPEG2000":
        file_path = band_options[idx_finest_res] + ".jp2"
    else:
        raise NotImplementedError(f"Image format {img_format} not supported (yet).")

    return file_path


def get_offset_mtd(mtd_xml):
    mtd_dict = xmltodict.parse(mtd_xml)
    try:
        offset_list = mtd_dict["n1:Level-2A_User_Product"]["n1:General_Info"][
            "Product_Image_Characteristics"
        ]["BOA_ADD_OFFSET_VALUES_LIST"]["BOA_ADD_OFFSET"]
        has_regular_structure = True
    except KeyError:
        try:
            offset_list = mtd_dict["n1:Level-2A_User_Product"]["n1:General_Info"][
                "L2A_Product_Image_Characteristics"
            ]["BOA_ADD_OFFSET_VALUES_LIST"]["BOA_ADD_OFFSET"]
            has_regular_structure = False
        except KeyError:
            raise KeyError(f"Unexpected metadata structure: {mtd_dict}.")

    # check if all boa offsets the same and convert ot float value
    add_offsets = [item["#text"] for item in offset_list]
    if len(set(add_offsets)) == 1:  # all same
        try:
            add_offset = float(add_offsets[0])
            raise_error = False
        except TypeError:
            raise_error = True
    else:
        raise_error = True

    if raise_error:
        raise NotImplementedError(
            f"Unexpected behaviour. All offsets {add_offsets} expected to be "
            f"the same and convertible to float values. They are not."
        )
    else:
        return add_offset


def band_path_local(product_path, band_name):
    mtd_xml = get_metadata_local(product_path)
    img_file = get_img_path_mtd(mtd_xml, band_name)
    full_img_path = f"{product_path}/{img_file}"

    return full_img_path


def band_path_google(product_id, band_name):
    # get image file path from metadata
    tile_id = product_id.split("_")[5][1:]
    utm_zone = tile_id[:2]
    latitude_band = tile_id[2]
    grid_square = tile_id[3:]
    product_url = (
        f"https://storage.googleapis.com/gcp-public-data-sentinel-2/L2/tiles/"
        f'{utm_zone}/{latitude_band}/{grid_square}/{product_id.rstrip(".SAFE")}.SAFE'
    )
    mtd_xml = get_metadata_google(product_url)
    img_file = get_img_path_mtd(mtd_xml, band_name)
    full_img_path = f"{product_url}/{img_file}"

    return full_img_path


def offset_scale_local(product_path):
    mtd_xml = get_metadata_local(product_path)
    add_offset = get_offset_mtd(mtd_xml)
    scale_factor = 1 / QUANTIFICATION_VALUE

    return add_offset, scale_factor


def offset_scale_google(product_id):
    # get image file path from metadata
    tile_id = product_id.split("_")[5][1:]
    utm_zone = tile_id[:2]
    latitude_band = tile_id[2]
    grid_square = tile_id[3:]
    product_url = (
        f"https://storage.googleapis.com/gcp-public-data-sentinel-2/L2/tiles/"
        f'{utm_zone}/{latitude_band}/{grid_square}/{product_id.rstrip(".SAFE")}.SAFE'
    )
    mtd_xml = get_metadata_google(product_url)
    add_offset = get_offset_mtd(mtd_xml)
    scale_factor = 1 / QUANTIFICATION_VALUE

    return add_offset, scale_factor


def read_product_s2(
    product,
    bounds,
    bands_needed=[
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B11",
        "B12",
        "B8A",
        "SCL",
    ],
):
    # reads bands from S-2 product
    if Path(product).is_dir():
        is_local_product = True
        product_id = Path(product).name
    else:
        is_local_product = False
        product_id = product

    data_orig = {}
    for band_name in bands_needed:
        # get path to image band
        if is_local_product:
            img_path = band_path_local(product, band_name)
            add_offset, scale_factor = offset_scale_local(product)
        else:
            img_path = band_path_google(product_id, band_name)
            add_offset, scale_factor = offset_scale_google(product_id)

        if band_name == "SCL":
            add_offset, scale_factor = 0, 1

        # read image with bounds as reading window
        img, transform, img_epsg = geo.read_bounds(
            img_path, bounds, bounds_epsg=4326, snap_pixel_size=20
        )

        # conversion to reflectances
        img = (img.squeeze() + add_offset) * scale_factor

        data_orig[band_name] = img.astype("float32")

    # 20m band, e.g. B12: resample to 10m (by efficient numpy repeat)
    data = {}
    for band_name, img in data_orig.items():
        if band_name in S2_L2A_RESOLUTIONS_BY_RES["R10m"]:
            data[band_name] = img
        else:
            if band_name in S2_L2A_RESOLUTIONS_BY_RES["R20m"]:
                data[band_name] = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)

                # modify transform
                transform = list(transform)
                transform[0] = 10
                transform[4] = -10
                transform = rasterio.Affine(
                    transform[0],
                    transform[1],
                    transform[2],
                    transform[3],
                    transform[4],
                    transform[5],
                )
            else:
                raise NotImplementedError(
                    f"band_name {band_name} not supported (yet) because "
                    f"resampling is not handled yet."
                )

    return data, transform, img_epsg


def date_from_product_id(product_id):
    date_str = product_id.split("_")[2]
    isodate = str(datetime.datetime.strptime(date_str[:8], "%Y%m%d"))[:10]

    return isodate
