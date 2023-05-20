import argparse
import shutil
import warnings
from copy import copy, deepcopy
from pathlib import Path
from pprint import pprint

import geopandas as gpd
import numpy as np
import yaml
from matplotlib import pyplot as plt
from rasterio.features import shapes
from shapely.geometry import shape
from skimage.morphology import binary_dilation

import geo
import sentinel2
import utils
import utils_burnt


def burnt_area_mapper(
    prefire_product,
    postfire_product,
    ll,
    ur,
    algorithm_names=None,
    mask_clouds=True,
    th=0.1,
    mmu=1000000,
    min_confidence=0.67,
    export_intermediate_results=False,
    place_name=None,
):
    if algorithm_names is None:
        algorithm_names = ["NBR"]

    bands_needed = ["B08", "B12"]  # bands needed for NBR calculation
    bands_needed += [
        "B03",
        "B04",
    ]  # bands needed for plotting an image as background for the map
    if "MIRBI" in algorithm_names:
        bands_needed += ["B11"]
    if "BAIS2" in algorithm_names:
        bands_needed += ["B06", "B07", "B8A"]
    if mask_clouds:
        bands_needed += ["SCL"]

    # get image data and compute NBR for both prefire- and postfire-product
    bounds = geo.bounds_from_corners(ll, ur)

    data_images = []
    for product in [prefire_product, postfire_product]:
        # unpack if zipped product
        if product.endswith(".zip"):
            print("Extracting archive...")
            dataset_new = product.rstrip(".zip")
            shutil.unpack_archive(product, Path(product).parent)
            product = copy(dataset_new)

        if Path(product).is_dir():
            # determine product ID from product file name
            product_id = Path(product).name

            # get image data from product
            print(f"Reading data from product: {product_id}")
            data, transform, img_epsg = sentinel2.read_product_s2(
                product, bounds, bands_needed
            )
        else:
            # determine product ID using earth-search STAC
            try:
                search_window = sentinel2.search_window_from_isodate(
                    date_isoformat=product
                )
            except ValueError:
                raise ValueError(
                    f"Input dataset {product} seems to be neither a Sentinel-2 product, nor a "
                    f"valid ISO date."
                )

            print(f"Searching imagery for search window: {search_window}...")
            product_id = sentinel2.search_and_select_product(
                search_window[0], search_window[1], bounds
            )

            # get image data from product
            print(f"Reading data from product: {product_id}")
            data, transform, img_epsg = sentinel2.read_product_s2(
                product_id, bounds, bands_needed
            )

        postfire_date = sentinel2.date_from_product_id(product_id)

        data_images.append(data)

    # BURNT AREA ALGORITHM #
    algorithm_names = [
        a.lower() for a in algorithm_names
    ]  # later on, only lower-case names used

    # compute necessary indices and deltas for each of the algorithms in algorithm_names
    burnt_area_maps = []
    for algo_name in algorithm_names:
        # 1. compute delta of burnt area indices/vegetation indices (vi)
        print(f"Computing delta {algo_name.upper()}...")
        data_nbr = []
        for data in data_images:
            # compute burnt area index/vegetation index (vi)
            if algo_name == "nbr":
                vi = (data["B08"] - data["B12"]) / (data["B08"] + data["B12"])
            elif algo_name == "mirbi":
                vi = 10 * data["B12"] - 9.8 * data["B11"] + 2
                th = -0.15  # thresh differs from classical NBR
            elif algo_name == "bais2":
                b4 = data["B04"]
                b6 = data["B06"]
                b7 = data["B07"]
                b8a = data["B8A"]
                b12 = data["B12"]
                vi = (1 - np.sqrt((b6 * b7 * b8a) / b4)) * (
                    (b12 - b8a) / (np.sqrt(b12 + b8a)) + 1
                )
                th = -0.15  # thresh differs from classical NBR
            elif algo_name == "ndswiri":
                vi = (data["B11"] - data["B12"]) / (data["B11"] + data["B12"]) + data[
                    "B08"
                ]
                th = 0.12
            elif algo_name == "ndsri":
                vi = (data["B11"] - data["B04"]) / (data["B11"] + data["B04"])
                th = 0.1
            else:
                raise NotImplementedError(
                    f"Algorithm name unknown/not supported: {algo_name}"
                )

            # mask clouds
            if mask_clouds:
                # recode SCL
                mapping_dict_clouds = {
                    0: 1,
                    1: 1,
                    3: 1,
                    8: 1,
                    9: 1,
                    10: 1,
                    11: 1,
                    2: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                }  # True values/ones are considered INvalid
                mask = utils.recode_array_fast(
                    data["SCL"], mapping_dict=mapping_dict_clouds
                ).astype("bool")

                # dilate mask
                mask_buffer = 0  # e.g. 10 to 20
                for i in range(mask_buffer):
                    mask = binary_dilation(mask)

                # apply mask
                vi[mask] = np.nan

            data_nbr.append(vi)

            if export_intermediate_results:
                # plot NBR
                nbr_clipped = np.clip(
                    vi,
                    a_min=np.nanpercentile(vi.flatten(), 10),
                    a_max=np.nanpercentile(vi.flatten(), 90),
                )
                plt.imshow(nbr_clipped)
                plt.show()

        delta_nbr = data_nbr[0] - data_nbr[1]

        if export_intermediate_results:
            # export
            out_path = f"delta_{algo_name}.tif"
            geo.export_to_tiff_w_transform(
                delta_nbr[..., np.newaxis],
                img_epsg,
                transform,
                out_path,
                compress="LZW",
                nodata=None,
                raise_error=False,
            )

            # plot dNBR
            delta_nbr_clipped = np.clip(
                delta_nbr,
                a_min=np.nanpercentile(delta_nbr.flatten(), 10),
                a_max=np.nanpercentile(delta_nbr.flatten(), 90),
            )
            plt.imshow(delta_nbr_clipped)
            plt.colorbar()
            plt.show()

        # 2. apply threshold
        bam = deepcopy(delta_nbr)
        if algo_name in ["nbr", "ndswiri", "ndsri"]:
            bam[delta_nbr < th] = np.nan
        elif algo_name in ["mirbi", "bais2"]:
            bam[delta_nbr >= th] = np.nan
        else:
            raise NotImplementedError(
                f"Algorithm name unknown/not supported: {algo_name}"
            )

        bam[~np.isnan(bam)] = 1
        bam[np.isnan(bam)] = 0

        burnt_area_maps.append(bam)

    # 3. create confidence map and apply confidence threshold
    if len(burnt_area_maps) > 1:
        stack = np.stack(burnt_area_maps, axis=-1)
        confidence_map = np.nansum(stack, axis=-1)

        if export_intermediate_results:
            # export confidence map
            out_path = f"confidence_{'-'.join(algorithm_names)}.tif"
            geo.export_to_tiff_w_transform(
                confidence_map[..., np.newaxis],
                img_epsg,
                transform,
                out_path,
                compress="LZW",
                nodata=None,
                raise_error=False,
            )

        # apply confidence threshold: burnt area in final map
        bam_final = deepcopy(confidence_map)
        bam_final[confidence_map < (len(burnt_area_maps) * min_confidence)] = np.nan
        bam_final[confidence_map >= (len(burnt_area_maps) * min_confidence)] = 1
        bam_final[np.isnan(bam_final)] = 0
    else:
        bam_final = burnt_area_maps[0]

    # 4. segment burnt area: morphological filtering, conversion to vector
    print(f"Segmenting burnt area...")
    bam = utils_burnt.segment_burnt_area_dnbr(bam_final)

    if export_intermediate_results:
        plt.imshow(bam)
        plt.colorbar()
        plt.show()

        # export segmented burnt area as raster image
        out_path = f"./out_th{th}_flat_morph.tif"
        geo.export_to_tiff_w_transform(
            bam[..., np.newaxis],
            img_epsg,
            transform,
            out_path,
            compress="LZW",
            nodata=None,
            raise_error=False,
        )

    # convert raster mask to vector
    print(f"Converting raster mask to vectors...")
    shape_list = [
        (shape(s), v) for s, v in shapes(bam.astype("uint8"), transform=transform)
    ]
    gdf = gpd.GeoDataFrame(
        dict(zip(["geometry", "class"], zip(*shape_list))), crs=img_epsg
    )
    gdf = gdf.loc[gdf["class"] == 1]

    # export segmented burnt area as vector file
    vector_file = "burnt_area_segments.gpkg"
    try:
        gdf.to_file(vector_file, driver="GPKG")
    except PermissionError:
        warnings.warn(
            f"Error when writing file {vector_file}. Perhaps file already exists."
        )

    # for burnt area map background: export false color images
    print(f"Creating burnt area map...")

    # export of postfire false color image (if all stack bands have been loaded and are
    # available in data)
    stack_bands = ["B08", "B04", "B03"]
    out_suffix = "_fc"
    if mask_clouds:
        masks = []
    for i, data in enumerate(data_images):
        # creating mask for area that is NoData in any of the input images (e.g. clouds)
        if mask_clouds:
            # recode SCL
            mask = utils.recode_array_fast(
                data["SCL"], mapping_dict=mapping_dict_clouds
            ).astype("bool")
            masks.append(mask)

        if i == 1:  # only postfire image
            all_bands_in_data = np.all(np.isin(stack_bands, list(data.keys())))
            if all_bands_in_data:
                img_stack = np.stack([data[b] for b in stack_bands], axis=-1)

    cumulated_mask = np.any(np.stack(masks, axis=-1), axis=-1).astype(
        "bool"
    )  # True values
    # considered INvalid
    plt.imshow(cumulated_mask)
    plt.colorbar()
    plt.show()
    # apply mask and export
    nodata_value = -9999
    img_stack[cumulated_mask] = nodata_value
    img_stack_clipped = np.clip(
        img_stack,
        a_min=np.nanpercentile(img_stack.flatten(), 10),
        a_max=np.nanpercentile(img_stack.flatten(), 90),
    )
    plt.imshow(img_stack_clipped * 2)
    plt.show()
    out_path = f"./{product_id}{out_suffix}.tif"
    geo.export_to_tiff_w_transform(
        img_stack,
        img_epsg,
        transform,
        out_path,
        compress="LZW",
        nodata=nodata_value,
        raise_error=False,
    )
    raster_path = deepcopy(
        out_path
    )  # last in list (postfire product) as background for map
    image_out = "burnt_area_false_color.jpg"
    utils_burnt.export_burnt_map(
        raster_path, vector_file, mmu, image_out, postfire_date, place_name
    )
    print(f"Processing finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="./../config.yaml", help="config YAML"
    )
    args, _ = parser.parse_known_args()

    config_path = args.config

    print(f"Reading config file: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    pprint(config)

    prefire_product = config["prefire_product"]
    postfire_product = config["postfire_product"]
    ll = config["aoi"]["ll"]
    ur = config["aoi"]["ur"]
    algorithm_names = config["algorithm_names"]
    mask_clouds = config["mask_clouds"]
    th = config["segmentation"]["th"]
    mmu = config["segmentation"]["mmu"]
    min_confidence = config["min_confidence"]
    export_intermediate_results = (
        config["export_intermediate_results"]
        if "export_intermediate_results" in config.keys()
        else False
    )
    place_name = (
        config["aoi"]["place_name"] if "place_name" in config["aoi"].keys() else None
    )

    # run automatic burnt area mapper
    burnt_area_mapper(
        prefire_product,
        postfire_product,
        ll,
        ur,
        algorithm_names,
        mask_clouds,
        th,
        mmu,
        min_confidence,
        export_intermediate_results,
        place_name,
    )
