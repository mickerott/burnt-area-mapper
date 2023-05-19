import argparse
import shutil
import warnings
from copy import copy, deepcopy
from pathlib import Path

import geopandas as gpd
import numpy as np
import yaml
from matplotlib import pyplot as plt
from rasterio.features import shapes
from shapely.geometry import shape

import geo
import sentinel2
import utils_burnt


def burnt_area_mapper(
    prefire_product,
    postfire_product,
    ll,
    ur,
    th=0.1,
    mmu=1000000,
    export_intermediate_results=False,
    place_name=None
):
    bands_needed = ["B08", "B12"]  # bands needed for NBR calculation
    bands_needed += ["B03", "B04"]  # bands needed for plotting an image as background for the map

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

        data_images.append(data)

    # BURNT AREA ALGORITHM #

    # 1. compute dNBR
    print(f"Computing dNBR...")
    data_nbr = []
    for data in data_images:
        # compute NBR
        nbr = (data["B08"] - data["B12"]) / (data["B08"] + data["B12"])
        data_nbr.append(nbr)

        if export_intermediate_results:
            # plot NBR
            nbr_clipped = np.clip(
                nbr,
                a_min=np.nanpercentile(nbr.flatten(), 10),
                a_max=np.nanpercentile(nbr.flatten(), 90),
            )
            plt.imshow(nbr_clipped)
            plt.show()

    delta_nbr = data_nbr[0] - data_nbr[1]

    if export_intermediate_results:
        # plot dNBR
        delta_nbr_clipped = np.clip(
            delta_nbr,
            a_min=np.nanpercentile(delta_nbr.flatten(), 10),
            a_max=np.nanpercentile(delta_nbr.flatten(), 90),
        )
        plt.imshow(delta_nbr_clipped)
        plt.colorbar()
        plt.show()

    # 2. segment burnt area: apply threshold, morphological filtering, conversion to vector
    print(f"Segmenting burnt area...")
    if export_intermediate_results:
        out_path = "../out.tif"
        geo.export_to_tiff_w_transform(
            delta_nbr[..., np.newaxis],
            img_epsg,
            transform,
            out_path,
            compress="LZW",
            nodata=None,
            raise_error=False,
        )

        # nbr greyscale with threshold applied
        bam = deepcopy(delta_nbr)
        bam[delta_nbr < th] = np.nan
        plt.imshow(bam)
        plt.colorbar()
        plt.show()
        out_path = f"./out_th{th}.tif"
        geo.export_to_tiff_w_transform(
            bam[..., np.newaxis],
            img_epsg,
            transform,
            out_path,
            compress="LZW",
            nodata=None,
            raise_error=False,
        )

        # apply threshold and export as binary mask
        bam = deepcopy(delta_nbr)
        bam[delta_nbr < th] = np.nan
        bam[~np.isnan(bam)] = 1
        bam[np.isnan(bam)] = 0
        out_path = f"./out_th{th}_flat.tif"
        geo.export_to_tiff_w_transform(
            bam[..., np.newaxis],
            img_epsg,
            transform,
            out_path,
            compress="LZW",
            nodata=None,
            raise_error=False,
        )

    bam = utils_burnt.segment_burnt_area_dnbr(delta_nbr, th)

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
    for data in data_images:
        # export of false color image (if all stack bands have been loaded and are available in
        # data)
        stack_bands = ["B08", "B04", "B03"]
        out_suffix = "_fc"
        all_bands_in_data = np.all(np.isin(stack_bands, list(data.keys())))
        if all_bands_in_data:
            img_stack = np.stack([data[b] for b in stack_bands], axis=-1)
            out_path = f"./{product_id}{out_suffix}.tif"
            geo.export_to_tiff_w_transform(
                img_stack,
                img_epsg,
                transform,
                out_path,
                compress="LZW",
                nodata=None,
                raise_error=False,
            )
            raster_path = deepcopy(
                out_path
            )  # last in list (postfire product) as background for map

    image_out = "burnt_area_false_color.jpg"
    utils_burnt.export_burnt_map(raster_path, vector_file, mmu, image_out, place_name)
    print(
        f"Processing finished."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default="./../config.yaml", help="config YAML"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    prefire_product = config["prefire_product"]
    postfire_product = config["postfire_product"]
    ll = config["aoi"]["ll"]
    ur = config["aoi"]["ur"]
    th = config["segmentation"]["th"]
    mmu = config["segmentation"]["mmu"]
    export_intermediate_results = (
        config["export_intermediate_results"]
        if "export_intermediate_results" in config.keys()
        else False
    )
    place_name = (
        config["aoi"]["place_name"]
        if "place_name" in config["aoi"].keys()
        else None
    )

    # run automatic burnt area mapper
    burnt_area_mapper(
        prefire_product, postfire_product, ll, ur, th, mmu, export_intermediate_results, place_name
    )
