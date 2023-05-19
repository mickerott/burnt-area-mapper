import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.mask
import shapely


def bounds_from_corners(ll, ur):
    left, bottom, right, top = ll[1], ll[0], ur[1], ur[0]

    return (
        left,
        bottom,
        right,
        top,
    )


def polygon_from_bounds(bounds):
    polygon = shapely.geometry.box(bounds[0], bounds[1], bounds[2], bounds[3])

    return polygon


def read_bounds(fp, bounds, bounds_epsg, snap_pixel_size=None):
    with rasterio.open(fp) as src:
        src_epsg = src.crs.to_epsg()

        # reproject to raster CRS
        gdf = gpd.GeoDataFrame({"geometry": [polygon_from_bounds(bounds)]})
        gdf.crs = bounds_epsg
        gdf = gdf.to_crs(src_epsg)
        polygon = gdf.geometry.iloc[0]

        if snap_pixel_size:
            # snap bounds to nearest multiple of snap_pixel_size (when you want to make sure
            # that arrays of different resolutions are aligned)
            x_min = np.rint(polygon.bounds[0] / snap_pixel_size) * snap_pixel_size
            y_min = np.rint(polygon.bounds[1] / snap_pixel_size) * snap_pixel_size
            x_max = np.rint(polygon.bounds[2] / snap_pixel_size) * snap_pixel_size
            y_max = np.rint(polygon.bounds[3] / snap_pixel_size) * snap_pixel_size
            bounds_snapped = (
                x_min,
                y_min,
                x_max,
                y_max,
            )
            polygon_snapped = polygon_from_bounds(bounds_snapped)
        else:
            polygon_snapped = polygon

        img, src_transform = rasterio.mask.mask(
            src, shapes=[polygon_snapped], all_touched=True, crop=True
        )

    return img, src_transform, src_epsg


def export_to_tiff_w_transform(
    array, epsg, transform, out_path, compress=None, nodata=None, raise_error=True
):
    """Saves numpy array with image data as GeoTIFF to local path
    :param array: 3-D numpy ndarray in the shape (H, W, C); height, width in first 2 dimensions;
    channels in 3rd dimension
    :param epsg: EPSG code of CRS (int)
    :param transform: rasterio transform
    :param out_path: filename of GeoTIFF
    :param compress: GeoTIFF compression technique (e.g. 'LZW')
    :param nodata: NoData value of GeoTIFF
    :param raise_error: whether to raise erro when file exists (or just warn)
    :return:
    """
    profile = {"driver": "GTiff", "tiled": True, "compress": compress}
    dtype = rasterio.dtypes.get_minimum_dtype(array)
    crs = rasterio.crs.CRS.from_epsg(epsg)
    height = array.shape[0]
    width = array.shape[1]
    count = array.shape[2]

    with rasterio.Env():
        profile.update(
            dtype=dtype,
            count=count,
            height=height,
            width=width,
            crs=crs,
            transform=transform,
            nodata=nodata,
        )

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with rasterio.open(out_path, "w", **profile) as dst:
                for band_idx in range(count):
                    arr_out = array[:, :, band_idx].astype(
                        rasterio.dtypes.get_minimum_dtype(array)
                    )
                    dst.write(arr_out, 1 + band_idx)
        except Exception as e:
            if raise_error:
                raise e
            else:
                warnings.warn(
                    f"Error when writing file {out_path}. Perhaps file already exists."
                )

    return out_path
