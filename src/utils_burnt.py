from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib import pyplot as plt
from rasterio import plot as rasterplot
from skimage import exposure
from skimage.morphology import (
    binary_opening,
    binary_closing,
    binary_erosion,
    binary_dilation,
)


def segment_burnt_area_dnbr(bam):
    # segment burnt area: morphological filtering of raster
    bam = binary_opening(bam)
    bam = binary_closing(bam)
    bam = binary_erosion(bam)
    bam = binary_erosion(bam)
    bam = binary_dilation(bam)
    bam = binary_dilation(bam)

    return bam


def export_burnt_map(raster_path, vector_file, mmu, image_out, date, place_name=None):
    # export map

    # read files
    with rasterio.open(raster_path) as src:
        src_crs = src.crs
        raster_extent = [src.bounds[0], src.bounds[2], src.bounds[1], src.bounds[3]]
        channel_list = []
        for channel in range(1, src.count + 1):
            channel_list.append(src.read(channel))

    vector_file = gpd.read_file(vector_file)
    vector_file = vector_file.to_crs(src_crs)

    # clean vector file
    vector_file["area"] = vector_file.geometry.area
    vector_file = vector_file.loc[vector_file.area >= mmu]

    # improve image contrast
    img = np.stack(channel_list, axis=0)
    img = np.clip(img, 0, 1)
    img = exposure.rescale_intensity(
        img,
        in_range=(
            np.nanpercentile(img, 10),
            np.nanpercentile(img, 90),
        ),
        out_range=(
            0,
            1,
        ),
    )

    fig, ax = plt.subplots(
        figsize=(
            14,
            10,
        ),
        dpi=150,
    )

    # plot raster
    rasterplot.show(img, extent=raster_extent, ax=ax)

    # add title and axis labels
    crs_name = vector_file.crs.name
    if place_name is not None:
        title = f"Burnt Area Map from {date} for {place_name} ({crs_name})"
    else:
        title = f"Burnt Area Map from {date} ({crs_name})"
    ax.set_title(title)
    ax.set_xlabel("East")
    ax.set_ylabel("North")

    # export image without vectors
    plt.savefig(image_out)

    # plot vector file and export
    vector_file.plot(ax=ax, facecolor="none", edgecolor="cyan", alpha=0.7)
    image_out = (
        Path(image_out).parent
        / f"{Path(image_out).stem}_segmented{Path(image_out).suffix}"
    )
    plt.savefig(image_out)

    print(f"Saved Burnt Area Map to: {Path(image_out).resolve()}")
