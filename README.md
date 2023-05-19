# Automatic Burnt Area Mapper

Simple multi-temporal burnt area detection algorithm using Sentinel-2.

## Inputs
All parameters need to be defined in a YAML file named `config.yaml` that is to be passed as a parameter when running `main.py`. 

### Main parameters
* `prefire_product`: Sentinel-2 scene BEFORE the fire event; see next section for details on the supported "formats"
* `postfire_product`: Sentinel-2 scene AFTER the fire event; see next section for details on the supported "formats"
* `aoi`:
  * `ll`: lower-left (i.e. South-West) corner of bounding box (Northing and Easting value in decimal degrees as a list; e.g. [48,11.5] for Strasslach near Munich)
  * `ur`: upper-right (i.e. North-East) corner of bounding box

### Input options: Sentinel-2 products
  * **DATE** (default): string in ISO-format (`YYYY-MM-dd` format, e.g. `2023-03-05`)
  * **ARCHIVE/FOLDER**: zipped or unpacked Sentinel-2 L2A product, as disseminated by esa (e.g. file name `S2A_MSIL2A_20230305T000221_N0509_R030_T55HGD_20230305T024055.SAFE.zip`)


## Algorithm
The workflow employs a simple multi-temporal burnt area detection algorithm. The (default) workflow consists of the following steps:
* search for Sentinel-2 L2A scenes using earth-search STAC on AWS
* download products to disk from Google Cloud
* reads images from disk for the given bounding box
* compute a simple [multi-temporal burn severity index](https://un-spider.org/advisory-support/recommended-practices/recommended-practice-burn-severity/in-detail/normalized-burn-ratio) which includes:
  * computation of NBR for each date
  * computation of difference: `dNBR = prefireNBR - postfireNBR`
* export the result as a map

## Outputs

Main output is a map where the burnt area is overlayed over the postfire-dataset. Here is an example (burn scar boundaries in cyan):
![Burnt Area Map](/src/burnt_area_false_color_segmented.jpg "Burnt Area Map")

## How to run the Burnt Area Mapper using Docker (example):

Build container:
```
docker build -t bam .
```

Run container in interactive mode:
```
docker run -it bam bash
```

Launch Burnt Area Mapper script:
```
cd /python-app/src
python main.py --config /python-app/config.yaml
```
