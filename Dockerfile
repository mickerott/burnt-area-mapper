FROM ghcr.io/osgeo/gdal:ubuntu-small-3.6.2

COPY requirements.txt /python-app/requirements.txt
COPY config.yaml /python-app/config.yaml
COPY ./src /python-app/src

RUN apt-get update \
    && apt-get -y install python3-pip \
    && apt-get install nano
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /python-app/requirements.txt
