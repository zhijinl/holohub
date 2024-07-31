FROM nvcr.io/nvidia/pytorch:24.07-py3

ARG DEBIAN_FRONTEND=noninteractive

# Build Orthanc
RUN mkdir -p /tmp/orthanc && cd /tmp/orthanc/ && \
    wget https://orthanc.uclouvain.be/downloads/sources/orthanc/Orthanc-1.12.4.tar.gz -O orthanc && \
    tar xzvf orthanc

RUN mkdir -p /tmp/orthanc/build && \
    cd /tmp/orthanc/build && \
    cmake -DSTATIC_BUILD=ON -DCMAKE_BUILD_TYPE=Release ../Orthanc-1.12.4/OrthancServer/ && \
    make -j4 && \
    make install

# Build the DICOMWeb plugin
RUN mkdir -p /tmp/orthanc/plugins && cd /tmp/orthanc/plugins && \
    wget https://orthanc.uclouvain.be/downloads/sources/orthanc-dicomweb/OrthancDicomWeb-1.17.tar.gz \
    -O OrthancDicomWeb-1.17.tar.gz && \
    tar xzvf OrthancDicomWeb-1.17.tar.gz

RUN mkdir -p /tmp/orthanc/plugins/build-dicomweb && cd /tmp/orthanc/plugins/build-dicomweb && \
    cmake -DSTATIC_BUILD=ON -DCMAKE_BUILD_TYPE=Release ../OrthancDicomWeb-1.17 && \
    make -j4 && \
    make install

# Install Orthanc dependency
RUN apt update && apt install -y tzdata

# Install MONAI Label
RUN pip install monailabel
