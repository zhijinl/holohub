# Use a suitable base image
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    build-essential \
    uuid-dev \
    unzip \
    cmake \
    chromium-browser

# Set up Python environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY visu/apps/ /opt/apps
COPY visu/start_monailabel.sh /opt/start_monailabel.sh
# Install Python dependencies
COPY visu/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Install Orthanc and DICOMWeb plugin
RUN mkdir -p /tmp/orthanc && cd /tmp/orthanc/ && \
    wget https://orthanc.uclouvain.be/downloads/sources/orthanc/Orthanc-1.12.2.tar.gz -O orthanc && \
    tar xzvf orthanc

RUN mkdir -p /tmp/orthanc/build && \
    cd /tmp/orthanc/build && \
    cmake -DSTATIC_BUILD=ON -DCMAKE_BUILD_TYPE=Release ../Orthanc-1.12.2/OrthancServer/ && \
    make -j && \
    make install

RUN mkdir -p /tmp/orthanc/plugins && cd /tmp/orthanc/plugins && \
    wget https://orthanc.uclouvain.be/downloads/sources/orthanc-dicomweb/OrthancDicomWeb-1.16.tar.gz \
    -O OrthancDicomWeb-1.16.tar.gz && \
    tar xzvf OrthancDicomWeb-1.16.tar.gz

RUN mkdir -p /tmp/orthanc/plugins/build-dicomweb && cd /tmp/orthanc/plugins/build-dicomweb && \
    cmake -DSTATIC_BUILD=ON -DCMAKE_BUILD_TYPE=Release ../OrthancDicomWeb-1.16 && \
    make -j && \
    make install

# Add configuration file
COPY orthancDocker/orthanc/orthanc-config.json /etc/orthanc/orthanc.json

# Copy the visualization script
COPY visu/visu2_no_upload_inside_docker.py /opt/visu2_no_upload.py

# Make the script executable
RUN chmod +x /opt/visu2_no_upload.py

# Expose necessary ports
EXPOSE 8042 4242

# Set the entrypoint to run the visualization script
ENTRYPOINT ["/opt/venv/bin/python", "/opt/visu2_no_upload.py"]


