# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
