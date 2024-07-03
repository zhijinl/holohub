# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


ARG BASE_IMAGE
ARG GPU_TYPE

FROM ${BASE_IMAGE} AS base

ARG DEBIAN_FRONTEND=noninteractive


# --------------------------------------------------------------------------
#
# Holohub run setup
#

RUN mkdir -p /tmp/scripts
COPY run /tmp/scripts/
RUN mkdir -p /tmp/scripts/utilities
COPY utilities/holohub_autocomplete /tmp/scripts/utilities/
RUN chmod +x /tmp/scripts/run
RUN /tmp/scripts/run setup

# Enable autocomplete
RUN echo ". /etc/bash_completion.d/holohub_autocomplete" >> /etc/bash.bashrc

# - This variable is consumed by all dependencies below as an environment variable (CMake 3.22+)
# - We use ARG to only set it at docker build time, so it does not affect cmake builds
#   performed at docker run time in case users want to use a different BUILD_TYPE
ARG CMAKE_BUILD_TYPE=Release


# Build & install astra-toolbox

RUN git clone https://github.com/astra-toolbox/astra-toolbox.git /astra-toolbox

RUN apt update && apt install -y autotools-dev \
    autoconf \
    libtool

RUN python -m pip install setuptools==58.2.0
RUN python -m pip install Cython six scipy

RUN cd /astra-toolbox/build/linux/ && ./autogen.sh && ./configure --with-cuda=/usr/local/cuda \
    --with-python \
    --with-install-type=module && \
    make && make install

RUN echo 'alias ll="ls -lah --color"' >> ~/.bashrc


# Install tomosipo and ts_algorithms
RUN pip install git+https://github.com/ahendriksen/tomosipo@develop
RUN pip install git+https://github.com/ahendriksen/ts_algorithms.git

RUN pip install scikit-image


# Install pytorch from jetson pre-built wheels
RUN cd /tmp && wget https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl
RUN cd /tmp && pip install torch-2.2.0a0+6a974be.nv23.11-cp310-cp310-linux_aarch64.whl
