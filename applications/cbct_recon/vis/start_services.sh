#! /bin/bash
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

# Launch Orthanc
Orthanc /workspace/orthanc/orthanc_config.json &

# Download MONAI Bundle app for MONAI Label
monailabel apps --download --name monaibundle --output /workspace/monailabel

# Start MONAI Label server
export MONAI_LABEL_DICOMWEB_USERNAME=orthanc
export MONAI_LABEL_DICOMWEB_PASSWORD=orthanc

monailabel start_server \
           --app /workspace/monailabel/monaibundle \
           --studies http://127.0.0.1:8042/dicom-web \
           --conf models wholeBody_ct_segmentation
