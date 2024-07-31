#! /bin/bash

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
