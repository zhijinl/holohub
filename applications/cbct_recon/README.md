# Cone-Beam Computed Tomography Reconstruction Pipeline

## Introduction

This example demonstrates an end-to-end pipeline of Cone-Beam Computed
Tomography (CBCT) reconstruction using Holoscan. The pipeline consists
of the following modules:
1. **Data streaming**: streaming of CBCT projection data from an
   independent device (a stand-alone workstation for example) to the
   device running the Holoscan pipeline (an IGX device for
   example). This is to simulate a realistic clinical workflow, where
   projection data is acquired from a CBCT scanner, and then stream
   out to a side-car device for subsequent processing.
2. **CBCT reconstruction**: reconstruct a 3D volume from received CBCT
   projection data.
3. **AI denoising**: apply UNet-based denoising on either the received
   CBCT projection data, or the reconstructed 3D volume, or both, in
   order to improve the image quality.
4. **3D image segmentation**: apply Deep Learning-based image segmentation
   models on reconstructed 3D volume to segment key anatomies, to
   facilitate further analysis by clinicians.
5. **Visualization**: visualize both the reconstructed 3D volume, as
   well as the segmented anatomies in 3D.

## Implementation

Different modules are implemented as follows.

### Data Streaming

Using advanced networking operator? DDS?

### CBCT Reconstruction

A custom operator was implemented, wrapping APIs of
[`astra-toolbox`](https://astra-toolbox.com/) to perform on-line
Feldkamp (FDK) reconstruction algorithm.

### AI Denoising

[Two AI denoising
models](https://github.com/brudfors/monai-dl-cbct/tree/main) are added
in the reconstruction pipeline:
- A sinogram denoising model which denoise the sinogram slice by slice
- A volumetric denoising model which denoise the reconstructed volume

### 3D Image Segmentation

MONAI Full-body CT Segmentation model.

### Visualization

OHIF viewer.


## Build and Launch Sample Apps

Build images:

1. Build docker image for reconstruction and denoising
```
./dev_container build --docker_file applications/cbct_recon/recon/Dockerfile --img holohub-cbct-recon:latest
```

2. Build docker image for segmentation and visualization
```
./dev_container build --docker_file applications/cbct_recon/vis/Dockerfile --img holohub-cbct-vis:latest
```

Run applications:

1. Launch docker container for reconstruction and denoising
```
./dev_container launch --img holohub-cbct-recon
```

2. Launch docker container for segmentation and visualization in a
separate terminal
```
./dev_container launch --img holohub-cbct-vis
```

3. Launch reconstruction and denoising app
```
# In holohub-cbct-recon container
python3 applications/cbct_recon/recon/cbct_recon.py
```
After this, the `holohub-cbct-vis` container will receive the
reconstructed volume.

4. Visualize and segment reconstructed volume

Open browser and go to `http://127.0.0.1:8000/ohif/` to access OHIF
viewer to visualize and segment the reconstructed volume with MONAI
Label.
