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
[`astra-toolbox`](https://astra-toolbox.com/) to perform Feldkamp (FDK)
reconstruction algorithm.

### AI Denoising

[Two AI denoising
models](https://github.com/brudfors/monai-dl-cbct/tree/main) are added
in the reconstruction pipeline:
- A sinogram denoising model which denoise the sinogram slice by slice
- A volumetric denoising model which denoise the reconstructed volume

### 3D Image Segmentation

MONAI Full-body CT Segmentation model.

### Visualization

OHIF and 3D Slicer.


## Installation

Build image:
./dev_container build --docker_file applications/cbct_recon/Dockerfile --img holohub-cbct-recon:latest

Launch image:
./dev_container launch --img holohub-cbct-recon
