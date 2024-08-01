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

- **Data Streaming**: TODO, using advanced networking operator? DDS?
- **CBCT Reconstruction**: A custom operator was implemented, wrapping
  APIs of [`astra-toolbox`](https://astra-toolbox.com/) to perform
  [FDK (Feldkamp, Davis and Kress) CBCT reconstruction
  algorithm](https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-1-6-612&id=996). An
  online version of the FDK algorithm was implemented, meaning that
  the reconstruction starts as soon as the first projection images
  arrives, and then incrementally update the reconstructed volume as
  rest of the projection images arrives.
  - **AI Denoising**: [Two AI denoising
  models](https://github.com/brudfors/monai-dl-cbct/tree/main) are
  added in the reconstruction pipeline:
  1. A sinogram denoising model which denoise the sinogram slice by
     slice. **NOTE: TODO, NOT FINISHED, need to train a slice-by-slice
     model!!**
  2. A volumetric denoising model which denoise the reconstructed
     volume.
- **Visualization and 3D Image Segmentation**: Orthanc was used as a
  DICOM-web server and OHIF was used as viewer. CBCT volume
  segmentation can be perform using MONAI Label inside OHIF
  viewer. The [whole-body CT Segmentation
  model](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation)
  from MONAI Model Zoo was used to segment anatomis in reconstructed
  CBCT volumes.

## Build and Launch Sample Apps

> **_NOTE:_**  Please build and launch the visualization app first, if
> you want to visualize the reconstructed data in OHIF viewer and
> perform segmentation using MONAI Label. If the visualization app is
> not launched, the Orthanc DICOM server will not be ready, therefore
> the reconstruction app will not be able to push the reconstructed
> volumes to the DICOM server for OHIF to display.

Please refer to the following individual sections for how to build and
launch the reconstruction and visualization apps.

- [Build and launch the reconstruction app](recon/README.md)
- [Build and launch the visualization app](vis/README.md)
