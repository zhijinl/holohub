# CBCT Reconstruction App

## Introduction

This is the reconstruction part of the CBCT Reconstruction
pipeline. In this part, you will be able to build a Holoscan sample
application that performs on-line [FDK (Feldkamp, Davis and Kress)
CBCT reconstruction
algorithm](https://opg.optica.org/josaa/fulltext.cfm?uri=josaa-1-6-612&id=996). Specifically,
this app contains the following components:
- First, we start with a data stream component, where CBCT projection
  images will be streamed one-by-one, either from local disk, or from
  a remote machine, to where the reconstruction will be performed.
- Before the reconstruction step, an AI denoising model will be
  applied, to each CBCT projection image, for noise removal.
- Then the core reconstruction component will receive the denoised
  projection images and perform CBCT reconstruction. The
  reconstruction will be done in an __online__ fashion, meaning that
  the reconstruction algorithm will not wait until all projection
  images arrive to start reconstructing. Instead, reconstruction
  starts immediately as soon as the first projection image arrives,
  and then we incrementally update the volume as rest of the
  projection images arrives. In theory, such an on-line reconstruction
  can be implemented for any linear reconstruction algorithms.
- After reconstruction, another denoising model will be applied, this
  time on the reconstructed CBCT volume.
- Both the original and denoise 3D volumes will be pushed to the
  Orthanc DICOM server (part of the [visualization
  app](../vis/README.md)) for further analysis and visualization.

> **_NOTE:_**  You need to build and launch the visualization app
> first, otherwise the Orthanc DICOM server will not be ready, therefore
> the reconstruction app will not be able to push the reconstructed
> volumes to the DICOM server.

## Build and Run the App

We can use the `dev_container` tool provided in holohub root directory
to build the image for running this app:
```
./dev_container build --docker_file applications/cbct_recon/recon/Dockerfile --img holohub-cbct-recon
```

Launch a container from the built image:
```
./dev_container launch --img holohub-cbct-recon
```

Before running the app, make sure that you have the sample sinogram
data and the denoising models under the `data` folder. TODO: add the
proper way to pull data, from NGC?

To run the reconstruction app:
```
python3 applications/cbct_recon/recon/cbct_recon.py
```
The reconstruction app is configured using the
[cbct_recon.yaml](cbct_recon.yaml) file. You can modify various fields
in this file to change settings such as projection geometry. Check
this file for detailed documentation.

When app starts to run, you will be able to visualze the on-line FDK
reconstruction process, where the anatomical details of the central
slice of the reconstructed volume becomes more and more clear (see
the video below).

<video src="figs/fdk-recon-low.mp4" width="640" height="640" controls></video>
