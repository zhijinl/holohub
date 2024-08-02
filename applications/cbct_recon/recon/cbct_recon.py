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


import os
import torch
import random
import skimage
import numpy as np
import tomosipo as ts

from copy import deepcopy
from ts_algorithms import fdk

import pydicom
import pydicom._storage_sopclass_uids
from pynetdicom import AE, debug_logger
from pynetdicom.sop_class import CTImageStorage

from holoscan import as_tensor
from holoscan.resources import CudaStreamPool
from holoscan.conditions import CountCondition, BooleanCondition
from holoscan.operators import HolovizOp, InferenceOp
from holoscan.core import Application, Operator, OperatorSpec, ConditionType


try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "CuPy must be installed to run this example. See "
        "https://docs.cupy.dev/en/stable/install.html"
    )


def convert_to_dcm(
        inp,
        pixel_spacing_x,
        pixel_spacing_y,
        slice_thickness,
):
    dcm_ds = pydicom.Dataset()

    dcm_ds.AccessionNumber = "REMOVED"
    dcm_ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

    # Meta data
    meta_ds = pydicom.Dataset()
    meta_ds.MediaStorageSOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    meta_ds.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    dcm_ds.file_meta = meta_ds

    # Manufacturer
    dcm_ds.Manufacturer = 'ICASSP CBCT Recon Challenge'
    dcm_ds.ReferringPhysicianName = 'ICASSP'
    dcm_ds.ManufacturerModelName = 'sample'

    # Patient
    dcm_ds.PatientName = "ANON"
    dcm_ds.PatientID = pydicom.uid.generate_uid()
    dcm_ds.PatientBirthDate = "19000101"
    dcm_ds.PatientSex = "M"
    dcm_ds.PatientOrientation = "L\P"
    patient_position_origin = [-91.5, -91.5, -80.85]


    # Modality
    dcm_ds.SOPClassUID = pydicom._storage_sopclass_uids.CTImageStorage
    dcm_ds.Modality = 'CT'
    dcm_ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']

    # Study
    dcm_ds.StudyInstanceUID = pydicom.uid.generate_uid()
    dcm_ds.StudyDescription = 'holoscan sample data'
    dcm_ds.StudyDate = '19000101'                   # needed to create DICOMDIR
    dcm_ds.StudyID = str(random.randint(0,1000))
    dcm_ds.StudyTime = "120000"

    # Series
    dcm_ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    dcm_ds.SeriesDescription = 'holoscan sample data'
    dcm_ds.SeriesNumber = str(random.randint(0,1000))
    dcm_ds.Laterality = "L"

    # Image
    dcm_ds.AcquisitionNumber = 1
    dcm_ds.KVP = 110

    dcm_ds.Rows = inp.shape[1]
    dcm_ds.Columns = inp.shape[2]

    dcm_ds.PixelSpacing = [pixel_spacing_x, pixel_spacing_y]
    dcm_ds.SliceThickness = slice_thickness

    dcm_ds.PatientPosition = 'HFS'
    dcm_ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dcm_ds.PositionReferenceIndicator = 'SN'

    dcm_ds.SamplesPerPixel = 1
    dcm_ds.PhotometricInterpretation = 'MONOCHROME2'
    dcm_ds.BitsAllocated = 16
    dcm_ds.BitsStored = 16
    dcm_ds.HighBit = 15

    dcm_ds.RescaleIntercept = "0.0"
    dcm_ds.RescaleSlope = "1.0"
    dcm_ds.RescaleType = 'HU'

    # Data
    dcm_ds.PixelRepresentation = 1

    inp = cp.asarray(inp)
    inp = (inp - inp.min()) / (inp.max() - inp.min()) * 6000
    inp = cp.asarray(inp, dtype=cp.uint16)

    slices = []
    for frame_id in range(inp.shape[0]):
        slice_ds = deepcopy(dcm_ds)

        sop_instance_uid = pydicom.uid.generate_uid()
        slice_ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        slice_ds.SOPInstanceUID  = sop_instance_uid
        slice_ds.InstanceNumber = frame_id

        slice_patient_position = deepcopy(patient_position_origin)
        slice_patient_position[-1] += slice_thickness * frame_id
        slice_ds.ImagePositionPatient = slice_patient_position

        slice_ds.PixelData = inp[frame_id,:,:].tobytes()
        slices.append(slice_ds)

        slice_ds.save_as(f'./volume/slice-{frame_id:03}.dcm', write_like_original=False)

    return slices


class InputOp(Operator):
    """
    Operator which simulates streaming of CBCT projection
    data one-by-one.

    The number of projection images is determined by the
    `num_angles` property in the config file.
    """
    def __init__(
            self,
            fragment,
            *args,
            sinogram_path,
            sinogram_size_x,
            sinogram_size_y,
            sinogram_size_z,
            num_angles,
            **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)

        # Load sinogram
        sino = np.load(sinogram_path, allow_pickle=True)

        # Optionally scale up sinogram for a more
        # realistic resolution.
        sino = self.__scale_sinogram(
            sino,
            sinogram_size_x,
            sinogram_size_y,
            sinogram_size_z,
        )

        self.sino = cp.asarray(sino, dtype=cp.float32)
        self.angles = cp.linspace(0, 2*cp.pi, num_angles, endpoint=False)
        self.counter = 0

    def __scale_sinogram(
            self,
            sino,
            sinogram_size_x,
            sinogram_size_y,
            sinogram_size_z,
    ):
        if any((sino.shape[0] != sinogram_size_x,
                sino.shape[1] != sinogram_size_y,
                sino.shape[2] != sinogram_size_z)):
            return skimage.transform.resize(
                sino,
                output_shape=(
                    sinogram_size_x,
                    sinogram_size_y,
                    sinogram_size_z,
                )
            )
        else:
            return sino

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        sino_emit = self.sino[:,self.counter:self.counter+1,:]
        angles_emit = self.angles[self.counter:self.counter+1]

        self.counter += 1

        op_output.emit(
            {
                "sino": sino_emit,
                "angles": angles_emit,
            },
            "out"
        )


class FDKReconOp(Operator):
  """
  Operator doing Feddkamp (FDK) reconstruction
  upon received sinogram data. The FDK algorithm
  from `ts_algorithms` was used.

  """
  def __init__(
      self,
      fragment,
      *args,
      fov_x_lu,
      fov_y_lu,
      fov_z_lu,
      fov_x_voxels,
      fov_y_voxels,
      fov_z_voxels,
      detector_x_lu,
      detector_y_lu,
      detector_x_pixels,
      detector_y_pixels,
      num_angles,
      src_orig_dist,
      src_det_dist,
      **kwargs,
  ):
    super().__init__(fragment, *args, **kwargs)

    self.fov_size_lu = [
      fov_x_lu,
      fov_y_lu,
      fov_z_lu
    ]
    self.fov_size_voxels = [
      fov_x_voxels,
      fov_y_voxels,
      fov_z_voxels
    ]

    self.detector_size_pixels = [
      detector_x_pixels, detector_y_pixels
    ]
    self.detector_size_lu = [
      detector_x_lu,
      detector_y_lu
    ]

    self.sod = src_orig_dist
    self.sdd = src_det_dist

    self.volume = ts.volume(
      shape=self.fov_size_voxels,
      size=self.fov_size_lu
    )

    self.final_recon = torch.zeros(
      self.volume.shape,
      dtype=torch.float32,
      device="cuda",
    )

    self.counter = 0
    self.num_angles = num_angles

  def setup(self, spec: OperatorSpec):
    spec.input("in")
    spec.output("out_vis")
    spec.output("out_recon")

  def create_projector(self, angles):
    proj_geometry = ts.cone(
      angles=torch.tensor(angles, dtype=torch.float32, device="cpu"),
      shape=self.detector_size_pixels,
      size=self.detector_size_lu,
      src_orig_dist=self.sod,
      src_det_dist=self.sdd
    )
    return ts.operator(self.volume, proj_geometry)

  def compute(self, op_input, op_output, context):
    in_message = op_input.receive("in")

    sino = in_message["sino"]
    angles = in_message["angles"]

    projector = self.create_projector(angles)
    recon = fdk(projector, torch.tensor(sino, device="cuda"))

    self.final_recon += recon
    self.counter += angles.shape[0]

    vis = self.final_recon[128,:,:].clone().detach()

    vis = (vis - vis.min()) / (vis.max() - vis.min()) * 255.0
    vis = vis.squeeze(1)
    vis = vis[:,:,None].repeat(1, 1, 3)
    vis = vis.to(torch.uint8)

    op_output.emit(
        {
            "vis": cp.asarray(vis),
        },
        "out_vis"
    )

    op_output.emit(
        {
            "recon": cp.asarray(self.final_recon),
            "recon_complete": cp.asarray([int(self.counter == self.num_angles)])
        },
        "out_recon"
    )


class DICOMSenderOp(Operator):
    """
    Operator which handles converting CBCT volume to
    DICOM dataset, and sending it to DICOM server.

    """
    def __init__(
            self,
            fragment,
            *args,
            ip,
            port,
            fov_x_lu,
            fov_y_lu,
            fov_z_lu,
            fov_x_voxels,
            fov_y_voxels,
            fov_z_voxels,
            **kwargs,
    ):
        super().__init__(fragment, *args, **kwargs)
        self.ip = ip
        self.port = port

        self.fov_x_lu = fov_x_lu
        self.fov_y_lu = fov_y_lu
        self.fov_z_lu = fov_z_lu
        self.fov_x_voxels = fov_x_voxels
        self.fov_y_voxels = fov_y_voxels
        self.fov_z_voxels = fov_z_voxels

    def setup(self, spec: OperatorSpec):
        spec.input("in")

        # Initialise Application Entity
        # and request CT storage
        self.ae = AE()
        self.ae.add_requested_context(CTImageStorage)

    def send_c_store(self, dcm_slices):
        # Associate with DICOM server
        assoc = self.ae.associate(self.ip, self.port)

        if assoc.is_established:
            print(f"connected to DICOM server {self.ip} @port {self.port}")

            for slice_id, slice_ds in enumerate(dcm_slices):
                status = assoc.send_c_store(slice_ds)

                if status:
                    print(
                        "\r",
                        "slice {0:03} sent, status: 0x{0:04x}".format(
                            slice_id + 1, status.Status),
                        end="",
                        flush=True,
                    )
                else:
                    print('timed out, was aborted or received invalid response')
            assoc.release()
        else:
            print('association rejected, aborted or never connected')

    def compute(self, op_input, op_output, context):
        in_message = op_input.receive("in")
        recon_complete = in_message["recon_complete"]

        if cp.asarray(recon_complete)[0]:
            print("recon complete, sending results to DICOM server...")

            pixel_spacing_x = float(self.fov_x_lu) / self.fov_x_voxels
            pixel_spacing_y = float(self.fov_y_lu) / self.fov_y_voxels
            slice_thickness = float(self.fov_z_lu) / self.fov_z_voxels

            dcm_ds = convert_to_dcm(
                in_message["recon"],
                pixel_spacing_x,
                pixel_spacing_y,
                slice_thickness
            )

            self.send_c_store(dcm_ds)


class FDKReconApp(Application):

  def compose(self):

    cuda_stream_pool = CudaStreamPool(
      self,
      name="cuda_stream",
      dev_id=0,
      stream_flags=0,
      stream_priority=0,
      reserved_size=1,
      max_size=5,
    )

    # Input operator
    input_op = InputOp(
      self,
      CountCondition(self, self.kwargs("geometry")["num_angles"]),
      name="input",
      sinogram_path=self.kwargs("input")["sinogram_path"],
      sinogram_size_x=self.kwargs("geometry")["detector_x_pixels"],
      sinogram_size_y=self.kwargs("geometry")["num_angles"],
      sinogram_size_z=self.kwargs("geometry")["detector_y_pixels"],
      num_angles=self.kwargs("geometry")["num_angles"],
    )

    # denoising_sino_op = InferenceOp(
    #   self,
    #   name="denoising_sinogram",
    #   allocator=cuda_stream_pool,
    #   **self.kwargs("denoising_sinogram"),
    # )

    recon_op = FDKReconOp(
      self,
      name="fdk_recon",
      **self.kwargs("geometry"),
    )

    # denoising_volume_op = InferenceOp(
    #     self,
    #     name="denoising_volume",
    #     allocator=cuda_stream_pool,
    #     **self.kwargs("denoising_volume"),
    # )

    # Use VTK for 3D visualization
    visualizer_op = HolovizOp(
      self,
      name="visualizer",
      cuda_stream_pool=cuda_stream_pool,
      tensors=[dict(name="vis", type="color")],
      width=self.kwargs("geometry")["fov_x_voxels"],
      height=self.kwargs("geometry")["fov_y_voxels"],
    )

    # Send data to DICOM server when recon is complete
    dcm_sender_op = DICOMSenderOp(
        self,
        ip=self.kwargs("dicom_server")["ip"],
        port=self.kwargs("dicom_server")["port"],
        **self.kwargs("geometry"),
    )

    self.add_flow(input_op, recon_op, {("out", "in")})
    self.add_flow(recon_op, visualizer_op, {("out_vis", "receivers")})
    self.add_flow(recon_op, dcm_sender_op, {("out_recon", "in")})

    # self.add_flow(input_op, denoising_sino_op, {("out", "receivers")})
    # self.add_flow(denoising_sino_op, recon_op, {("transmitter", "in")})
    # self.add_flow(recon_op, visualizer_op, {("out", "receivers")})


if __name__ == "__main__":

  config_file = os.path.join(os.path.dirname(__file__), "cbct_recon.yaml")

  app = FDKReconApp()
  app.config(config_file)
  app.run()
