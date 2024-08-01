#! /usr/bin/env python3
## ---------------------------------------------------------------------------
##
## File: app.py<cbct_recon> for CBCT Recon
##
## Created by Zhijin Li
## E-mail:   <zhijinl@nvidia.com>
##
## Started on  Mon May 13 18:04:23 2024 Zhijin Li
## Last update Thu Aug  1 12:55:28 2024 Zhijin Li
## ---------------------------------------------------------------------------


import os
import torch
import skimage
import numpy as np
import tomosipo as ts
from ts_algorithms import fdk

from holoscan import as_tensor
from holoscan.resources import CudaStreamPool
from holoscan.conditions import CountCondition
from holoscan.operators import HolovizOp, InferenceOp
from holoscan.core import Application, Operator, OperatorSpec, ConditionType


try:
  import cupy as cp
except ImportError:
  raise ImportError(
      'CuPy must be installed to run this example. See '
      'https://docs.cupy.dev/en/stable/install.html'
    )


IMAGE_SIZE = 1024


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
    sino = np.load(
      sinogram_path,
      allow_pickle=True
    )

    # Optionally scale up sinogram for a more
    # realistic resolution.
    sino = self.__scale_sinogram(
      sino,
      sinogram_size_x,
      sinogram_size_y,
      sinogram_size_z
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

  def setup(self, spec: OperatorSpec):
    spec.output('out')

  def compute(self, op_input, op_output, context):

    sino_emit = self.sino[:,self.counter:self.counter+1,:]
    angles_emit = self.angles[self.counter:self.counter+1]

    self.counter += 1

    op_output.emit(
      {
        'sino': sino_emit,
        'angles': angles_emit,
      },
      'out'
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
      device='cuda',
    )

  def setup(self, spec: OperatorSpec):
    spec.input('in')
    spec.output('out')

  def create_projector(self, angles):
    proj_geometry = ts.cone(
      angles=torch.tensor(angles, dtype=torch.float32, device='cpu'),
      shape=self.detector_size_pixels,
      size=self.detector_size_lu,
      src_orig_dist=self.sod,
      src_det_dist=self.sdd
    )
    return ts.operator(self.volume, proj_geometry)

  def compute(self, op_input, op_output, context):
    in_message = op_input.receive('in')

    sino = in_message['sino']
    angles = in_message['angles']

    projector = self.create_projector(angles)
    recon = fdk(projector, torch.tensor(sino, device='cuda'))

    self.final_recon += recon

    output = torch.tensor(
      self.final_recon[128,:,:],
      device='cuda',
      dtype=torch.float32
    )

    output = (output - output.min()) / (output.max() - output.min()) * 255.0
    output = output.squeeze(1)
    output = output[:,:,None].repeat(1, 1, 3)
    output = output.to(torch.uint8)

    op_output.emit(
      {
        'recon': cp.asarray(output)
      },
      'out'
    )


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
      CountCondition(self, self.kwargs('geometry')['num_angles']),
      name='input',
      sinogram_path=self.kwargs('input')['sinogram_path'],
      sinogram_size_x=self.kwargs('geometry')['detector_x_pixels'],
      sinogram_size_y=self.kwargs('geometry')['num_angles'],
      sinogram_size_z=self.kwargs('geometry')['detector_y_pixels'],
      num_angles=self.kwargs('geometry')['num_angles'],
    )

    # denoising_sino_op = InferenceOp(
    #   self,
    #   name='denoising_sinogram',
    #   allocator=cuda_stream_pool,
    #   **self.kwargs('denoising_sinogram'),
    # )

    recon_op = FDKReconOp(
      self,
      name='fdk_recon',
      **self.kwargs('geometry'),
    )

    # denoising_volume_op = InferenceOp(
    #     self,
    #     name='denoising_volume',
    #     allocator=cuda_stream_pool,
    #     **self.kwargs('denoising_volume'),
    # )

    # Use VTK for 3D visualization
    visualizer_op = HolovizOp(
      self,
      name="visualizer",
      cuda_stream_pool=cuda_stream_pool,
      tensors=[dict(name='recon', type='color')],
      width=self.kwargs('geometry')['fov_x_voxels'],
      height=self.kwargs('geometry')['fov_y_voxels'],
    )

    self.add_flow(input_op, recon_op, {('out', 'in')})
    self.add_flow(recon_op, visualizer_op, {('out', 'receivers')})

    # self.add_flow(input_op, denoising_sino_op, {('out', 'receivers')})
    # self.add_flow(denoising_sino_op, recon_op, {('transmitter', 'in')})
    # self.add_flow(recon_op, visualizer_op, {('out', 'receivers')})


if __name__ == '__main__':

  config_file = os.path.join(os.path.dirname(__file__), 'cbct_recon.yaml')

  app = FDKReconApp()
  app.config(config_file)
  app.run()
