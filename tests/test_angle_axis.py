import kornia
import pytest
import torch
import pytrafo


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3)])
def test_angle_axis_to_rotation_matrix(shape):
    x3d = torch.rand(shape)
    R_hat = pytrafo.angle_axis_to_matrix(x3d)
    x3d_flat = torch.flatten(x3d, end_dim=-2)
    R = kornia.geometry.angle_axis_to_rotation_matrix(x3d_flat)
    assert torch.allclose(torch.flatten(R_hat, end_dim=-3), R)
