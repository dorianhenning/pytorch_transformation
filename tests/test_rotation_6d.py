import pytest
import pytrafo
import torch


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3)])
def test_rotation_6d_and_angle_axis(shape):
    x3d = torch.rand(shape)
    x6d = pytrafo.angle_axis_to_rotation_6d(x3d)
    x3d_hat = pytrafo.rotation_6d_to_axis_angle(x6d)
    assert torch.allclose(x3d_hat, x3d_hat)


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3)])
def test_rotation_6d_rotation_matrix(shape):
    x3d = torch.rand(shape)
    R = pytrafo.angle_axis_to_matrix(x3d)
    x6d_hat = pytrafo.matrix_to_rotation_6d(R)
    R_hat = pytrafo.rotation_6d_to_rotation_matrix(x6d_hat)
    assert torch.allclose(R_hat, R, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3)])
def test_rotation_6d_cross_transforms(shape):
    x3d = torch.rand(shape)
    x6d = pytrafo.angle_axis_to_rotation_6d(x3d)
    R = pytrafo.angle_axis_to_matrix(x3d)
    x6d_hat = pytrafo.matrix_to_rotation_6d(R)
    assert torch.allclose(x6d_hat, x6d)
