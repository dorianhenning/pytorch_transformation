import kornia
import pytest
import pytrafo
import torch


@pytest.mark.parametrize("shape", [(4, 2), (4, 1, 3, 1), (8,)])
def test_inverse_transformation(shape):
    q = torch.rand((*shape, 3), dtype=torch.float32)
    t = torch.rand((*shape, 3), dtype=torch.float32)

    T = torch.zeros((*shape, 4, 4), dtype=torch.float32)
    T[..., :3, :3] = kornia.geometry.angle_axis_to_rotation_matrix(q.view(-1, 3)).view((*shape, 3, 3))
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0

    T_inv = pytrafo.inverse_transformation(T)
    T_hat = pytrafo.inverse_transformation(T_inv)
    assert torch.allclose(T_hat, T, atol=1e-4)
