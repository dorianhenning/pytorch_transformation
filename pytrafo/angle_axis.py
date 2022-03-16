import kornia
import torch


def angle_axis_to_matrix(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape[:-1]
    x_flat = torch.flatten(x, end_dim=-2)
    R_flat = kornia.geometry.angle_axis_to_rotation_matrix(x_flat)
    return R_flat.view(*shape, 3, 3)
