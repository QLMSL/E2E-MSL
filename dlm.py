import torch
import torch.nn as nn

from .dcs import SigmoidThreshold
from .pims import DifferentiableSoftMeanShift2D

class DifferentiableSourceClusteringSTE(nn.Module):
    # def __init__(self, image_size=200, tau=0.2, sharpness=60.0,
    #              num_iters=12, sigma=2.0, merge_radius=0.001, merge_temp=0.05, ste_temp=0.5):
    def __init__(self, image_size=256, tau=0.2, sharpness=60.0,
                 num_iters=10, sigma=2, merge_radius=0.01, merge_temp=0.05, ste_temp=0.5):
        super().__init__()
        self.image_size = image_size
        self.binarizer = SigmoidThreshold(tau, sharpness)

        x = torch.arange(image_size, dtype=torch.float32).repeat(image_size, 1)
        y = torch.arange(image_size, dtype=torch.float32).repeat(image_size, 1).T
        self.register_buffer("complex_grid", torch.complex(x, y))

        self.dsms = DifferentiableSoftMeanShift2D(
            num_iters=num_iters, sigma=sigma,
            merge_radius=merge_radius, merge_temp=merge_temp,
            ste_temp=ste_temp
        )

    def forward(self, img_tensor):
        soft_mask = self.binarizer(img_tensor)
        z = soft_mask * self.complex_grid[None, None, :, :]
        soft_mask_cpu = soft_mask.float().detach().cpu().numpy()
        # z_cpu = z.float().detach().cpu().numpy()
        img_tensor_cpu = img_tensor.float().detach().cpu().numpy()
        centers_list = self.dsms(z, img_tensor, soft_mask)
        # centers_list = self.dsms(z, img_tensor)
        return centers_list