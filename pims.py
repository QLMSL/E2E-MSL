import math

import torch
import torch.nn as nn

from .ste import MaskedSelectSTE, GradScale

class DifferentiableSoftMeanShift2D(nn.Module):
    def __init__(
            self,
            num_iters=50,
            sigma=1.0,
            merge_radius=4.0,
            merge_temp=0.05,
            ste_temp=0.5,
            eta=0.01,
            gate_sharpness=50.0):

        super().__init__()

        self.num_iters = num_iters
        self.sigma = sigma
        self.merge_radius = merge_radius
        self.merge_temp = merge_temp
        self.ste_temp = torch.tensor(float(ste_temp))
        self.eta = eta

        self.eps_raw = nn.Parameter(
            torch.tensor(math.log(0.1 / (1 - 0.1)))
        )

        self.gate_sharpness = gate_sharpness

    def forward(self, z, img_tensor, soft_mask):

        device = z.device

        B, _, H, W = z.shape

        centers_list = []


        eps = torch.sigmoid(self.eps_raw)

        eps = GradScale.apply(
            eps,
            20.0  
        )

        for b in range(B):

            x = z[b, 0]

            sm = soft_mask[b, 0]

            mask_soft = torch.sigmoid(
                (sm - eps) * self.gate_sharpness
            )


            nonzero_vals = MaskedSelectSTE.apply(
                x,
                mask_soft
            )

            rss_vals = MaskedSelectSTE.apply(
                img_tensor[b, 0],
                mask_soft
            )

            if nonzero_vals.numel() == 0:
                centers_list.append(
                    torch.zeros((0, 2), device=device)
                )

                continue

            points = torch.stack(
                [
                    nonzero_vals.real,
                    nonzero_vals.imag
                ],
                dim=1
            )

            rss_vals = rss_vals.reshape(-1)
            for _ in range(self.num_iters):
                dists = torch.cdist(
                    points,
                    points
                )

                kernel = torch.exp(
                    -dists ** 2 /
                    (2 * self.sigma ** 2)
                )

                weights = (
                        kernel *
                        rss_vals.unsqueeze(1)
                )

                points = (
                                 weights @ points
                         ) / (
                                 weights.sum(
                                     dim=1,
                                     keepdim=True
                                 ) + 1e-6
                         )

            cluster_centers = self.soft_merge_centers(
                points,
                merge_temp=self.merge_radius
            )

            centers_list.append(
                cluster_centers
            )

        print(
            "eps =",
            torch.sigmoid(self.eps_raw).item()
        )

        return centers_list

    def soft_merge_centers(self, points, merge_temp=0.05):
        if points.shape[0] == 0:
            return points

        unique_mask = torch.ones(points.size(0), device=points.device)
        final_centers = []
        for i in range(points.size(0)):
            if unique_mask[i] == 0:
                continue
            diff = torch.norm(points - points[i], dim=1)
            close = (diff < merge_temp)
            final_centers.append(points[close].mean(dim=0))
            unique_mask[close] = 0

        return torch.stack(final_centers)
