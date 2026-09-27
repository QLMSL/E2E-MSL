import torch


class MaskedSelectSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, mask_soft):
        mask_hard = (mask_soft > 0.5)

        ctx.save_for_backward(x, mask_soft, mask_hard)
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype

        return x[mask_hard]

    @staticmethod
    def backward(ctx, grad_output):
        # mask_soft, mask_hard = ctx.saved_tensors
        x, mask_soft, mask_hard = ctx.saved_tensors


        grad_1 = torch.zeros(
            ctx.x_shape,
            dtype=ctx.x_dtype,
            device=x.device
        )
        grad_1[mask_hard] = grad_output

        grad_x = torch.zeros_like(mask_soft)

        # 被选中的位置接收梯度
        grad_x[mask_hard] = grad_output.abs().mean()

        grad_mask = grad_x

        return grad_1, grad_mask

class GradScale(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None
