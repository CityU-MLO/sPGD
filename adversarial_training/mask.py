import torch

class MaskingA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k):
        b, c, h, w = mask.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # stochastic sampling
        # mask_proj = torch.softmax(mask_proj/0.001, dim=1)
        # idx = torch.multinomial(mask_proj, num_samples=k, replacement=False)
        # keep k largest elements of mask and set the rest to 0
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, c, h, w)
        # mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx, 1).view(b, 1, h, w)

        ctx.save_for_backward(x, mask)
        return x * mask_proj, mask_proj

    @staticmethod
    def backward(ctx, grad_output, _):
        x, mask = ctx.saved_tensors
        h, w = mask.shape[-2:]
        grad_mask = grad_output * x
        if h == 1:
            grad_mask = grad_mask.mean(dim=2, keepdim=True)
        elif w == 1:
            grad_mask = grad_mask.mean(dim=3, keepdim=True)

        return grad_output * mask, grad_mask, None, None


class MaskingB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k):
        b, c, h, w = mask.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        # mask_back = mask.clone().view(b, -1).scatter_(1, idx[:, k:], 0).view(b, 1, h, w)
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, c, h, w)

        ctx.save_for_backward(x, mask_proj)
        return x * mask_proj, mask_proj

    @staticmethod
    def backward(ctx, grad_output, _):
        x, mask = ctx.saved_tensors
        h, w = mask.shape[-2:]
        grad_mask = grad_output * x
        if h == 1:
            grad_mask = grad_mask.mean(dim=2, keepdim=True)
        elif w == 1:
            grad_mask = grad_mask.mean(dim=3, keepdim=True)

        return grad_output * mask, grad_mask, None, None


class BlockMaskingA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k, block_size, block_stride, pattern):
        b, c, h, w = mask.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, c, h, w)

        # modify this to support different block sizes and patterns
        if pattern is None:
            kernel = torch.ones(1, 1, block_size[0], block_size[1], dtype=torch.float32).to(mask.device)
        else:
            kernel = pattern.float().view(1, 1, block_size[0], block_size[1]).to(mask.device)
        mask_proj_expand = torch.nn.functional.conv_transpose2d(mask_proj, kernel, stride=block_stride)
        mask_proj_expand = torch.clip(mask_proj_expand, 0, 1)

        mask_expand = torch.nn.functional.conv_transpose2d(mask, kernel, stride=block_stride)
        mask_expand = torch.clip(mask_expand, 0, 1)

        ctx.save_for_backward(x, mask_expand, kernel)
        ctx.block_size = block_size
        ctx.block_stride = block_stride
        return x * mask_proj_expand, mask_proj

    @staticmethod
    def backward(ctx, grad_output, _):
        x, mask, kernel = ctx.saved_tensors
        grad_mask = (grad_output * x).sum(dim=1, keepdim=True)
        # downsample grad_mask to the same size as mask
        grad_mask = torch.nn.functional.conv2d(grad_mask, kernel / kernel.sum(), stride=ctx.block_stride)
        return grad_output * mask, grad_mask, None, None, None, None, None


class BlockMaskingB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k, block_size, block_stride, pattern):
        b, c, h, w = mask.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, c, h, w)

        if pattern is None:
            kernel = torch.ones(1, 1, block_size[0], block_size[1], dtype=torch.float32).to(mask.device)
        else:
            kernel = pattern.float().view(1, 1, block_size[0], block_size[1]).to(mask.device)
        mask_expand = torch.nn.functional.conv_transpose2d(mask_proj, kernel, stride=block_stride)
        mask_expand = torch.clip(mask_expand, 0, 1)

        ctx.save_for_backward(x, mask_expand, kernel)
        ctx.block_size = block_size
        ctx.block_stride = block_stride
        return x * mask_expand, mask_proj

    @staticmethod
    def backward(ctx, grad_output, _):
        x, mask, kernel = ctx.saved_tensors
        grad_mask = (grad_output * x).sum(dim=1, keepdim=True)
        # downsample grad_mask to the same size as mask
        grad_mask = torch.nn.functional.conv2d(grad_mask, kernel / kernel.sum(), stride=ctx.block_stride)
        return grad_output * mask, grad_mask, None, None, None, None, None