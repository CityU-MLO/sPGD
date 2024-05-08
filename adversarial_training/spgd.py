import torch
from torch.nn import functional as F
import numpy as np
from skimage.util import random_noise


def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    # return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
    #         x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)
    return -(x[u, y] - x[u, y_target])


def dlr_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    # return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
    #         1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))


def margin_loss(logits, x, y, targeted=False):
    """
        :param y:        correct labels if untargeted else target labels
        """
    u = torch.arange(x.shape[0])
    y_corr = logits[u, y].clone()
    logits[u, y] = -float('inf')
    y_others = logits.max(dim=-1)[0]

    if not targeted:
        return y_corr - y_others
    else:
        return y_others - y_corr


class MaskingA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k):
        b, c, h, w = x.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        # keep k largest elements of mask and set the rest to 0
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, 1, h, w)

        ctx.save_for_backward(x, mask)
        return x * mask_proj

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, grad_output * x, None, None


class MaskingB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, k):
        b, c, h, w = x.shape
        # project x onto L0 ball
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :k], 1).view(b, 1, h, w)

        ctx.save_for_backward(x, mask_proj)
        return x * mask_proj

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, grad_output * x, None, None


class SparsePGD(object):
    def __init__(self, model, epsilon=255 / 255, k=10, t=30, random_start=True, patience=3, classes=10, alpha=0.25,
                 beta=0.25, unprojected_gradient=True, verbose=False):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.t = t
        self.random_start = random_start
        self.alpha = epsilon * alpha
        self.beta = beta
        self.patience = patience
        self.classes = classes
        self.masking = MaskingA() if unprojected_gradient else MaskingB()
        self.weight_decay = 0.0
        self.p_init = 1.0
        self.verbose = verbose
        self.verbose_interval = 100

    def initial_perturb(self, x, seed=-1):
        if self.random_start:
            if seed != -1:
                torch.random.manual_seed(seed)
            perturb = x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            perturb = x.new(x.size()).zero_()
        perturb = torch.min(torch.max(perturb, -x), 1 - x)
        return perturb

    def update_perturbation(self, perturb, grad, x):
        b, c, h, w = perturb.size()
        step_size = self.alpha * torch.ones(b, device=perturb.device)
        perturb1 = perturb + step_size.view(b, 1, 1, 1) * grad.sign()
        perturb1 = perturb1.clamp_(-self.epsilon, self.epsilon)
        perturb1 = torch.min(torch.max(perturb1, -x), 1 - x)
        return perturb1

    def update_mask(self, mask, grad):
        # prev_mask = self.project_mask(mask.clone())
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        b, c, h, w = mask.size()
        grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        d = grad / (grad_norm + 1e-10)

        step_size = np.sqrt(h * w * c) * self.beta * torch.ones(b, device=mask.device)
        step_size = step_size.scatter_(0, (grad_norm.view(-1) < 2e-10).nonzero().squeeze(), 0)
        mask = mask + step_size.view(b, 1, 1, 1) * d

        return mask

    def initial_mask(self, x, it=0, prev_mask=None):

        if x.dim() == 3:
            x = x.unsqueeze(0)
            prev_mask = prev_mask.unsqueeze(0) if prev_mask is not None else None
        b, c, h, w = x.size()

        mask = torch.randn(b, 1, h, w).to(x.device)

        if prev_mask is not None:
            prev_mask = prev_mask.view(b, -1)
            _, idx = torch.sort(prev_mask.view(b, -1), dim=1, descending=True)
            k_idx = idx[:, :self.k]
            for i in range(len(idx)):
                k_idx[i] = k_idx[i][torch.randperm(self.k)]

            # print(rand_idx.shape, idx.shape)
            p = self.p_selection(it)
            p = max(1, int(p * self.k))
            mask_mask = torch.ones_like(prev_mask).scatter_(1, k_idx[:, :p], 0)
            mask = mask_mask * prev_mask + (1 - mask_mask) * mask.view(b, -1)
            mask = mask.view(b, 1, h, w)

        return mask

    def project_mask(self, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        b, c, h, w = mask.size()
        mask_proj = mask.clone().view(b, -1)
        _, idx = torch.sort(mask_proj, dim=1, descending=True)
        # keep k largest elements of mask and set the rest to 0
        mask_proj = torch.zeros_like(mask_proj).scatter_(1, idx[:, :self.k], 1).view(b, c, h, w)
        return mask_proj

    def check_shape(self, x):
        return x if len(x.shape) == 4 else x.unsqueeze(0)

    def check_low_confidence(self, logits, y, threshold=0.5):
        logits_tmp = logits.clone()
        b, c = logits.size()
        u = torch.arange(b).to(logits.device)
        correct_logit = logits[u, y]
        logits_tmp[u, y] = -float('inf')
        wrong_logit = logits_tmp.max(dim=1)[0]
        return (correct_logit - wrong_logit) < threshold

    def __call__(self, x, y, seed=-1, targeted=False, target=None):
        b, c, h, w = x.size()
        it = torch.zeros(b, dtype=torch.long, device=x.device)

        # generate initial perturbation
        perturb = self.initial_perturb(x, seed)

        # generate initial mask
        mask = self.initial_mask(x)

        mask_best = mask.clone()
        perturb_best = perturb.clone()

        training = self.model.training
        if training:
            self.model.eval()

        ind_all = torch.arange(b).to(x.device)
        reinitial_count = torch.zeros(b, dtype=torch.long, device=x.device)
        x_adv_best = x.clone()

        # remove misclassified examples
        with torch.no_grad():
            logits = self.model(x)
            loss_best = F.cross_entropy(logits, y, reduction='none')
        clean_acc = (logits.argmax(dim=1) == y).float()
        ind_fail = (clean_acc == 1).nonzero().squeeze()
        if self.t == 0:
            return x, clean_acc, it
        if ind_fail.numel() == 0:
            ind_fail = torch.arange(b, device=x.device)

        x = self.check_shape(x[ind_fail])
        perturb = self.check_shape(perturb[ind_fail])
        mask = self.check_shape(mask[ind_fail])
        y = y[ind_fail]
        ind_all = ind_all[ind_fail]
        reinitial_count = reinitial_count[ind_fail]
        if target is not None:
            target = target[ind_fail]
        if ind_fail.numel() == 1:
            y.unsqueeze_(0)
            ind_all.unsqueeze_(0)
            reinitial_count.unsqueeze_(0)
            if target is not None:
                target.unsqueeze_(0)

        if self.verbose:
            acc_list = []

        # First loop
        perturb.requires_grad_()
        mask.requires_grad_()
        proj_perturb = self.masking.apply(perturb, F.sigmoid(mask), self.k)
        # proj_perturb = self.masking.apply(perturb, mask, self.k)
        with torch.no_grad():
            assert torch.norm(proj_perturb.sum(1), p=0, dim=(1, 2)).max().item() <= self.k, 'projection error'
            assert torch.max(x + proj_perturb).item() <= 1.0 and torch.min(
                x + proj_perturb).item() >= 0.0, 'perturbation exceeds bound, min={}, max={}'.format(
                torch.min(x + proj_perturb).item(),
                torch.max(x + proj_perturb).item())
        logits = self.model(x + proj_perturb)

        loss = F.cross_entropy(logits, y, reduction='none')

        loss.sum().backward()
        grad_perturb = perturb.grad.clone()
        grad_mask = mask.grad.clone()

        for i in range(self.t):
            it[ind_all] += 1

            perturb = perturb.detach()
            mask = mask.detach()

            # update mask
            prev_mask = mask.clone()
            mask = self.update_mask(mask, grad_mask)

            # update perturbation using PGD
            perturb = self.update_perturbation(perturb=perturb, grad=grad_perturb, x=x)

            # forward pass
            perturb.requires_grad_()
            mask.requires_grad_()
            proj_perturb = self.masking.apply(perturb, F.sigmoid(mask), self.k)
            with torch.no_grad():
                assert torch.norm(proj_perturb.sum(1), p=0, dim=(1, 2)).max().item() <= self.k, 'projection error'
                assert torch.max(x + proj_perturb).item() <= 1.0 and torch.min(
                    x + proj_perturb).item() >= 0.0, 'perturbation exceeds bound, min={}, max={}'.format(
                    torch.min(x + proj_perturb).item(),
                    torch.max(x + proj_perturb).item())
            logits = self.model(x + proj_perturb)

            # adaptive loss, calculate DLR loss for examples with low confidence, and use CE loss for the rest
            loss = F.cross_entropy(logits, y, reduction='none')

            # backward pass
            loss.sum().backward()
            grad_perturb = perturb.grad.clone()
            grad_mask = mask.grad.clone()

            logits = logits.detach()

            with torch.no_grad():
                fool_label = logits.argmax(dim=1)
                acc = (fool_label == y).float()

                # save the best adversarial example
                loss = loss.detach()
                loss_improve_idx = (loss >= loss_best[ind_all]).nonzero().squeeze()
                if loss_improve_idx.numel() > 0:
                    loss_best[ind_all[loss_improve_idx]] = loss[loss_improve_idx]
                    x_adv_best[ind_all[loss_improve_idx]] = (x + proj_perturb)[loss_improve_idx].detach().clone()

                ind_success = (acc == 0).nonzero().squeeze()
                if ind_success.numel() > 0:
                    x_adv_best[ind_all[ind_success]] = (x + proj_perturb)[ind_success].detach().clone()

                ind_fail = (acc == 1).nonzero().squeeze()
                if ind_fail.numel() > 0:
                    # count the number of times the mask is not updated
                    delta_mask_norm = torch.norm(
                        self.project_mask(mask[ind_fail]) - self.project_mask(prev_mask[ind_fail]), p=0, dim=(1, 2, 3))
                    ind_unchange = (delta_mask_norm <= 0).nonzero().squeeze()
                    if ind_unchange.numel() > 0:
                        if ind_fail.numel() == 1:
                            reinitial_count[ind_fail] += 1
                        else:
                            reinitial_count[ind_fail[ind_unchange]] += 1
                    else:
                        reinitial_count[ind_fail] = 0

                    # reinitialize mask and perturbation when the mask is not updated for 3 consecutive iterations
                    ind_reinit = (reinitial_count >= self.patience).nonzero().squeeze()
                    if ind_reinit.numel() > 0:
                        mask[ind_reinit] = self.initial_mask(x[ind_reinit])
                        reinitial_count[ind_reinit] = 0

                    # remove successfully attacked examples
                    x = self.check_shape(x[ind_fail])
                    perturb = self.check_shape(perturb[ind_fail])
                    mask = self.check_shape(mask[ind_fail])
                    grad_perturb = self.check_shape(grad_perturb[ind_fail])
                    grad_mask = self.check_shape(grad_mask[ind_fail])
                    y = y[ind_fail]
                    ind_all = ind_all[ind_fail]
                    reinitial_count = reinitial_count[ind_fail]
                    if target is not None:
                        target = target[ind_fail]
                    if ind_fail.numel() == 1:
                        y.unsqueeze_(0)
                        ind_all.unsqueeze_(0)
                        reinitial_count.unsqueeze_(0)
                        if target is not None:
                            target.unsqueeze_(0)

            if self.verbose and (i+1) % self.verbose_interval == 0:
                acc_list.append(acc.sum().item())
            if torch.sum(acc) == 0.:
                break
        if training:
            self.model.train()
        if self.verbose:
            if len(acc_list) != self.t // self.verbose_interval:
                acc_list += [acc_list[-1]] * (self.t // self.verbose_interval - len(acc_list))
            return x_adv_best, acc, it, acc_list
        return x_adv_best, acc, it

    def perturb(self, x, y):
        if self.verbose:
            x_adv, acc, it, acc_list = self.__call__(x, y, targeted=False)
            return x_adv, acc.sum(), it, acc_list
        x_adv, acc, it = self.__call__(x, y, targeted=False)
        return x_adv, acc.sum(), it

    def change_masking(self):
        if isinstance(self.masking, MaskingA):
            self.masking = MaskingB()
        else:
            self.masking = MaskingA()
