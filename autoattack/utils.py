import numpy as np
import torch
import copy


def clip_image_values(x, minv, maxv):
    return x.clamp_(min=minv, max=maxv)


def valid_bounds(img, delta=255):

    im = copy.deepcopy(np.asarray(img))
    im = im.astype(np.int)

    # General valid bounds [0, 255]
    valid_lb = np.zeros_like(im)
    valid_ub = np.full_like(im, 255)

    # Compute the bounds
    lb = im - delta
    ub = im + delta

    # Validate that the bounds are in [0, 255]
    lb = np.maximum(valid_lb, np.minimum(lb, im))
    ub = np.minimum(valid_ub, np.maximum(ub, im))

    # Change types to uint8
    lb = lb.astype(np.uint8)
    ub = ub.astype(np.uint8)

    return lb, ub


def inv_tf(x, mean, std):

    for i in range(len(mean)):

        x[i] = np.multiply(x[i], std[i], dtype=np.float32)
        x[i] = np.add(x[i], mean[i], dtype=np.float32)

    x = np.swapaxes(x, 0, 2)
    x = np.swapaxes(x, 0, 1)

    return x


def inv_tf_pert(r):

    pert = np.sum(np.absolute(r), axis=0)
    pert[pert != 0] = 1

    return pert


def get_label(x):
    s = x.split(' ')
    label = ''
    for l in range(1, len(s)):
        label += s[l] + ' '

    return label


def nnz_pixels(arr):
    return np.count_nonzero(np.sum(np.absolute(arr), axis=0))


def project_L0_box(y, x, eps, lb, ub):
    """
    projection of the batch y to a batch x such that:
    - each image of the batch x has at most k pixels with non-zero channels
    - lb <= x <= ub
    """
    b, c, h, w = y.shape
    p1 = torch.sum((y - x) ** 2, dim=1)
    z = y.clamp_(min=lb, max=ub)
    p2 = torch.sum((y - z) ** 2, dim=1)
    gain = p1 - p2
    p3, ind = torch.sort(gain.view(b, -1), dim=1, descending=True)
    # only keep the k pixels corresponding to the largest gains
    p3 = p3[:, eps].view(b, 1, 1).expand(b, h, w)
    # generate masks
    mask = (gain > p3).float()
    mask = mask.view(b, 1, h, w).expand(b, c, h, w)
    # apply masks
    z = (1 - mask) * x + mask * z
    return z
    

def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    #u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()
    
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)
    
    s1 = -u.sum(dim=1)
    
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
    if c2.nelement != 0:
    
      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)
      
      #print(c2.shape, lb.shape)
      
      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
          
      while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2.)
        counter2 = counter4.type(torch.LongTensor)
        
        c8 = s[c2, counter2] + c[c2] < 0
        ind3 = c8.nonzero().squeeze(1)
        ind32 = (~c8).nonzero().squeeze(1)
        #print(ind3.shape)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]
        
        #print(lb, ub)
        counter += 1
        
      lb2 = lb.long()
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
    return (sigma * d).view(x2.shape)


class Logger():
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()