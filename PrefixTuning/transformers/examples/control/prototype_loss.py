import torch
from torch.nn import functional as F
import torch.nn as nn

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1).mean()
        return b

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(query_samples, prototypes, classes, n_query, device, entropy_loss = None):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''


    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    if entropy_loss is not None:
        prototypes = prototypes.detach().clone()

    dists = euclidean_dist(query_samples, prototypes)

    if entropy_loss is not None:
        loss_val = entropy_loss(-dists)
    else:
        n_classes = len(classes)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

        target_inds = torch.arange(0, n_classes).to(device)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    # _, y_hat = log_p_y.max(2)
    #acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val



def loss_variational(mu, z_var, lambda_kl=1, reduction="mean"):
    raw_kl_loss = torch.exp(z_var) + mu**2 - 1.0 - z_var
    if reduction == "mean":
        kl_loss = 0.5 * torch.mean(raw_kl_loss)
    elif reduction == "sum":
        kl_loss = 0.5 * torch.sum(raw_kl_loss)
    return lambda_kl * kl_loss


def hsic(Kx, Ky):
    Kxy = torch.mm(Kx, Ky)
    n = Kxy.shape[0]
    h = torch.trace(Kxy) / n**2 + torch.mean(Kx) * torch.mean(Ky) - 2 * torch.mean(Kxy) / n
    return h * n**2 / (n - 1)**2


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()