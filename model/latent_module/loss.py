import torch
from torch import nn
from torch.nn import functional as F

class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=0)
        return kl.mean()

def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum

class MaximumMeanDiscrepancyLoss(nn.Module):
    def __init__(self):
        super(MaximumMeanDiscrepancyLoss, self).__init__()

    def forward(self, z_tilde, z, z_var):
        r"""Calculate maximum mean discrepancy described in the WAE paper.
        Args:
            z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
                2D Tensor(batch_size x dimension).
            z (Tensor): samples from prior distributions. same shape with z_tilde.
            z_var (Number): scalar variance of isotropic gaussian prior P(Z).
        """
        # batch_size = z_tilde.size(1)
        # z_tilde = z_tilde.view(batch_size, -1)
        # z = z.view(batch_size, -1)
        assert z_tilde.size() == z.size()
        assert z.ndimension() == 2

        n = z.size(0)
        out = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1)) + \
            im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1)) + \
            -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)

        return out

# class MaximumMeanDiscrepancyLoss3(nn.Module):
#     def __init__(self):
#         super(MaximumMeanDiscrepancyLoss3, self).__init__()

#     def forward(self, x, y, z_var):
#         batch_size = x.size(0)
#         xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

#         rx = (xx.diag().unsqueeze(0).expand_as(xx))
#         ry = (yy.diag().unsqueeze(0).expand_as(yy))

#         K = torch.exp(- z_var * (rx.t() + rx - 2*xx))
#         L = torch.exp(- z_var * (ry.t() + ry - 2*yy))
#         P = torch.exp(- z_var * (rx.t() + ry - 2*zz))

#         beta = (1./(batch_size*(batch_size-1)))
#         gamma = (2./(batch_size*batch_size)) 

#         return torch.abs(beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P))

# class MaximumMeanDiscrepancyLoss2(nn.Module):
#     def __init__(self, device):
#         super(MaximumMeanDiscrepancyLoss2, self).__init__()
#         self.kernel = 'multiscale'
#         self.device = device

#     def forward(self, x, y, z_var):
#         xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

#         xx = xx.to(self.device)
#         yy = yy.to(self.device)
#         zz = zz.to(self.device)

#         rx = (xx.diag().unsqueeze(0).expand_as(xx))
#         ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
#         dxx = rx.t() + rx - 2. * xx # Used for A in (1)
#         dyy = ry.t() + ry - 2. * yy # Used for B in (1)
#         dxy = rx.t() + ry - 2. * zz # Used for C in (1)
        
#         XX, YY, XY = (torch.zeros(xx.shape).to(self.device),
#                     torch.zeros(xx.shape).to(self.device),
#                     torch.zeros(xx.shape).to(self.device))
        
#         if self.kernel == "multiscale":
            
#             bandwidth_range = [0.2, 0.5, 0.9, 1.3]
#             for a in bandwidth_range:
#                 XX += a**2 * (a**2 + dxx)**-1
#                 YY += a**2 * (a**2 + dyy)**-1
#                 XY += a**2 * (a**2 + dxy)**-1
                
#         if self.kernel == "rbf":\
        
#             bandwidth_range = [10, 15, 20, 50]
#             for a in bandwidth_range:
#                 XX += torch.exp(-0.5*dxx/a)
#                 YY += torch.exp(-0.5*dyy/a)
#                 XY += torch.exp(-0.5*dxy/a)

#         return torch.mean(XX + YY - 2. * XY)