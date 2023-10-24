import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Sequence

'''
@implementation from https://github.com/wagner-d/TimeSeAD/
@license: MIT License
'''

class DonutModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, mask_prob) -> None:
        super().__init__()
        
        self.z_dim = z_dim
        self.mask_prob = mask_prob
        
        encoder = VaeEncoder(input_dim, hidden_dim, z_dim)
        decoder = VaeEncoder(z_dim, hidden_dim, input_dim)
        
        self.vae = Vae(encoder=encoder, decoder=decoder, logvar_out=False)
    
    def forward(self, inputs: torch.Tensor, num_samples: int=1) -> Tuple[torch.Tensor, ...]:
        # x: (B, T, D)
        x = inputs
        B, T = x.shape

        if self.training:
            # Randomly mask some inputs
            mask = torch.empty_like(x)
            mask.bernoulli_(1 - self.mask_prob)
            x = x * mask
        else:
            mask = None

        # Run the VAE
        x = x.view(x.shape[0], -1)
        
        if self.training or num_samples == 1:
            mean_z, std_z, mean_x, std_x, sample_z = self.vae(x)

            # Reshape the outputs
            mean_x = mean_x.view(B, T)
            std_x = std_x.view(B, T)
            return mean_z, std_z, mean_x, std_x, sample_z, mask
        
        else:
            z_mu, z_std, x_dec_mean, x_dec_std, sample_z = self.vae(x, num_samples)
            
            x_dec_mean = x_dec_mean.view(num_samples, B, T)
            x_dec_std = x_dec_std.view(num_samples, B, T)
            
            nll_output = torch.sum(F.gaussian_nll_loss(x_dec_mean[:, :, -1], x[:, -1].unsqueeze(0),
                                                   x_dec_std[:, :, -1]**2, reduction='none'), dim=(0))
            nll_output /= num_samples
            return nll_output
    

def normal_standard_normal_kl(mean: torch.Tensor, std_or_log_var: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        kl_loss = torch.sum(1 + std_or_log_var - mean.pow(2) - std_or_log_var.exp(), dim=-1)
    else:
        kl_loss = torch.sum(1 + torch.log(std_or_log_var.pow(2)) - mean.pow(2) - std_or_log_var.pow(2), dim=-1)
    return -0.5 * kl_loss
    

def normal_normal_kl(mean_1: torch.Tensor, std_or_log_var_1: torch.Tensor, mean_2: torch.Tensor,
                     std_or_log_var_2: torch.Tensor, log_var: bool = False) -> torch.Tensor:
    if log_var:
        return 0.5 * torch.sum(std_or_log_var_2 - std_or_log_var_1 + (torch.exp(std_or_log_var_1)
                               + (mean_1 - mean_2)**2) / torch.exp(std_or_log_var_2) - 1, dim=-1)

    return torch.sum(torch.log(std_or_log_var_2) - torch.log(std_or_log_var_1) \
                     + 0.5 * (std_or_log_var_1**2 + (mean_1 - mean_2)**2) / std_or_log_var_2**2 - 0.5, dim=-1)



class VAELoss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', logvar_out: bool = True):
        super(VAELoss, self).__init__(size_average, reduce, reduction)
        self.logvar_out = logvar_out

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        z_mean, z_std_or_log_var, x_dec_mean, x_dec_std = predictions[:4]
        if len(predictions) > 4:
            z_prior_mean, z_prior_std_or_logvar = predictions[4:]
        else:
            z_prior_mean, z_prior_std_or_logvar = None, None

        y, = targets

        # Gaussian nnl loss assumes multivariate normal with diagonal sigma
        # Alternatively we can use torch.distribution.Normal(x_dec_mean, x_dec_std).log_prob(y).sum(-1)
        # or torch.distribution.MultivariateNormal(mean, cov).log_prob(y).sum(-1)
        # with cov = torch.eye(feat_dim).repeat([1,bz,1,1])*std.pow(2).unsqueeze(-1).
        # However setting up a distribution seems to be an unnecessary computational overhead.
        # However, this requires pytorch version > 1.9!!!
        nll_gauss = F.gaussian_nll_loss(x_dec_mean, y, x_dec_std.pow(2), reduction='none').sum(-1)
        # For pytorch version < 1.9 use:
        # nll_gauss = -torch.distribution.Normal(x_dec_mean, x_dec_std).log_prob(y).sum(-1)

        # get KL loss
        if z_prior_mean is None and z_prior_std_or_logvar is None:
            # If a prior is not given, we assume standard normal
            kl_loss = normal_standard_normal_kl(z_mean, z_std_or_log_var, log_var=self.logvar_out)
        else:
            if z_prior_mean is None:
                z_prior_mean = torch.tensor(0, dtype=z_mean.dtype, device=z_mean.device)
            if z_prior_std_or_logvar is None:
                value = 0 if self.logvar_out else 1
                z_prior_std_or_logvar = torch.tensor(value, dtype=z_std_or_log_var.dtype, device=z_std_or_log_var.device)

            kl_loss = normal_normal_kl(z_mean, z_std_or_log_var, z_prior_mean, z_prior_std_or_logvar,
                                       log_var=self.logvar_out)

        # Combine
        final_loss = nll_gauss + kl_loss

        if self.reduction == 'none':
            return final_loss
        elif self.reduction == 'mean':
            return torch.mean(final_loss)
        elif self.reduction == 'sum':
            return torch.sum(final_loss)


class MaskedVAELoss(VAELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MaskedVAELoss, self).__init__(size_average, reduce, reduction, logvar_out=False)

    def forward(self, predictions: Tuple[torch.Tensor, ...], targets: Tuple[torch.Tensor, ...], *args, **kwargs) \
            -> torch.Tensor:
        mean_z, std_z, mean_x, std_x, sample_z, mask = predictions
        actual_x, = targets

        if mask is None:
            mean_z = mean_z.unsqueeze(1)
            std_z = std_z.unsqueeze(1)
            return super(MaskedVAELoss, self).forward((mean_z, std_z, mean_x, std_x), (actual_x,), *args, **kwargs)

        # If the loss is masked, one of the terms in the kl loss is weighted, so we can't compute it exactly
        # anymore and have to use a MC approximation like for the output likelihood
        nll_output = torch.sum(mask * F.gaussian_nll_loss(mean_x, actual_x, std_x**2, reduction='none'), dim=-1)

        # This is p(z), i.e., the prior likelihood of Z. The paper assumes p(z) = N(z| 0, I), we drop constants
        beta = torch.mean(mask, dim=(1)).unsqueeze(-1)
        nll_prior = beta * 0.5 * torch.sum(sample_z * sample_z, dim=-1, keepdim=True)

        nll_approx = torch.sum(F.gaussian_nll_loss(mean_z, sample_z, std_z**2, reduction='none'), dim=-1, keepdim=True)

        final_loss = nll_output + nll_prior - nll_approx

        if self.reduction == 'none':
            return final_loss
        elif self.reduction == 'mean':
            return torch.mean(final_loss)
        elif self.reduction == 'sum':
            return torch.sum(final_loss)


        

class VaeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int):
        super(VaeEncoder, self).__init__()
        
        self.std_epsilon = 1e-4
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.std = nn.Linear(hidden_dim, z_dim)
        
        self.softplus = nn.Softplus()
        
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc(x)
        
        mean = self.mean(x)
        std = self.std(x)
        std = self.softplus(std) + self.std_epsilon
        
        return mean, std
    
class Vae(nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, logvar_out: bool = True) -> None:
        super(Vae, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.log_var = logvar_out
        
    def sample_normal(self, mu: torch.Tensor, std_or_log_var: torch.Tensor, log_var: bool = False, num_samples: int = 1):
    # ln(σ) = 0.5 * ln(σ^2) -> σ = e^(0.5 * ln(σ^2))
        if log_var:
            sigma = std_or_log_var.mul(0.5).exp_()
        else:
            sigma = std_or_log_var

        if num_samples == 1:
            eps = torch.randn_like(mu)  # also copies device from mu
        else:
            eps = torch.rand((num_samples,) + mu.shape, dtype=mu.dtype, device=mu.device)
            mu = mu.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
        # z = μ + σ * ϵ, with ϵ ~ N(0,I)
        return eps.mul(sigma).add_(mu)
        
    def forward(self, x: torch.Tensor, num_samples: int = 1,
                force_sample: bool = False) -> Tuple[torch.Tensor, ...]:
        z_mu, z_std_or_log_var = self.encoder(x)

        if self.training or num_samples > 1 or force_sample:
            z_sample = self.sample_normal(z_mu, z_std_or_log_var, log_var=self.log_var, num_samples=num_samples)
        else:
            z_sample = z_mu

        x_dec_mean, x_dec_std = self.decoder(z_sample)

        return z_mu, z_std_or_log_var, x_dec_mean, x_dec_std, z_sample