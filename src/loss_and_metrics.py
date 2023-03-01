import torch
import torch.nn.functional as F
import numpy as np


class VAELoss:

    def __init__(self, c_reg):
        self.c_reg = c_reg
        self.train_var = None  # calculated variance
        self.train_logvar = None  # calculated log variance
        self.adj_nll = 0.5 * np.log(2 * np.pi)  # adjustment to calculate the NLL
        self.scale_prior = 1

    def __call__(self, outputs, inputs, targets):
        inputs = inputs[0]
        kld, kld_real = self.kld_loss(inputs, outputs, targets)
        NLL, MSE = self.nll_loss(inputs, outputs['recon'])
        return {'Criterion': MSE.mean() + self.c_reg * kld.mean(),
                'real_loss': NLL + kld_real.sum(),
                'KLD_Smooth': kld.sum(),
                'KLD': kld_real.sum(),
                'NLL': NLL,
                'MSE': MSE.sum(),
                }

    def nll_loss(self, inputs, recon):
        SE = F.mse_loss(recon, inputs, reduction='none')
        if self.train_var is None:
            self.train_var = SE.mean(0).detach()
            self.train_logvar = torch.log(self.train_var)
        local_device = recon.device  # makes sure you can change device during training
        train_var, train_logvar = self.train_var.to(local_device), self.train_logvar.to(local_device)
        NLL = 0.5 * (SE / train_var + train_logvar).sum() + self.adj_nll * recon.numel()
        if recon.requires_grad:
            self.train_var = 0.9 * train_var + 0.1 * SE.mean(0).detach()
            self.train_logvar = torch.log(self.train_var)
        return NLL, SE.sum([1, 2, 3])

    def kld_loss(self, inputs, outputs, targets, freebits=0):
        prior_mu = self.condition(inputs, outputs, targets)
        q_mu = outputs['mu'][0]
        q_logvar = outputs['log_var'][0]
        kld_matrix = -1 - q_logvar + (q_mu - prior_mu) ** 2 + q_logvar.exp()
        kld_free_bits = F.softplus(kld_matrix - 2 * freebits) + 2 * freebits
        kld = 0.5 * kld_free_bits.sum((1, 2, 3))
        kld_real = 0.5 * kld_matrix.sum((1, 2, 3))
        for d_mu, d_logvar, p_logvar in zip(outputs['mu'][1:], outputs['log_var'][1:], outputs['prior_log_var']):
            # d_mu = q_mu - p_mu
            # d_logvar = q_logvar - p_logvar
            kld_matrix = -1 - d_logvar + d_logvar.exp() + (d_mu ** 2) / p_logvar.exp()
            kld_free_bits = F.softplus(kld_matrix - 2 * freebits) + 2 * freebits
            kld += 0.5 * kld_free_bits.sum((1, 2, 3))
            kld_real += 0.5 * kld_matrix.sum((1, 2, 3))
        return kld, kld_real

    def condition(self, inputs, outputs, targets):
        return torch.zeros_like(outputs['mu'][0])


class VAEXLoss(VAELoss):
    # prior condition on stored probabilities
    def condition(self, inputs, outputs, targets):
        probs = outputs['condition'].squeeze()
        conditional = torch.zeros_like(outputs['mu'][0])
        conditional[:, :1, ...] = probs.view(-1, 1, 1, 1) * self.scale_prior
        return conditional


class MLPLoss:

    def __init__(self):
        self.ce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, outputs, inputs, targets):
        logits = outputs['logits']
        CE = self.ce(logits, targets)
        return {'Criterion': CE.mean(),
                'CE': CE.sum(),
                }


class AELoss:

    def __init__(self):
        self.se = torch.nn.MSELoss(reduction='none')

    def __call__(self, outputs, inputs, targets):
        recon = outputs['recon']
        SE = self.se(recon, inputs)
        return {'Criterion': SE.mean(),
                'MCE': SE.sum(),
                }


def get_vae_loss(settings):
    c_reg = settings['c_reg']
    return VAEXLoss(c_reg=c_reg)


def get_mlp_loss(settings):
    return MLPLoss()


def get_ae_loss(settings):
    return AELoss()
