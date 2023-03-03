import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


class MLPLoss:

    def __init__(self):
        self.ce = torch.nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, outputs, inputs, targets):
        logits = outputs['logits']
        ce = self.ce(logits, targets)
        return {'Criterion': ce.mean(),
                'CE': ce.sum(),
                }


class MLPMetrics(MLPLoss):

    def __call__(self, outputs, inputs, targets):
        prediction = outputs['logits'] > 0
        correct = prediction == targets
        ce = super().__call__(outputs, inputs, targets)['CE']
        return {'Accuracy': correct.sum(),
                'CE': ce,
                }


class ClassMetrics:

    def __call__(self, targets, pred):
        conf_matrix = confusion_matrix(targets, pred)
        print()
        print(pd.DataFrame(conf_matrix, columns=('Pred Neg', 'Pred Pos'), index=('Neg', 'Pos')))
        tn, fp, fn, tp = conf_matrix.ravel()
        print()
        print(f'Accuracy: {(tn + tp) / (fn + fp + tn + tp):.2f}  '
              f'Specificity: {tn / (tn + fp):.2f}  Sensitivity: {tp / (fn + tp):.2f}')
        #print()
        #print(classification_report(targets, pred))


class AELoss:

    def __init__(self):
        self.se = torch.nn.MSELoss(reduction='none')

    def __call__(self, outputs, inputs, targets):
        recon = outputs['recon']
        mse = self.se(recon, inputs[0]).mean(1)
        return {'Criterion': mse.mean(),
                'MSE': mse.sum(),
                }


class VAELoss:

    def __init__(self, c_reg):
        super().__init__()
        self.c_reg = c_reg
        self.se = torch.nn.MSELoss(reduction='none')
        self.scale_prior = 1

    def __call__(self, outputs, inputs, targets):
        inputs = inputs[0]
        kld, kld_real = self.kld_loss(inputs, outputs, targets)
        recon = outputs['recon']
        mse = self.se(recon, inputs).mean(1)
        return {'Criterion': mse.mean() + self.c_reg * kld.mean(),
                'KLD Smooth': kld.sum(),
                'KLD': kld_real.sum(),
                'MSE': mse.sum(),
                }

    def kld_loss(self, inputs, outputs, targets, freebits=0):
        prior_mu = self.condition(inputs, outputs, targets)
        q_mu = outputs['mu'][0]
        q_logvar = outputs['log_var'][0]
        kld_matrix = -1 - q_logvar + (q_mu - prior_mu) ** 2 + q_logvar.exp()
        kld_free_bits = F.softplus(kld_matrix - 2 * freebits) + 2 * freebits
        kld = 0.5 * kld_free_bits.sum(1)
        kld_real = 0.5 * kld_matrix.sum(1)
        for d_mu, d_logvar, p_logvar in zip(outputs['mu'][1:], outputs['log_var'][1:], outputs['prior_log_var']):
            # d_mu = q_mu - p_mu
            # d_logvar = q_logvar - p_logvar
            kld_matrix = -1 - d_logvar + d_logvar.exp() + (d_mu ** 2) / p_logvar.exp()
            kld_free_bits = F.softplus(kld_matrix - 2 * freebits) + 2 * freebits
            kld += 0.5 * kld_free_bits.sum(1)
            kld_real += 0.5 * kld_matrix.sum(1)
        return kld, kld_real

    def condition(self, inputs, outputs, targets):
        return torch.zeros_like(outputs['mu'][0])


class VAEXLoss(VAELoss):
    # prior condition on stored probabilities
    def condition(self, inputs, outputs, targets):
        probs = outputs['condition'].squeeze()
        conditional = torch.zeros_like(outputs['mu'][0])
        conditional[:, :1] = probs.view(-1, 1) * self.scale_prior
        return conditional
