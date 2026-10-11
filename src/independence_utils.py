"""Helpers to load the trained counterfactual VAE + its conditioning classifier
and extract reconstructions / residuals for every partition."""
import json
import os

import joblib
import numpy as np
import torch

from options import MainParser
from src.models import get_model
from src.trainers import get_trainer


def load_setup(cond='svc', exp='final', extra_args=()):
    """Mirrors main.py: returns (trainer, classifier_fn, args)."""
    argv = ['--exp', exp, '--cond', cond, '--eval', '--load', '0', *extra_args]
    args = MainParser().parse_args(argv)
    final = args.exp[:5] == 'final'
    base_exp = args.exp
    args.exp = '_'.join(filter(bool, [args.cond, base_exp]))
    trainer = get_trainer(args, final)
    trainer.model.eval()

    model_path = os.path.join(args.data_dir, 'models')
    if cond == 'svc':
        svc_model = joblib.load(os.path.join(model_path, '_'.join(filter(bool, ['svc', base_exp])) + '.joblib'))

        def classifier(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            return svc_model.predict_proba(x)[:, 1]
    else:
        mlp_model = get_model('MLP', args.res, args.z_dim)
        pretrain = 'pretrain' if cond == 'ae' else ''
        d = os.path.join(model_path, '_'.join(filter(bool, ['mlp', pretrain, base_exp])))
        final_epoch = json.load(open(os.path.join(d, 'settings.json')))['training_epochs']
        state = torch.load(os.path.join(d, f'model_epoch{final_epoch}.pt'),
                           map_location=torch.device(trainer.device))
        mlp_model.load_state_dict(state, strict=False)
        mlp_model = mlp_model.to(trainer.device).eval()

        @torch.inference_mode()
        def classifier(x):
            if not torch.is_tensor(x):
                x = torch.from_numpy(x)
            out = torch.sigmoid(mlp_model(x.float().to(trainer.device))['logits'])
            return out.cpu().squeeze(-1).numpy()

    return trainer, classifier, args


@torch.inference_mode()
def reconstruct(trainer, matrices, probs, batch=64):
    """Deterministic (eval-mode) reconstruction with the true condition."""
    trainer.model.eval()
    recs = []
    for i in range(0, len(matrices), batch):
        x = torch.from_numpy(matrices[i:i + batch]).float().to(trainer.device)
        p = torch.from_numpy(probs[i:i + batch]).float().view(-1, 1).to(trainer.device)
        data = trainer.model.encoder(x, p)
        recs.append(trainer.model.decoder(data=data, condition=p)['recon'].cpu().numpy())
    return np.concatenate(recs, 0)


def get_partition(trainer, part):
    loader = {'train': trainer.train_loader, 'val': trainer.val_loader,
              'test': trainer.test_loader}[part]
    ds = loader.dataset
    return ds.matrices.astype(np.float32), ds.labels.astype(int), ds.condition.astype(np.float32)
