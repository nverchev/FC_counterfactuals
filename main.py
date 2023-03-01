import torch
import numpy as np
import os
from options import parse_main_args
from src.dataset import get_loaders
from src.optim import get_opt, CosineSchedule
from src.trainers import get_vae_trainer
from src.models import get_vae


def main():
    args = parse_main_args().parse_args()
    exp_name = 'vae' + '_' * bool(args.exp) + args.exp
    ae = args.ae
    dir_path = args.dir_path
    data_dir = os.path.join(dir_path, 'abide_fc_dataset')
    resolution = args.init_resolution
    n_spatial_compressions = args.n_spatial_compressions
    batch_size = args.batch_size
    opt_name = args.optim
    initial_learning_rate = args.lr
    weight_decay = args.wd
    c_reg = args.c_reg
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    training_epochs = args.epochs
    decay_period = args.decay_period
    min_decay = args.min_decay
    checkpoint_every = args.checkpoint
    ind = args.ind
    load = args.load
    model_eval = args.eval
    n_gen = args.gen
    torch.manual_seed = np.random.seed = 112358
    data_loader_settings = dict(
        data_dir=data_dir,
        jitter=True,
        random_crop=True,
        batch_size=batch_size,
        resolution=resolution,
        model_eval=model_eval,
    )

    model_settings = dict(
        x_dim=resolution,
        ae=ae,
        h_dims=[2 ** i for i in range(4, 4 + n_spatial_compressions + 1)],
        k_dims=[3] * n_spatial_compressions,
        strides=[2] * n_spatial_compressions,
        paddings=[1] * n_spatial_compressions,
        z_dims=list(range(n_spatial_compressions + 1)),
        depths=[1] * n_spatial_compressions,
        interpolate=True,
    )
    model = get_vae(**model_settings)
    train_loader, val_loader, test_loader = get_loaders(**data_loader_settings)
    lr = initial_learning_rate
    optimizer, optim_args = get_opt(opt_name, lr, weight_decay)
    trainer_settings = dict(
        opt_name=opt_name,
        optimizer=optimizer,
        optim_args=optim_args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        batch_size=batch_size,
        training_epochs=training_epochs,
        schedule=CosineSchedule(decay_steps=decay_period, min_decay=min_decay),
        ae=ae,
        c_reg=c_reg,
        data_dir=data_dir
    )
    block_args = {**data_loader_settings, **model_settings, **trainer_settings}
    trainer = get_vae_trainer(model, exp_name, block_args)

    # loads last model
    if load == 0:
        trainer.load()
    elif load > 0:
        trainer.load(load)
    if n_gen:
        trainer.generate_samples(n_gen)
    if not model_eval:
        while training_epochs > trainer.epoch:
            trainer.train(checkpoint_every)
            trainer.save()
            trainer.test('val')
    else:
        trainer.test(partition='val', save_outputs=ind)
        for i in ind:
            trainer.viz_sample(i)
            trainer.generate_counterfactuals(i)


if __name__ == '__main__':
    main()
