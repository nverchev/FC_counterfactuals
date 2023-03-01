import torch
import numpy as np
import os
from options import parse_mlp_args
from src.dataset import get_loaders
from src.optim import get_opt, CosineSchedule
from src.trainers import get_mlp_trainer
from src.models import get_mlp


def main():
    args = parse_mlp_args().parse_args()
    exp_name = 'mlp' + '_' * bool(args.exp) + args.exp
    dir_path = args.dir_path
    data_dir = os.path.join(dir_path, 'abide_fc_dataset')
    resolution = args.init_resolution
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
    load = args.load
    model_eval = args.eval
    torch.manual_seed = np.random.seed = 112358
    data_loader_settings = dict(
        data_dir=data_dir,
        batch_size=batch_size,
        resolution=resolution,
        model_eval=model_eval,
    )

    model_settings = dict(
        x_dim=resolution,
    )
    model = get_mlp(**model_settings)
    train_loader, val_loader, test_loader = get_loaders(**data_loader_settings)

    lr = initial_learning_rate
    optimizer, optim_args = get_opt(opt_name, lr, weight_decay)
    trainer_settings = dict(
        opt_name=opt_name,
        optimizer=optimizer,
        optim_args=optim_args,
        train_loader=train_loader,
        val_loader=None,  # FixMe
        test_loader=test_loader,
        device=device,
        batch_size=batch_size,
        training_epochs=training_epochs,
        schedule=None,
        c_reg=c_reg,
        data_dir=data_dir
    )
    block_args = {**data_loader_settings, **model_settings, **trainer_settings}
    trainer = get_mlp_trainer(model, exp_name, block_args)

    # loads last model
    if load == 0:
        trainer.load()
    elif load > 0:
        trainer.load(load)
    if not model_eval:
        while training_epochs > trainer.epoch:
            trainer.train(checkpoint_every)
            trainer.save()
            # trainer.test('val')
    else:
        trainer.test(partition='test', save_outputs=True)


if __name__ == '__main__':
    main()
