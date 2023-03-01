import argparse
from argparse import ArgumentTypeError


def bounded_num(numeric_type, imin=None, imax=None):
    def check_lower_bound(value):
        try:
            number = numeric_type(value)
            if imin is not None and number < imin:
                raise ArgumentTypeError(f'Input {value} must be greater than {imin}.')
            if imax is not None and imax < number:
                raise ArgumentTypeError(f'Input {value} must be smaller than {imin}.')
        except ValueError:
            raise ArgumentTypeError(f'Input incompatible with type {numeric_type.__name__}')
        except TypeError:
            raise ArgumentTypeError(f'{numeric_type.__name__} does not support inequalities')
        return number

    return check_lower_bound


def parse_main_args():
    parser = argparse.ArgumentParser(description='Counterfactual VAE options')

    parser.add_argument('--exp', type=str, default='', help='Name of the experiment')
    parser.add_argument('--dir_path', type=str, default='/scratch/dataset',
                        help='Directory for storing data and models')
    parser.add_argument('--batch_size', type=bounded_num(int, imin=1), default=64)
    parser.add_argument('--init_resolution', type=int, default=129, choices=[129], help='Only one option for now')
    parser.add_argument('--n_spatial_compressions', type=int, default=5, choices=[4, 5])
    parser.add_argument('--optim', type=str, default='AdamW', choices=['SGD', 'SGD_momentum', 'Adam', 'AdamW'],
                        help='SGD_momentum has momentum = 0.9')
    parser.add_argument('--lr', type=bounded_num(float, imin=0), default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=bounded_num(float, imin=0), default=0.00001, help='Weight decay')
    parser.add_argument('--min_decay', type=bounded_num(float, imin=0), default=0.01,
                        help='fraction of the initial lr at the end of train')
    parser.add_argument('--c_reg', type=bounded_num(float, imin=0), default=0.05, help='Coefficient for regularization')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Runs on CPU')
    parser.add_argument('--epochs', type=bounded_num(int, imin=1), default=350, help='Number of total training epochs')
    parser.add_argument('--decay_period', type=bounded_num(int, imin=0), default=250,
                        help='Number of epochs before lr decays stops')
    parser.add_argument('--checkpoint', type=bounded_num(int, imin=1), default=10,
                        help='Number of epochs between checkpoints')
    parser.add_argument('--ind', type=bounded_num(int, imin=0), default=[0], nargs='+',
                        help='index for reconstruction to visualize and counterfact')
    parser.add_argument('--load', type=bounded_num(int, imin=-1), default=-1,
                        help='Load a saved model with the same settings. -1 for starting from scratch,'
                             '0 for most recent, otherwise epoch after which the model was saved')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the model)')
    parser.add_argument('--gen', type=bounded_num(int, imin=0), default=0, help='Generate /number/ random samples)')

    return parser


# TODO: write common arguments only once
def parse_mlp_args():
    parser = argparse.ArgumentParser(description='MLP options')

    parser.add_argument('--exp', type=str, default='', help='Name of the experiment')
    parser.add_argument('--dir_path', type=str, default='/scratch/dataset',
                        help='Directory for storing data and models')
    parser.add_argument('--batch_size', type=bounded_num(int, imin=1), default=64)
    parser.add_argument('--init_resolution', type=int, default=129, choices=[129], help='Only one option for now')
    parser.add_argument('--optim', type=str, default='AdamW', choices=['SGD', 'SGD_momentum', 'Adam', 'AdamW'],
                        help='SGD_momentum has momentum = 0.9')
    parser.add_argument('--lr', type=bounded_num(float, imin=0), default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=bounded_num(float, imin=0), default=0.00001, help='Weight decay')
    parser.add_argument('--min_decay', type=bounded_num(float, imin=0), default=0.01,
                        help='fraction of the initial lr at the end of train')
    parser.add_argument('--c_reg', type=bounded_num(float, imin=0), default=0.05,
                        help='Coefficient for regularization')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Runs on CPU')
    parser.add_argument('--epochs', type=bounded_num(int, imin=1), default=350,
                        help='Number of total training epochs')
    parser.add_argument('--decay_period', type=bounded_num(int, imin=0), default=250,
                        help='Number of epochs before lr decays stops')
    parser.add_argument('--checkpoint', type=bounded_num(int, imin=1), default=10,
                        help='Number of epochs between checkpoints')
    parser.add_argument('--load', type=bounded_num(int, imin=-1), default=-1,
                        help='Load a saved model with the same settings. -1 for starting from scratch,'
                             '0 for most recent, otherwise epoch after which the model was saved')
    parser.add_argument('--pretrained', action='store_true', default=False, help='from AE model with same exp name)')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the model)')

    return parser


# TODO: write common arguments only once
def parse_ae_args():
    parser = argparse.ArgumentParser(description='AE options')

    parser.add_argument('--exp', type=str, default='', help='Name of the experiment')
    parser.add_argument('--dir_path', type=str, default='/scratch/dataset',
                        help='Directory for storing data and models')
    parser.add_argument('--batch_size', type=bounded_num(int, imin=1), default=64)
    parser.add_argument('--init_resolution', type=int, default=129, choices=[129], help='Only one option for now')
    parser.add_argument('--optim', type=str, default='AdamW', choices=['SGD', 'SGD_momentum', 'Adam', 'AdamW'],
                        help='SGD_momentum has momentum = 0.9')
    parser.add_argument('--lr', type=bounded_num(float, imin=0), default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=bounded_num(float, imin=0), default=0.00001, help='Weight decay')
    parser.add_argument('--min_decay', type=bounded_num(float, imin=0), default=0.01,
                        help='fraction of the initial lr at the end of train')
    parser.add_argument('--c_reg', type=bounded_num(float, imin=0), default=0.05,
                        help='Coefficient for regularization')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Runs on CPU')
    parser.add_argument('--epochs', type=bounded_num(int, imin=1), default=350,
                        help='Number of total training epochs')
    parser.add_argument('--decay_period', type=bounded_num(int, imin=0), default=250,
                        help='Number of epochs before lr decays stops')
    parser.add_argument('--checkpoint', type=bounded_num(int, imin=1), default=10,
                        help='Number of epochs between checkpoints')
    parser.add_argument('--load', type=bounded_num(int, imin=-1), default=-1,
                        help='Load a saved model with the same settings. -1 for starting from scratch,'
                             '0 for most recent, otherwise epoch after which the model was saved')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the model)')

    return parser

# TODO: write common arguments only once
def parse_svc_args():
    parser = argparse.ArgumentParser(description='SVC options')

    parser.add_argument('--exp', type=str, default='', help='Name of the experiment')
    parser.add_argument('--dir_path', type=str, default='/scratch/dataset',
                        help='Directory for storing data and models')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf', 'poly'],
                        help='kernel SVC')
    parser.add_argument('--C', type=bounded_num(float, imin=0), default=100, help='Regularization parameter')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the model)')

    return parser