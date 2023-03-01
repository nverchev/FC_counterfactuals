import argparse
import os.path
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


class CommonParser(argparse.ArgumentParser):
    # Defaults of specific Parser
    _defaults = {}

    def __init__(self, name):
        super().__init__(name)
        self.add_argument('--exp', type=str, default='', help='Name of the experiment')
        self.add_argument('--data_dir', type=str, default='', help='Directory for data and models')
        self.add_argument('--init_res', type=bounded_num(int, imin=1), default=444, help='Initial resolution')
        self.add_argument('--z_dim', type=bounded_num(int, imin=1), default=32, help='dim encoding/final features')
        self.add_argument('--batch_size', type=bounded_num(int, imin=1), default=64)
        self.add_argument('--optim', type=str, default='AdamW', choices=['SGD', 'SGD_momentum', 'Adam', 'AdamW'],
                          help='SGD_momentum has momentum = 0.9')
        self.add_argument('--lr', type=bounded_num(float, imin=0), default=0.001, help='Learning rate')
        self.add_argument('--wd', type=bounded_num(float, imin=0), default=0.00001, help='Weight decay')
        self.add_argument('--c_reg', type=bounded_num(float, imin=0), default=0.05,
                          help='Coefficient for regularization')
        self.add_argument('--epochs', type=bounded_num(int, imin=1), default=350,
                          help='Number of total training epochs')
        self.add_argument('--decay_period', type=bounded_num(int, imin=0), default=250,
                          help='Number of epochs before lr decays stops')
        self.add_argument('--min_decay', type=bounded_num(float, imin=0), default=1,
                          help='fraction of the initial lr at the end of train')
        self.add_argument('--checkpoint', type=bounded_num(int, imin=1), default=10,
                          help='Number of epochs between checkpoints')
        self.add_argument('--no_cuda', action='store_true', default=False, help='Runs on CPU')
        self.add_argument('--load', type=bounded_num(int, imin=-1), default=-1,
                          help='Load a saved model with the same settings. -1 for starting from scratch,'
                               '0 for most recent, otherwise epoch after which the model was saved')
        self.add_argument('--eval', action='store_true', default=False, help='Evaluate the model)')
        self.add_argument('--seed', type=bounded_num(int, imin=1), default=0, help='Torch/Numpy seed (0 no seed)')

        if os.path.exists('dataset_path.txt'):
            with open('dataset_path.txt', 'r') as f:
                self.set_defaults(data_dir=f.read())
        self.set_defaults(name=name, **self._defaults)


class MainParser(CommonParser):
    _defaults = {'lr': 0.01,
                 'min_decay': 0.01}

    def __init__(self):
        super().__init__('Counterfactual VAE')

        self.add_argument('--ind', type=bounded_num(int, imin=0), default=[0], nargs='+',
                          help='index for reconstruction to visualize and counterfact')
        self.add_argument('--gen', type=bounded_num(int, imin=0), default=0, help='Generate /number/ random samples)')


class MLPParser(CommonParser):

    def __init__(self):
        super().__init__('MLP')
        self.add_argument('--pretrain', action='store_true', default=False, help='load AE model with same exp name)')


class AEParser(CommonParser):

    def __init__(self):
        super().__init__('AE')


class SVCParser(CommonParser):

    def __init__(self):
        super().__init__('SVC')
        self.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'rbf', 'poly'], help='kernel SVC')
        self.add_argument('--C', type=bounded_num(float, imin=0), default=100, help='Regularization parameter')
        self.add_argument('--eval', action='store_true', default=False, help='Evaluate the model)')
