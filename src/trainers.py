import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch import autocast
import os
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from collections import UserDict
from src.dataset import get_loaders, plot_connectome
from src.models import get_model
from src.loss_and_metrics import MLPLoss, AELoss, VAEXLoss, MLPMetrics, ClassMetrics
from src.optim import get_opt, CosineSchedule


# Allows a temporary change using the with statement
class UsuallyFalse:
    _value = False

    def __bool__(self):
        return self._value

    def __enter__(self):
        self._value = True

    def __exit__(self, *_):
        self._value = False


# Apply recursively lists or dictionaries until check
def apply(obj, check, f):  # changes device in dictionary and lists
    if check(obj):
        return f(obj)
    elif isinstance(obj, list):
        obj = [apply(item, check, f) for item in obj]
    elif isinstance(obj, dict):
        obj = {k: apply(v, check, f) for k, v in obj.items()}
    else:
        raise ValueError(f' Cannot apply {f} on Datatype {type(obj)}')
    return obj


# Dict for (nested) list of Tensor
class TorchDictList(UserDict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.info = {}

    def __getitem__(self, key_or_index):
        if isinstance(key_or_index, int):
            return self._index_dict_list(key_or_index)
        return super().__getitem__(key_or_index)

    # Indexes a (nested) list in a dictionary
    def _index_dict_list(self, ind):
        out_dict = {}
        for k, v in self.items():
            if not v or isinstance(v[0], list):
                new_v = [elem[ind].unsqueeze(0) for elem in v]
            else:
                new_v = v[ind].unsqueeze(0)
            out_dict[k] = new_v
        return out_dict

    # Separates batch into list and appends (or creates) to structure dict of (nested) lists
    def extend_dict(self, new_dict):
        for key, value in new_dict.items():
            if isinstance(value, list):
                for elem, new_elem in zip(self.setdefault(key, [[] for _ in value]), value):
                    assert torch.is_tensor(new_elem)
                    elem.extend(new_elem)
            else:
                assert torch.is_tensor(value)
                self.setdefault(key, []).extend(value)


'''
This abstract class manages training and general utilities.
The outputs from the network are saved in a dictionary and stored in a list.
The dictionary also handles list of tensors as values.

The loss is an abstract method later defined and returns a dictionary dict with 
the different components of the criterion, plus eventually other useful metrics. 
dict['Criterion'] = loss to backprop (summed over batch)
'''


class Trainer(metaclass=ABCMeta):
    quiet_mode = UsuallyFalse()  # less output
    max_output = np.inf  # maximum amount of stored evaluated test samples

    def __init__(self, model, exp_name, device, optimizer, train_loader, val_loader=None,
                 test_loader=None, models_path='./models', amp=False, **block_args):

        self.epoch = 0
        self.device = device  # to cuda or not to cuda?
        self.model = model.to(device)
        self.exp_name = exp_name  # name used for saving and loading
        self.schedule = block_args['schedule']
        self.settings = {**model.settings, **block_args}
        self.optimizer_settings = block_args['optim_args'].copy()
        self.optimizer = optimizer(**self.optimizer_settings)
        self.scaler = GradScaler(enabled=amp and self.device.type == 'cuda')
        self.amp = amp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_losses, self.val_losses, self.test_losses = {}, {}, {}
        self.models_path = os.path.join(block_args['data_dir'], models_path)
        self.test_metadata, self.test_outputs = None, None  # store last test evaluation
        self.saved_metrics = {}  # saves metrics of last evaluation
        settings_path = self.paths()['settings']
        json.dump(self.settings, open(settings_path, 'w'), default=vars, indent=4)

    @property
    def optimizer_settings(self):  # settings shown depend on epoch
        if self.schedule is None:
            return {'params': self._optimizer_settings[0],
                    **self._optimizer_settings[1]}
        else:  # the scheduler modifies the learning rate(s)
            init_learning = self._optimizer_settings[0]
            scheduled_learning = []
            for group in init_learning:
                scheduled_learning.append({
                    'params': group['params'],
                    'lr': self.schedule(group['lr'], self.epoch)
                })
            return {'params': scheduled_learning,
                    **self._optimizer_settings[1]}

    @optimizer_settings.setter
    def optimizer_settings(self, optim_args):
        lr = optim_args.pop('lr')
        if isinstance(lr, dict):  # support individual lr for each parameter (for finetuning for example)
            self._optimizer_settings = \
                [{'params': getattr(self.model, k).parameters(), 'lr': v} for k, v in lr.items()], optim_args
        else:
            self._optimizer_settings = [{'params': self.model.parameters(), 'lr': lr}], optim_args
        return

    def update_learning_rate(self, new_lr):
        if not isinstance(new_lr, list):  # transform to list
            new_lr = [{'lr': new_lr} for _ in self.optimizer.param_groups]
        for g, up_g in zip(self.optimizer.param_groups, new_lr):
            g['lr'] = up_g['lr']
        return

    def train(self, num_epoch, val_after_train=False):
        if not self.quiet_mode:
            print('Experiment name ', self.exp_name)
        for _ in range(num_epoch):
            self.update_learning_rate(self.optimizer_settings['params'])
            self.epoch += 1
            if self.quiet_mode:
                print('\r====> Epoch:{:3d}'.format(self.epoch), end='')
            else:
                print('====> Epoch:{:3d}'.format(self.epoch))
            self.model.train()
            self._run_session(partition='train')
            if self.val_loader and val_after_train:  # check losses on val
                self.model.eval()
                with torch.inference_mode():
                    self._run_session(partition='val')
        return

    @torch.inference_mode()
    def test(self, partition, save_outputs=0, **kwargs):  # runs and stores evaluated test samples
        save_outputs = self.max_output if save_outputs else 0
        if not self.quiet_mode:
            print('Version ', self.exp_name)
        self.model.eval()
        self._run_session(partition=partition, save_outputs=save_outputs)
        return

    def _run_session(self, partition='train', save_outputs=0):
        if partition == 'train':
            loader = self.train_loader
            dict_losses = self.train_losses
        elif partition == 'val':
            loader = self.val_loader
            dict_losses = self.val_losses
        elif partition == 'test':
            loader = self.test_loader
            dict_losses = self.test_losses
        else:
            raise ValueError('partition options are: "train", "val", "test" ')
        if save_outputs:
            self.test_metadata, self.test_outputs = TorchDictList(), TorchDictList()
            self.test_metadata.info = dict(partition=partition, max_ouputs=self.max_output)

        epoch_loss = {}
        epoch_metrics = {}
        num_batch = len(loader)
        iterable = tqdm(enumerate(loader), total=num_batch, disable=self.quiet_mode)
        epoch_seen = 0
        for batch_idx, (inputs, targets, indices) in iterable:
            epoch_seen += indices.shape[0]
            inputs, targets = self.recursive_to([inputs, targets], self.device)
            inputs_aux = self.helper_inputs(inputs, targets)
            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                outputs = self.model(**inputs_aux)
            if torch.is_inference_mode_enabled():
                with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    batch_metrics = self.metrics(outputs, inputs, targets)
                for metric, value in batch_metrics.items():
                    epoch_metrics[metric] = epoch_metrics.get(metric, 0) + value.item()

            else:
                with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    batch_loss = self.loss(outputs, inputs, targets)
                criterion = batch_loss['Criterion']
                if torch.isnan(criterion):
                    raise ValueError('Criterion is nan')
                if torch.isinf(criterion):
                    raise ValueError('Criterion is inf')
                for loss, value in batch_loss.items():
                    epoch_loss[loss] = epoch_loss.get(loss, 0) + value.item()
                self.scaler.scale(criterion).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if not self.quiet_mode:
                    if batch_idx % (num_batch // 10 or 1) == 0:
                        iterable.set_postfix({'Seen': epoch_seen,
                                              'Loss': criterion.item()})
                    if batch_idx == num_batch - 1:  # clear after last
                        iterable.set_description('')
            if save_outputs > (batch_idx + 1) * loader.batch_size:
                self.test_outputs.extend_dict(self.recursive_to(outputs, 'detach_cpu'))
                self.test_metadata.extend_dict(dict(indices=indices))
                self.test_metadata.extend_dict(dict(targets=targets.cpu()))
        if torch.is_inference_mode_enabled():  # not averaged in batch
            self.saved_metrics = {metric: value / num_batch if metric == 'Criterion' else value / epoch_seen
                                  for metric, value in epoch_metrics.items()}
            print('Metrics:')
            for metric, value in self.saved_metrics.items():
                print('{}: {:.4e}'.format(metric, value), end='\t')
            print()
        else:
            epoch_loss = {loss: value / num_batch if loss == 'Criterion' else value / epoch_seen
                          for loss, value in epoch_loss.items()}
            for loss, value in epoch_loss.items():
                dict_losses.setdefault(loss, []).append(value)
            if not self.quiet_mode:
                print('Average {} losses :'.format(partition))
                for loss, value in epoch_loss.items():
                    print('{}: {:.4f}'.format(loss, value), end='\t')
                print()
        return

    @abstractmethod
    def loss(self, output, inputs, targets):
        pass

    def metrics(self, output, inputs, targets):
        return self.loss(output, inputs, targets)

    def helper_inputs(self, inputs, labels):
        return {'x': inputs}

    def plot_losses(self, loss):
        tidy_loss = ' '.join([s.capitalize() for s in loss.split('_')])
        epochs = np.arange(self.epoch)
        plt.plot(epochs, self.train_losses[loss], label='train')
        if self.val_loader:
            plt.plot(epochs, self.val_losses[loss], label='val')
        plt.xlabel('Epochs')
        plt.ylabel(tidy_loss)
        plt.title(f'{self.exp_name}')
        plt.show()
        return

    # Change device recursively to tensors inside a list or a dictionary
    @staticmethod
    def recursive_to(obj, device):  # changes device in dictionary and lists
        if device == 'detach_cpu':
            return apply(obj, check=torch.is_tensor, f=lambda x: x.detach().cpu())
        return apply(obj, check=torch.is_tensor, f=lambda x: x.to(device))

    def save(self, new_exp_name=None):
        self.model.eval()
        paths = self.paths(new_exp_name)
        torch.save(self.model.state_dict(), paths['model'])
        torch.save(self.optimizer.state_dict(), paths['optim'])
        json.dump(self.train_losses, open(paths['train_hist'], 'w'))
        json.dump(self.val_losses, open(paths['val_hist'], 'w'))
        if new_exp_name:
            json.dump(self.settings, open(paths['settings'], 'w'))
        print('Model saved at: ', paths['model'])
        return

    def load(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        else:
            past_epochs = []  # here it looks for the most recent model
            local_path = os.path.join(self.models_path, self.exp_name)
            if os.path.exists(local_path):
                for file in os.listdir(local_path):
                    if file[:5] == 'model':
                        past_epochs.append(int(''.join(filter(str.isdigit, file))))
            if not past_epochs:
                print('No saved models found')
                return
            else:
                self.epoch = max(past_epochs)
        paths = self.paths()
        self.model.load_state_dict(torch.load(paths['model'],
                                              map_location=torch.device(self.device)))
        self.optimizer.load_state_dict(torch.load(paths['optim'],
                                                  map_location=torch.device(self.device)))
        self.train_losses = json.load(open(paths['train_hist']))
        self.val_losses = json.load(open(paths['val_hist']))
        print('Loaded: ', paths['model'])
        return

    def paths(self, new_exp_name=None, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        if not os.path.exists(self.models_path):
            os.mkdir(self.models_path)
        if new_exp_name:  # save a parallel version to work with
            directory = os.path.join(self.models_path, new_exp_name)
        else:
            directory = os.path.join(self.models_path, self.exp_name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        paths = {'settings': os.path.join(directory, 'settings.json'),
                 'model': os.path.join(directory, f'model_epoch{epoch}.pt'),
                 'optim': os.path.join(directory, f'optimizer_epoch{epoch}.pt'),
                 'train_hist': os.path.join(directory, 'train_losses.json'),
                 'val_hist': os.path.join(directory, 'val_losses.json')}
        return paths


class MLPTrainer(Trainer):

    def __init__(self, model, exp_name, block_args):
        super().__init__(model, exp_name, **block_args)
        self._loss = MLPLoss()
        self._metrics = MLPMetrics()
        return

    def loss(self, output, inputs, targets):
        return self._loss(output, inputs, targets)

    def metrics(self, output, inputs, targets):
        return self._metrics(output, inputs, targets)

    def helper_inputs(self, inputs, labels):
        return {'x': inputs[0]}

    def test(self, partition, save_outputs=0, **kwargs):
        super().test(partition, save_outputs, **kwargs)
        if save_outputs:
            pred = torch.stack(self.test_outputs['logits']) > 0
            targets = torch.cat(self.test_metadata['targets'])
            ClassMetrics()(targets, pred)
        return

    def update_metadata(self, partition, prob_name):
        self.test(partition=partition, save_outputs=True)
        loader = self.train_loader if partition == 'train' else self.test_loader
        metadata = loader.dataset.metadata
        indices = torch.stack(self.test_metadata['indices'])
        probs = torch.sigmoid(torch.cat(self.test_outputs['logits']))
        metadata[prob_name].values[indices] = probs
        return metadata


class AETrainer(Trainer):

    def __init__(self, model, exp_name, block_args):
        super().__init__(model, exp_name, **block_args)
        self._loss = AELoss()
        return

    def loss(self, output, inputs, targets):
        return self._loss(output, inputs, targets)

    def helper_inputs(self, inputs, labels):
        return {'x': inputs[0], 'condition': inputs[1]}

    def viz_sample(self, ind):
        assert self.test_metadata is not None, 'Need to test a dataset first'
        part = self.test_metadata.info['partition']
        loader = self.test_loader if part == 'test' else self.val_loader if part == 'val' else self.train_loader
        metadata = self.test_metadata[ind]
        index = metadata['indices']
        [orig, prob], label, _ = loader.dataset[index]
        orig_data = self.test_outputs[ind]
        mse = self.metrics(orig_data, [orig.unsqueeze(0), prob.unsqueeze(0)], label)['MSE']
        print(f'Label: {label.item():.3f} -  Associated probability: {prob.item():.3f} -  MSE: {mse.item():.3f}')
        recon = orig_data['recon'].squeeze(0)
        if not os.path.exists('images'):
            os.mkdir('images')
        fig, subplots = plt.subplots(1, 2)
        for ind, (name, features) in enumerate({'Original': orig, 'Reconstruction': recon}.items()):
            ax = subplots[ind]
            img = self.matrix_from_feature(loader.dataset.res, features)
            ax.imshow(img, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(name)
            if ind == 1:
                fig.savefig(os.path.join('images', 'reconstruction_' + str(ind // 2)))
                plt.close()

    @staticmethod
    def matrix_from_feature(res, features):
        matrix = np.zeros((res, res))
        matrix[np.triu_indices(res, k=1)] = features
        matrix += matrix.transpose()
        matrix[np.diag_indices(res)] = 1
        return matrix


class VAETrainer(AETrainer):

    def __init__(self, model, exp_name, block_args):
        super().__init__(model, exp_name, block_args)
        self._loss = VAEXLoss(block_args['c_reg'])
        return

    @torch.inference_mode()
    def generate_samples(self, n_samples, condition=0.):
        sample = torch.randn(n_samples, self.model.z_dim, device=self.device).repeat_interleave(2, dim=0)
        condition = torch.zeros((n_samples, 1), device=self.device)
        condition = torch.cat((condition, torch.ones_like(condition)), dim=1).view(2 * n_samples, 1)
        sample[:, :1] = condition
        self.model.eval()
        gen_samples = self.model.decoder(data={}, sample=sample, condition=condition)
        gen_samples = gen_samples['recon'].cpu()
        fig, subplots = plt.subplots(1, 2)
        for ind, features in enumerate(gen_samples):
            ax = subplots[ind % 2]
            img = self.matrix_from_feature(self.train_loader.dataset.res, features)
            ax.imshow(img, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(ind % 2)
            if ind % 2 == 1:
                fig.savefig(os.path.join('images', 'generated_' + str(ind // 2)))
                plt.close()
                fig, subplots = plt.subplots(1, 2)
        plt.close(fig)

    @torch.inference_mode()
    def generate_counterfactuals(self, ind, counterfactual_prob=None, plot=True, show=True):
        part = self.test_metadata.info['partition']
        loader = self.test_loader if part == 'test' else self.val_loader if part == 'val' else self.train_loader
        metadata = self.test_metadata[ind]
        index = metadata['indices']
        [orig, prob], label, _ = loader.dataset[index]

        counterfactual_prob = np.round(prob < 0.5) if counterfactual_prob is None else counterfactual_prob
        if not show:
            print(f'Label: {label.item():.3f} -  Associated probability: {prob.item():.3f} -  '
                  f'Counterfactual probability: {counterfactual_prob.item():.3f}')
        self.model.eval()
        [orig, prob] = self.recursive_to([orig, prob], self.device)
        data = self.model.encoder(orig.unsqueeze(0), prob.unsqueeze(0))
        data_copy = {k: apply(v, check=torch.is_tensor, f=lambda x: x.clone()) for k, v in data.items()}
        data_copy['z'][-1][:, -1] = counterfactual_prob * self._loss.scale_prior
        gen_samples_rec = self.model.decoder(data=data, condition=torch.tensor([[prob]], device=self.device))
        gen_samples_rec = gen_samples_rec['recon'].cpu()[0]
        gen_samples_con = self.model.decoder(data=data_copy,
                                             condition=torch.tensor([[counterfactual_prob]], device=self.device))
        gen_samples_con = gen_samples_con['recon'].cpu()[0]

        orig = orig.cpu()
        counterfactual = orig - gen_samples_rec + gen_samples_con
        if plot:
            if not os.path.exists('images'):
                os.mkdir('images')
            fig, subplots = plt.subplots(2, 2, figsize=[15, 10])

            for i, (name, features, label) in enumerate(zip(['Original', 'Counterfactual'], [orig, counterfactual],
                                                            [prob.item(), counterfactual_prob.item()])):
                ax = subplots[0][i]
                img = self.matrix_from_feature(loader.dataset.res, features)
                ax.imshow(img, cmap='cool' + 'warm', vmin=-1, vmax=1)
                ax.set_title(name)
                ax = subplots[1][i]
                ax.set_title(f'Target Probability: {label:.3f}')
                plot_connectome(img, label, os.path.dirname(self.models_path), res=loader.dataset.res, fig=fig, ax=ax)
                if i == 1:
                    fig.savefig(os.path.join('images', 'counterfactual_' + str(ind)))
                    if not show:
                        plt.close()
        return counterfactual

    def evaluate_counterfactuals(self, model, s=0):
        part = self.test_metadata.info['partition']
        loader = self.test_loader if part == 'test' else self.val_loader if part == 'val' else self.train_loader
        estimated_probs = np.concatenate([inputs[0][1] for inputs in loader.dataset])
        counterfactuals = np.concatenate([self.generate_counterfactuals(i, plot=False, show=False)
                                          for i in range(len(loader.dataset))])
        counterfactuals_estimated_prob = model(counterfactuals.reshape(len(estimated_probs), -1))
        estimated_pred = np.greater(estimated_probs, 0.5)
        counterfactuals_estimated_pred = np.greater(counterfactuals_estimated_prob, 0.5)
        print('Success percentage: ', np.not_equal(estimated_pred, counterfactuals_estimated_pred).mean())


def get_trainer(args):
    if args.seed:
        torch.manual_seed = np.random.seed = args.seed
    train_loader, val_loader, test_loader = get_loaders(**vars(args))
    model = get_model(**vars(args))
    optimizer, optim_args = get_opt(args.optim, args.lr, args.wd)
    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    trainer_settings = dict(
        opt_name=args.optim,
        optimizer=optimizer,
        optim_args=optim_args,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        batch_size=args.batch_size,
        training_epochs=args.epochs,
        c_reg=args.c_reg,
        data_dir=args.data_dir,
        schedule=CosineSchedule(decay_steps=args.decay_period, min_decay=args.min_decay),
    )

    exp_name = '_'.join(filter(bool, args.name.lower().split(' ') + [args.exp]))
    trainer_mapping = {
        'AE': AETrainer,
        'MLP': MLPTrainer,
        'Counterfactual VAE': VAETrainer
    }
    trainer = trainer_mapping[args.name](model, exp_name, trainer_settings)

    # loads last model
    if args.load == 0:
        trainer.load()
    elif args.load > 0:
        trainer.load(args.load)

    return trainer
