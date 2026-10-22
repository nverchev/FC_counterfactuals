# Guidelines

Recommendations for projects built on this template. Follow them unless there
is a good reason not to. The [README](README.md) explains how the template works:
the configuration, the run names and where runs are stored.

## Models

drytorch records the representation of the module as the architecture of the
model (`architecture.yaml` in the run's metadata). The representation lists the
modules, so a model documents itself only as far as it is made of modules.

- Build architectures from `nn.Sequential` and `nn.ModuleList`, not from layers
  stored one by one as attributes. A sequence states the order of its layers in
  one place, and its representation lists them in that order.
- Write every operation as a module, not as a call inside `forward`: reshaping,
  activations, normalization, pooling. When `torch.nn` has no module for an
  operation, write a small one, as `View` in `src/module/layers.py` does for a
  reshape.
- Give a module with settings an `extra_repr` that returns them, as
  `torch.nn.Linear` shows its dimensions: without it, the representation
  shows the module's name but not, for example, its temperature or the
  dimension it pools.
- Keep `forward` to the flow of the data: which module receives which tensor,
  and how branches are joined.

For example, prefer

```python
self.head = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_dim, h_dim),
    nn.ReLU(),
    nn.Linear(h_dim, n_classes),
)
```

to two linear layers stored separately, with the flattening and the activation
written in `forward`: there, the representation would show the two layers but
not the flattening or the activation between them.

### Model settings

- Give every role in the model (an encoder, a decoder, a classifier) an
  abstract base class, such as `BaseDecoder`, that reads the settings all its
  implementations use and stores them as public `Final` attributes. An
  implementation, such as `MyDecoder`, reads only the settings it adds and
  builds its layers from the stored ones. `BaseArchitecture` in
  `src/module/architectures.py` does this for the classifiers.
- Layers do not read the configuration: the module that builds them passes
  their settings, the activation and the normalization included, as
  arguments.
- Keep general modules in `src/module/layers.py`. A module that only one class
  uses, for a task specific to it, stays in the module of that class.

## Configuration

The run records its configuration, so the configuration must say exactly what
the run did.

### Settings

- Write the value of every setting in the YAML files, even when the dataclass
  has a default: Hydra accepts an override only for a key that a file already
  contains (see [Overrides](README.md#overrides)). Do not leave a value empty
  for the code to fill with a constant: the recorded configuration would then
  show an empty value instead of the setting used.
- Remove options that nothing reads. A recorded option suggests that it had an
  effect.
- Make the dataclasses of the specifications frozen. Change a configuration
  with overrides (`get_experiment_config(overrides=[...])`) or with
  `dataclasses.replace`, never by assignment, so the recorded configuration
  is the one the run used.
- List the choices of a setting in an `enum.StrEnum` with upper-case member
  names that take `enum.auto()`: `SYNTHETIC = enum.auto()`. pydantic reads a
  choice by its value, and `enum.auto()` makes the value the lower-case name,
  so the YAML file and what the run prints read `synthetic`. Write the value
  out only when it must match a name outside the code, such as a folder of the
  dataset or a key in a saved file.
- Only `user` settings stay out of the run name (`get_variation` in
  `src/config/experiment.py`): they change how a run is executed or shown,
  such as the device, the trackers or which samples to plot. The evaluation
  is part of the experiment, so its settings are tracked like the training
  ones; for another evaluation, write another entry point or test instead of
  overriding a setting.
- Leave the seed unset by default (`seed:` in `user/user.yaml`), and set it for
  the runs that must be repeatable.

### File names

- Give each configuration file a name that says what it holds without its
  folder: editor tabs, defaults lists and overrides show only the name, and a
  generic name hides which settings it changes. `adam.yaml`, `cosine.yaml`,
  `linear.yaml` and the name of a dataset are clear on their own;
  `default.yaml`, `standard.yaml` or a setting's name repeated in several
  groups (`vqvae.yaml` in `model/`, `train/` and `loss/`) are not.
- When a name would be generic, add the group: `train_vqvae.yaml`,
  `model_vqvae.yaml`. A group with a single file takes the group's name:
  `train/train.yaml`.
- The entry configurations, directly in `configs/experiment`, are not options
  of a group and keep plain names: `mlp_classifier.yaml` (see
  [Entry configurations](README.md#entry-configurations)).

## Experiments

- Give a setup whose runs should not be mixed with the others its own entry
  configuration, not only a variation: its runs then get their own folder
  (see [Folders](README.md#folders)), and the run name does not need to repeat
  the configuration.
- Pass `cfg.par_dir` as the parent directory of every `Experiment`, so that
  every script stores its runs in the folder of their entry configuration.
- Compare a run with published results only under the same protocol
  (see [`final`](README.md#overrides)): `final=false` and `final=true` give
  different numbers for the same model.

## Pipelines

Some projects train several models in sequence, each using the ones before it.

- Give each model its own phase (see
  [Groups and setting names](README.md#groups-and-setting-names)).
- Run every phase with the same overrides, apart from `user.*`: a later phase
  finds the checkpoint of an earlier model by the run name, which the
  overrides form. VAEX finds the classifier this way.
- When tuning a stage that uses a trained model, load that model once and reuse
  it in every trial: register it in each trial's experiment with
  `register_model`, and remove it with `unregister_model` when the trial ends.
  PCGen's `tune_w_autoencoder.py` does this.

## Visualization

- Choose the samples to show with
  `drytorch.lib.loading.take_from_dataset(dataset, indices=...)`, with the
  indices in the configuration, so that every run shows the same samples and
  the samples arrive already batched and on the device.
