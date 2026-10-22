# FC Counterfactuals

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright - checked](https://img.shields.io/badge/basedpyright-checked-ffc000)](https://docs.basedpyright.com)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Code for *Counterfactual Functional Connectomes for Neurological Classifier Selection*
(Vercheval et al., EUSIPCO 2023). A hierarchical VAE, conditioned on the probability that
a classifier assigns to a subject, edits the functional connectome of the subject until the
classifier changes its decision. Comparing the counterfactuals of different classifiers
shows which connections each one relies on.

This version was rewritten using [drytorch](https://github.com/nverchev/drytorch).

## Setup

```console
uv sync --all-extras
```

Configure storage paths in a `.env` file (defaults are used if omitted):

- `DATA_DIR`: folder for datasets (default: `./data`). On the first run, the preprocessed ABIDE connectomes (~2.4 GB) are downloaded automatically to `<DATA_DIR>/abide_fc_dataset`.
- `ROOT_EXP_DIR`: root folder for experiment runs, checkpoints, and metrics (default: `./experiments`).

## Experimental Setting

### Task and Dataset
The goal is to diagnose Autism Spectrum Disorder (ASD) versus Typical Development (TD) from
resting-state functional MRI connectomes, and to generate counterfactual explanations that
reveal the connectivity features driving each classifier's decisions.

The experiments use preprocessed connectomes from the ABIDE dataset. Each connectome is
represented by the upper triangle of the correlation matrix between the 122 regions of the BASC
atlas, yielding $122 \times 121 / 2 = 7{,}381$ connection features per subject.

### Partitions
Subjects are divided into three random partitions:
- **Training** (64%): used to train the models;
- **Validation** (16%): used for model selection and checkpointing during development;
- **Test** (20%): held out for final evaluation.

By default, training scripts train on the training partition and evaluate on the validation
partition. Setting `final=true` trains on both training and validation subjects combined (80%)
and evaluates on the held-out test partition (20%), matching the setup in the published paper.

### Classifiers
Three distinct classifier architectures predict the diagnosis probability $p = P(\text{TD} \mid x) \in [0, 1]$:
- `svc` (default): a Support Vector Classifier with an RBF kernel (`C=100`, `gamma=auto`), whose
  probabilities come from a sigmoid fitted on cross-validated decision values
  (`CalibratedClassifierCV(ensemble=False)`);
- `mlp`: a multilayer perceptron classifier with fully connected hidden layers and ReLU activations;
- `ae`: the same MLP architecture, but with its encoder pretrained using an autoencoder on the
  connectome reconstruction task.

### Counterfactual Generation
A hierarchical conditional Variational Autoencoder (VAE) is trained for each classifier.
Conditioned on the probability $p$ that a classifier assigns to a subject, the VAE learns to
reconstruct the subject and edit their connectome toward a target probability $p^*$:

    counterfactual = clip(original + change, -1, 1)
    change = dec(z*, p*) - dec(z, p)

Keeping the reconstruction residual ($\text{residual} = \text{original} - \text{dec}(z, p)$)
preserves the subject-specific connectivity that the VAE does not reconstruct.

### Evaluation Targets and Metrics
The VAE is evaluated by its reconstruction, with **$D_{KL}$** and **MSE**, independently of the
target. Counterfactuals are evaluated under two target scenarios:
1. **Opposite Class** (`target=opposite`): The target probability is inverted relative to the
   decision boundary ($p^* = 0.0$ if $p \ge 0.5$, else $1.0$). Evaluated using **CFD**
   (Counterfactual Distance, the root mean square shift in classifier probability).
2. **Decision Boundary** (`target=half`): The target probability is set to the neutral boundary
   ($p^* = 0.5$). Evaluated using **DBS** (Decision Boundary Success, the fraction of subjects
   whose predicted classification is successfully flipped across the decision boundary).

---

## Running the Pipeline

Each classifier has its own entry configuration in `configs/experiment`: `svc.yaml` (the
default), `mlp.yaml` and `ae.yaml`, selected with `--config-name`. Each one keeps its runs in its
own folder, `<ROOT_EXP_DIR>/v<version>/<config>`.

Models are trained in sequence. Later stages locate the trained weights of earlier stages
automatically through the run name, which the overrides form. Always pass the same entry
configuration and overrides to all stages of a pipeline.

### 1. Final Pipeline

```console
# training; the ae classifier starts from the encoder of the autoencoder
uv run train_autoencoder.py --config-name ae final=true
for config in svc mlp ae; do
  uv run train_classifier.py --config-name $config final=true
  uv run train_counterfactual.py --config-name $config final=true
done

# evaluation on the test subjects
for config in svc mlp ae; do
  uv run evaluate_classifier.py --config-name $config final=true
  uv run evaluate_counterfactuals.py --config-name $config final=true
done
```

### 2. Metrics

`evaluate_classifier.py` publishes the accuracy, specificity and sensitivity of the classifier as
the source `Classification`, with its confusion matrix.

`evaluate_counterfactuals.py` publishes $D_{KL}$ and MSE as the source `Reconstruction`, and
the same metrics for the counterfactuals of each target, before and after denoising:
`Counterfactual_<target>` for the reconstructions under the target, $\text{dec}(z^*, p^*)$, and
`Denoised_counterfactual_<target>` for the counterfactuals that keep the residual, which removes
the reconstruction error. In these sources, `MSE` is the distance from the originals, not the
reconstruction error.

### 3. Visualization
Figures show the correlation matrix alongside the connectome graph, captioned with the true
diagnosis and the model's predicted probability:

```console
uv run visualize_counterfactuals.py final=true "user.plot.sample_indices=[0,5,7]"
```

Figures and metrics go to the configured trackers: Weights & Biases by default, and optionally
TensorBoard (`user.trackers.tensorboard=true`) or CSV files (`user.trackers.csv=true`).

---

## Results

Test results with `final=true`, compared to the published values (EUSIPCO 2023). CFD is reported
as the published code computed it, averaged over all subject pairs (`CFD_all_pairs`; the
per-subject CFD is `CFD`):

| Classifier | Values | D_KL | MSE (×10⁻²) | CFD | DBS |
| --- | --- | --- | --- | --- | --- |
| SVC | published | 35.82 | 1.28 | 45.9 | 63.6 % |
| SVC | reproduced | 35.69 | 1.28 | 45.1 | 65.7 % |
| MLP | published | 34.91 | 1.27 | 54.9 | 47.1 % |
| MLP | reproduced | 34.45 | 1.27 | 60.9 | 75.0 % |
| AE | published | 35.36 | 1.27 | 54.9 | 46.4 % |
| AE | reproduced | 34.66 | 1.27 | 65.8 | 62.9 % |

### Result Discussion
- **Reconstruction and Distance**: $D_{KL}$, MSE, and the CFD of the SVC consistently reproduce
  the published baseline within normal run-to-run stochastic variance.
- **Decision Boundary Sensitivity**: DBS exhibits higher variance across runs. Because the
  counterfactual targets $p^* = 0.5$, outputs cluster tightly around the decision boundary
  (typically within $\pm 0.04$). Whether an edited subject strictly crosses $0.5$ is therefore
  sensitive to small variations in training trajectory.
- **Classifier Saturation**: Neural classifiers (MLP and AE) produce sharper probability gradients
  near the decision boundary compared to the margin-based SVC, leading to larger probability shifts
  (higher CFD) when driven toward opposite classes.



## Reconstruction Orthogonality Analysis

A counterfactual applies a generated change to the original subject while preserving the
residual, the part the VAE does not reconstruct:

    counterfactual = clip(original + change)
    change = dec(z*, p*) - dec(z, p),  residual = original - dec(z, p)

The construction rests on an **orthogonality assumption**: the residual must be uncorrelated
with the change. The residual may carry class information, but it must not align with the direction
of the change; otherwise, preserving it would weaken or inflate the effect the metrics measure.

```console
uv run analysis_recon_orthogonality.py
uv run analysis_recon_orthogonality.py --classifiers svc mlp ae --overrides final=true
```

The script tests this assumption along three complementary dimensions:

- **G1 (Geometry)**: The cosine between the residual and the change is near zero ($|\cos| \le 0.007$)
  and matches a null baseline of residuals donated by other subjects ($p > 0.05$ on test).
  Its correlation with the class label is small ($\approx -0.14$ to $-0.26$).
- **G2 (Scale)**: The component of the residual projected along the change direction is near zero
  ($-0.03$ to $+0.08$ in units of the change, where $-1$ would mean the residual cancels the change).
- **G3 (Function)**: The classifier's logit response to the change with the residual is virtually
  identical to its response without it (attenuation ratio $\approx 0.93\text{--}1.04$, with a per-subject
  correlation of $r \approx 0.99\text{--}1.00$ between the two effects).

### Conclusion

The assumption holds: keeping the reconstruction residual preserves individual subject characteristics
without interfering with or attenuating the classifier's response to the counterfactual intervention.
The differences observed between classifiers are therefore driven by the model's targeted latent shift,
not by residual artifacts.


## Citation

If you use this codebase, please cite the following paper:

```bibtex
@inproceedings{vercheval2023counterfactual,
  author={Vercheval, Nicolas and Ben\v{c}evi\'{c}, Marin and Mu\v{z}evi\'{c}, Dario and Gali\'{c}, Irena and Pi\v{z}urica, Aleksandra},
  booktitle={2023 31st European Signal Processing Conference (EUSIPCO)},
  title={Counterfactual Functional Connectomes for Neurological Classifier Selection},
  year={2023},
  pages={1050--1054},
  doi={10.23919/EUSIPCO58844.2023.10289859}
}
```
