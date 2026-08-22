# FC Counterfactuals

This repository contains the codebase to replicate the counterfactual connectome generation experiments, including the exact metrics ($D_{KL}$, MSE, CFD, and DBS) presented in the accompanying article: *"Counterfactual Functional Connectomes for Neurological Classifier Selection"* (Vercheval et al., EUSIPCO 2023).

## Setup
Ensure that your Python virtual environment is activated and all required dependencies (e.g., PyTorch, Scikit-Learn, Nilearn) are installed.

## Dataset
The project uses the **ABIDE PCP** dataset. The code is designed to automatically download and preprocess the dataset using `nilearn` if it is not found locally. 

To specify where the dataset and models should be stored:
- Create a `.env` file in the root of the repository containing the absolute path to your desired data directory (e.g., `DATA_DIR=/scratch/dataset/abide_fc_dataset`).
- Alternatively, you can pass the path via the `--data_dir` argument when running the scripts.

---

## 1. Train the Classifiers (Conditioning Models)

First, train the three classifiers that will provide the target conditions for the VAE models. Note that the Autoencoder (AE) model is an MLP pre-trained as an unsupervised autoencoder before being fine-tuned as a classifier.

```bash
# 1. Train the Support Vector Classifier (SVC)
python train_svc.py --exp final

# 2. Train the Multi-Layer Perceptron (MLP)
python train_mlp.py --exp final

# 3. Train the Autoencoder (AE)
# First train the unsupervised backbone:
python train_ae.py --exp final
# Then fine-tune the classifier using the pre-trained weights:
python train_mlp.py --exp final --pretrain
```
*Note: Running these commands automatically saves the classification probabilities to `metadata.csv`.*

---

## 2. Train the Counterfactual VAEs

Once the classifiers are trained, train a separate VAE conditioned on each of the three models:

```bash
# VAE conditioned on SVC
python main.py --exp final --cond svc

# VAE conditioned on MLP
python main.py --exp final --cond mlp

# VAE conditioned on AE
python main.py --exp final --cond ae
```

---

## 3. Replicate the Evaluation Metrics

The results in the article (Table 5.2) combine two different counterfactual target scenarios. 
You can switch between these targets using the newly added `--target` flag.

### Replicating CFD, $D_{KL}$, and MSE
The **CFD** (Counterfactual Distance) metric measures how far the intervention pushes the classifier's probability when explicitly targeting the **opposite class** ($0.0$ or $1.0$).

Run the evaluation using the default target strategy:
```bash
python main.py --exp final --cond svc --eval --eval_counterfactuals --load 0 --target opposite
python main.py --exp final --cond mlp --eval --eval_counterfactuals --load 0 --target opposite
python main.py --exp final --cond ae --eval --eval_counterfactuals --load 0 --target opposite
```
*(This will reproduce the CFD values: $45.9$ for SVC, $54.9$ for MLP, and $54.9$ for AE).*

### Replicating DBS
The **DBS** (Decision Boundary Success) metric reports the proportion of successes when the intervention targets a **neutral boundary** ($0.5$).

Run the evaluation again, this time explicitly targeting the neutral boundary:
```bash
python main.py --exp final --cond svc --eval --eval_counterfactuals --load 0 --target half
python main.py --exp final --cond mlp --eval --eval_counterfactuals --load 0 --target half
python main.py --exp final --cond ae --eval --eval_counterfactuals --load 0 --target half
```
*(The "Success percentage" in the console output will reproduce the DBS values: $63.6\%$ for SVC, $47.1\%$ for MLP, and $46.4\%$ for AE).*

---

## 4. Results Summary

Below is the summary of the replicated metrics corresponding to the values published in the article:

| Condition Model | $D_{KL}$ | MSE ($\times 10^{-2}$) | CFD (Opposite) | DBS (Neutral) |
|-----------------|----------|------------------------|----------------|---------------|
| **SVC**         | 35.82    | 1.28                   | 45.9           | 63.6%         |
| **MLP**         | 34.91    | 1.27                   | 54.9           | 47.1%         |
| **AE**          | 35.36    | 1.27                   | 54.9           | 46.4%         |

---

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
