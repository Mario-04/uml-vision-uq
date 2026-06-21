# MC Dropout Uncertainty Estimation on CIFAR-10 (far-OOD: SVHN)

**Marinus van den Ende (s5460484), David van Wuijkhuijse (s5592968), Adelina Mazilu (s5484669)**

Uncertainty in ML - Assignment 3

## Overview

This project investigates whether Monte Carlo (MC) Dropout improves uncertainty estimation and calibration in CNN-based image classification, and whether its uncertainty estimates are useful for far out-of-distribution (far-OOD) detection. We use SVHN as a pure far-OOD test set against CIFAR-10 (we do not distinguish near-OOD / semantic-shift cases).

A CNN is trained on CIFAR-10 and compared against a deterministic baseline. At test time, MC Dropout keeps dropout active and performs multiple stochastic forward passes per image. The mean prediction is used for classification; variance and entropy serve as uncertainty measures.

## Method

- **Model**: CNN with dropout layers (Torch, Categorical Cross-Entropy loss)
- **Baseline**: Standard CNN with deterministic softmax prediction
- **MC Dropout**: Multiple stochastic forward passes at test time; mean for classification, variance/entropy for uncertainty

## Metrics

| Task | Metrics |
|------|---------|
| Classification | Accuracy, Brier score |
| Calibration | Expected Calibration Error (ECE), reliability plots |
| Far-OOD detection | AUROC, AUPR, FPR@95% |

## Datasets

- **CIFAR-10**: 10 classes, 6000 images per class — used for training, validation, testing, and calibration (the in-distribution set)
- **SVHN**: real-world house-number digit images — used as the far-OOD test set for OOD detection (CIFAR-10 = in-distribution, SVHN = out-of-distribution)

## Project Structure

```
src/
  data/            # Data loading and preprocessing
  models/          # CNN and MC Dropout model definitions
  visualisations/  # Reliability plots and result figures
  train.py         # File for training
  artifacts.py     # File for saving models
  evaluate.py      # File for evaluation
datasets/          # Raw dataset storage
artifacts/         # Raw model artifacts
main.py
```

## Running the Project

- ```python main.py --download_data``` to download CIFAR-10 and SVHN.

- ```python main.py --train --dropout_p 0.1 --epochs 20``` to train a CNN. `--dropout_p 0.0` gives the deterministic baseline; `--dropout_p > 0.0` enables MC-dropout at inference. The run is saved to `artifacts/`.

- ```python main.py --evaluate --run_dir <dir>``` to evaluate a model on the CIFAR-10 test set (accuracy, Brier, ECE, reliability diagram). Add `--mc_samples 30` to evaluate with MC-dropout instead of a deterministic single pass.

- ```python main.py --evaluate --ood --run_dir <dir> --mc_samples 30``` to run **far-OOD detection** (CIFAR-10 in-distribution vs SVHN out-of-distribution). The same trained model is scored deterministically (baseline) and with MC-dropout; AUROC / AUPR / FPR@95% are printed and an overlaid ROC curve is saved to `<dir>/ood_roc.png`.

Additional flags:
- ```--seed int``` to specify seed, otherwise defaults to 42.

- ```--dropout_p float``` to specify what probability to use for dropout.

- ```--epochs int``` number of training epochs.

- ```--mc_samples int``` number of stochastic forward passes (T) at evaluation; `1` = deterministic, `>1` = MC-dropout.

- ```--run_dir str``` to specify the directory path to a model.


## References

1. Will Cukierski. CIFAR-10 - Object Recognition in Images. https://kaggle.com/competitions/ cifar-10. Kaggle. 2013.
2. Alex Krizhevsky. Object Classification Experiments. Chapter 3 of technical report: Learning Multiple
Layers of Features from Tiny Images. University of Toronto, Apr. 2009, pp. 32–35. URL: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.