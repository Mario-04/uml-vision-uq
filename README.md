# MC Dropout Uncertainty Estimation on CIFAR-10/100

**Marinus van den Ende (s5460484), David van Wuijkhuijse (s5592968), Adelina Mazilu (s5484669)**

Uncertainty in ML - Assignment 3

## Overview

This project investigates whether Monte Carlo (MC) Dropout improves uncertainty estimation and calibration in CNN-based image classification, and whether its uncertainty estimates are useful for near out-of-distribution (near-OOD) detection.

A CNN is trained on CIFAR-10 and compared against a deterministic baseline. At test time, MC Dropout keeps dropout active and performs multiple stochastic forward passes per image. The mean prediction is used for classification; variance and entropy serve as uncertainty measures.

## Method

- **Model**: CNN with dropout layers (Keras, Categorical Cross-Entropy loss)
- **Baseline**: Standard CNN with deterministic softmax prediction
- **MC Dropout**: Multiple stochastic forward passes at test time; mean for classification, variance/entropy for uncertainty

## Metrics

| Task | Metrics |
|------|---------|
| Classification | Accuracy, Brier score |
| Calibration | Expected Calibration Error (ECE), reliability plots |
| Near-OOD detection | AUROC, AUPR, FPR@95% |

## Datasets

- **CIFAR-10**: 10 classes, 6000 images per class — used for training, validation, testing, and calibration
- **CIFAR-100**: 100 classes, 600 images per class — used as OOD data for near-OOD detection

CIFAR-100 classes are mutually exclusive with CIFAR-10 classes, making it suitable as a near-OOD set.

## Project Structure

```
src/
  data/            # Data loading and preprocessing
  models/          # CNN and MC Dropout model definitions
  visualisations/  # Reliability plots and result figures
datasets/          # Raw dataset storage
main.py
```

## References

1. Will Cukierski. CIFAR-10 - Object Recognition in Images. https://kaggle.com/competitions/ cifar-10. Kaggle. 2013.
2. Alex Krizhevsky. Object Classification Experiments. Chapter 3 of technical report: Learning Multiple
Layers of Features from Tiny Images. University of Toronto, Apr. 2009, pp. 32–35. URL: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.