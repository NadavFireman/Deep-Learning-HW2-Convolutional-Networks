# Deep Learning - Convolutional Networks (HW2)

**Home Assignment 2 (Grade 96, M.Sc. Data Science, HIT). Convolution and max-pooling layers implemented from scratch in PyTorch — analytic gradients verified numerically — then a custom CNN designed and tuned over a two-stage, 324-configuration grid search, reaching 87.61% test accuracy on CIFAR-10.**

## Key Features
- **Conv & MaxPool from Scratch:** Naive forward passes (stride, zero-padding) and manually derived backward passes, verified against centered finite differences — sanity-checked by applying hand-crafted grayscale and edge-detection kernels to real images.
- **Custom CNN (`MyCNN`):** Three Conv(3×3) → BatchNorm → GELU → MaxPool blocks + FC → Dropout → FC head (~1.8M parameters), with random-crop/flip augmentation on the training set.
- **Two-Stage Grid Search:** Stage 1 over optimization hyperparameters (81 combinations), Stage 2 over architecture (243) — confirming the channel-doubling design was already near-optimal.
- **Training Protocol:** AdamW with exponential LR decay, 20,000 iterations, early stopping (patience 40), and best-model checkpointing by validation accuracy.

## Results
Test accuracy **87.61%** (validation best: 88.11%) — a ~0.5% val–test gap. Strongest classes: car, frog, ship, truck (91%+); hardest: cat ↔ dog, the largest single confusion.

## Repository Structure
- `convolutional_networks.ipynb`: Walkthrough notebook — from-scratch layers, gradient checks, and the full `MyCNN` tuning/training/evaluation pipeline.
- `convolutional_networks.py`: Implementation of the `Conv` and `MaxPool` layers (naive forward/backward).
- `a3_helper.py`: CIFAR-10 loading and plotting helpers.
- `pytorch_quick_start.ipynb`: PyTorch quickstart reference (MNIST).
- `eecs598/` — Course helper package:
  - `__init__.py`: Package initialization and seed utilities.
  - `data.py`: CIFAR-10 downloading, loading and preprocessing.
  - `grad.py`: Numeric gradient checking (finite differences, relative error).
  - `solver.py`: Generic training-loop engine (optimization, checkpointing).
  - `utils.py`: General assignment utilities.
  - `vis.py`: Tensor and image visualization helpers.
  - `submit.py`: Submission packaging tooling.
