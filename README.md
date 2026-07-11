# Deep Learning - Convolutional Networks (HW2)

**Home Assignment 2 (M.Sc. Data Science, HIT). Convolution and max-pooling layers implemented from scratch in PyTorch — analytic gradients verified numerically — then a custom CNN designed and tuned over a two-stage, 324-configuration grid search, reaching 87.61% test accuracy on CIFAR-10.**

## Key Features
- **Conv & MaxPool from Scratch:** Naive forward passes (stride, zero-padding) and manually derived backward passes, verified against centered finite differences — sanity-checked by applying hand-crafted grayscale and edge-detection kernels to real images.
- **Custom CNN (`MyCNN`):** Three Conv(3×3) → BatchNorm → GELU → MaxPool blocks + FC → Dropout → FC head (~1.8M parameters), with random-crop/flip augmentation on the training set.
- **Two-Stage Grid Search:** Stage 1 over optimization hyperparameters (81 combinations), Stage 2 over architecture (243) — confirming the channel-doubling design was already near-optimal.
- **Training Protocol:** AdamW with exponential LR decay, 20,000 iterations, early stopping (patience 40), and best-model checkpointing by validation accuracy.

## Results
Test accuracy **87.61%** (validation best: 88.11%) — a ~0.5% val–test gap. Strongest classes: car, frog, ship, truck (91%+); hardest: cat ↔ dog, the largest single confusion.

## Repository Structure
- `convolutional_networks.ipynb` / `.py`: Walkthrough notebook and the from-scratch `Conv`/`MaxPool` implementations.
- `a3_helper.py` / `eecs598/`: CIFAR-10 loading, gradient utilities, training solver, and visualization helpers.
- `pytorch_quick_start.ipynb`: PyTorch quickstart reference (MNIST).

## Acknowledgment
Scaffolding and the `eecs598` package are based on the University of Michigan's EECS 498/598 *Deep Learning for Computer Vision* course.
