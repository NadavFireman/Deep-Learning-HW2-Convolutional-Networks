# Deep Learning - Convolutional Networks (HW2)
**Home Assignment 2 (M.Sc. Data Science, HIT). Implementing convolution and max-pooling layers from scratch in PyTorch — with analytic gradients verified by numeric gradient checking — then designing and tuning a custom CNN that reaches 87.61% test accuracy on CIFAR-10 after a two-stage grid search over 324 configurations.**

## Overview
This assignment has two parts. The first builds the core operations of a convolutional network from the ground up: a convolutional layer and a max-pooling layer, each with a naive forward pass and a manually derived backward pass, validated against numeric gradient approximations. The second part moves to PyTorch's high-level API to design, tune, and train a custom CNN (`MyCNN`) for CIFAR-10 classification — including data augmentation, a hierarchical hyperparameter grid search, a full training protocol with early stopping, and error analysis of the final model.

## Key Features
- **Convolutional Layer from Scratch:** Naive forward pass supporting stride and zero-padding, plus a manually derived backward pass — both verified against a centered finite-difference numeric approximation.
- **Max-Pooling from Scratch:** Forward and backward passes, with gradients routed only through the max position of each pooling window.
- **Image Processing via Convolutions:** Hand-crafted kernels (grayscale conversion, edge detection) applied to real images through the custom Conv layer as an implementation sanity check.
- **Custom CNN (`MyCNN`):** Three Conv(3×3) → BatchNorm → GELU → MaxPool blocks followed by an FC → Dropout → FC head (~1.8M parameters).
- **Data Augmentation:** Random crop + horizontal flip applied through a custom `Dataset` wrapper — training set only.
- **Hierarchical Grid Search:** Two-stage tuning — Stage 1 over optimization hyperparameters (learning rate, weight decay, dropout, LR decay; 81 combinations), then Stage 2 over architecture (filters per block, hidden dim, batch size; 243 combinations). Stage 2 confirmed the standard channel-doubling architecture was already near-optimal.
- **Full Training Protocol:** AdamW + CrossEntropyLoss with exponential LR decay, 20,000 iterations (~128 epochs), early stopping (patience 40), and best-model checkpointing by validation accuracy.
- **Evaluation & Error Analysis:** Training-curve analysis, held-out test evaluation, and a per-class confusion matrix.

## Results
- **Final test accuracy: 87.61%** (best validation accuracy: 88.11%) — a ~0.5% val–test gap, indicating the model generalizes well beyond the data used for tuning.
- Strongest classes: car, frog, ship and truck (91%+); hardest: cat and dog, with cat ↔ dog as the largest single confusion — consistent with known CIFAR-10 behavior.

## Tech Stack
- **Language:** Python
- **Framework:** PyTorch (torchvision transforms for augmentation; core conv/pool gradients implemented manually)
- **Evaluation & Visualization:** scikit-learn (confusion matrix), Matplotlib
- **Dataset:** CIFAR-10
- **Environment:** Google Colab (GPU)

## Repository Structure
- `convolutional_networks.ipynb`: Walkthrough notebook — from-scratch layers, gradient checks, image-processing demo, and the full `MyCNN` tuning/training/evaluation pipeline.
- `convolutional_networks.py`: Implementation of the `Conv` and `MaxPool` layers (naive forward/backward).
- `a3_helper.py`: CIFAR-10 loading and plotting/visualization helpers.
- `eecs598/`: Course helper package — data loading (`data.py`), numeric gradient utilities (`grad.py`), training solver (`solver.py`), visualization and assignment helpers (`utils.py`, `vis.py`), and submission tooling (`submit.py`).
- `pytorch_quick_start.ipynb`: PyTorch quickstart reference notebook (MNIST walkthrough) linked from the assignment as a starting point.

## Acknowledgment
The assignment scaffolding and helper package (`eecs598`) are based on the course materials from the University of Michigan's EECS 498/598 *Deep Learning for Computer Vision* course.
