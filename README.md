# Deep Learning-Assisted Geometric Extraction for Interfacial Characterization in Double-Emulsion Microfluidics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

Training code for the U-Net segmentation model used to extract the
dispersed-phase jet geometry in a double-emulsion flow-focusing
microfluidic system. The predicted masks are used for pixel-integrated
volume measurement and independent left/right interfacial-angle
extraction in the accompanying manuscript.

---

## Overview

The pipeline segments the dark dispersed-phase jet from top-down
microscope images and produces binary masks from which geometric and
interfacial quantities are subsequently derived. The model is trained
on manually annotated images and generalizes to unseen experimental
conditions.

Key features of the implementation:

- Four-channel input: RGB + CLAHE-enhanced grayscale channel
- Boundary-weighted binary cross-entropy combined with Tversky loss
  (asymmetric penalty against false negatives)
- Online augmentation: flips, rotation, zoom, translation, Gaussian
  blur, horizontal averaging, intensity shifts, additive noise
- Automatic mixed precision and gradient clipping
- Early stopping on validation Dice
- Reproducible training via fixed random seed

---

## Repository structure
