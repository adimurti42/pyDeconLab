# pyDeconLab

A high-performance Python framework for 3D microscopy image deconvolution with adaptive PSF estimation.

---

## Overview

**pyDeconLab** is a modular and efficient Python implementation of 3D deconvolution algorithms inspired by EPFL’s DeconvolutionLab2. It is designed to process large confocal microscopy stacks and reconstruct high-resolution volumes by reversing optical blur.

The pipeline supports multiple Point Spread Function (PSF) models, including experimental, physics-based, and data-driven estimation, making it flexible for real-world microscopy workflows.

---

## Features

- Richardson–Lucy (RL) deconvolution
- RL with Total Variation regularization (RLTV)
- FFT-accelerated convolution for large 3D stacks
- Blockwise processing for memory-efficient computation
- Multiple PSF modes:
  - Gaussian approximation
  - Gibson–Lanni optical model (via `pyotf`)
  - Measured PSF from bead stacks
  - Automatic PSF estimation from image data
- TIFF stack support with metadata preservation
- Modular architecture for easy extension

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/pyDeconLab.git
cd pyDeconLab
