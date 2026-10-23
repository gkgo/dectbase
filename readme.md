# DectBase: Few-Shot Strip Steel Surface Defect Detection Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

**DectBase** is an open-source deep learning benchmark and modular framework dedicated to **Few-Shot Strip Steel Surface Defect Detection**.

In real-world steel manufacturing and industrial quality inspection, collecting and annotating surface defect samples (such as crazing, inclusions, patches, pitted surfaces, rolled-in scale, and scratches) is costly and heavily constrained by extreme class imbalance. **DectBase** provides an out-of-the-box, reproducible, and high-precision few-shot object detection pipeline designed for data-scarce industrial visual inspection scenarios.

---

## Key Features

* **Industrial Few-Shot Learning Ready**: Optimized feature extraction, meta-learning, and two-stage fine-tuning strategies specifically tailored for extremely low-shot settings (1-shot, 2-shot, 3-shot, 5-shot, and 10-shot).
* **Multi-Scale & High-Aspect-Ratio Defect Perception**: Customized feature pyramid networks (FPN) and attention mechanisms designed to capture tiny point-like defects (e.g., pitted surfaces) and elongated cracks (e.g., crazing and scratches).
* **Cross-Modal Semantic Priors**: Supports integrating Vision-Language Model (VLM) and Large Language Model (LLM) fine-grained defect descriptions as semantic priors to boost low-shot generalization.
* **Standard Industrial Benchmarks**: Native support for popular strip steel surface defect datasets including **NEU-DET**, **GC10-DET**, and **Severstal**, with pre-configured few-shot split generators.
* **Modular & Extensible Architecture**: Easily swap backbones, necks, detection heads, and contrastive/augmentation modules via clean configuration files.

---

## Installation

```bash
# 1. Clone the repository
git clone [https://github.com/gkgo/dectbase.git](https://github.com/gkgo/dectbase.git)
cd dectbase

# 2. Create and activate a conda environment
conda create -n dectbase python=3.9 -y
conda activate dectbase

# 3. Install dependencies
pip install -r requirements.txt
