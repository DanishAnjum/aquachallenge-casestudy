# AQUA Platform — Technical Challenge: Self-Pruning Neural Network

Custom deep-learning framework and self-pruning neural network built from scratch in pure Python and NumPy for Tredence Inc.

## Overview
This repository contains a complete, zero-dependency (barring NumPy and Matplotlib) deep learning engine implementing:
1. **Reverse-mode autodiff** with topological sorting, broadcasting support, and numerical stability.
2. **Composable MLP layers** and an **Adam optimizer** built from scratch.
3. **Self-pruning mechanism** using First-Order Taylor Saliency, cubic sparsity scheduling, honest zero masking, and moment-freezing optimizer state protection for weight regrowth.
4. **Rigorous Pareto evaluation** comparing saliency pruning against pure magnitude pruning across multiple random seeds.

---

## Installation

Clone the repository and install dependencies using `pip`:

```bash
git clone [https://github.com/DanishAnjum/aquachallenge-casestudy.git](https://github.com/DanishAnjum/aquachallenge-casestudy.git)
cd aquachallenge-casestudy
pip install -r requirements.txt