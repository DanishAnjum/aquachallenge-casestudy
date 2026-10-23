# AQUA Platform — Technical Challenge: Self-Pruning Neural Network

Custom deep-learning framework and self-pruning neural network built from scratch in pure Python and NumPy for Tredence Inc.

## Overview
This repository contains a complete, zero-dependency (barring NumPy and Matplotlib) deep learning engine implementing:
1. **Reverse-mode autodiff** with topological sorting, broadcasting support, and numerical stability.
2. **Composable MLP layers** and an **Adam optimizer** built from scratch.
3. **Self-pruning mechanism** using First-Order Taylor Saliency, cubic sparsity scheduling, honest zero masking, and moment-freezing optimizer state protection for weight regrowth.
4. **Rigorous Pareto evaluation** comparing saliency pruning against pure magnitude pruning across multiple random seeds.


Key Features
Zero-Dependency Autodiff: Custom reverse-mode automatic differentiation engine supporting dynamic graph construction and backpropagation.
Honest Structural Masking: Binary masks ($M \in \{0, 1\}$) integrated into the forward pass and backed by rigorous chain-rule gradient scaling ($\nabla_W \mathcal{L} = \nabla_{\text{active}} \mathcal{L} \odot M$).
First-Order Taylor Saliency: Approximates weight importance via $|W \cdot \nabla_W \mathcal{L}|$ with global top-k pooling, stable tie-breaking (np.argsort with kind='stable'), and cubic sparsity scheduling.
Permanent Pruning & Optimizer State Reset: Locks pruned weights permanently at zero and clears Adam momentum ($m$) and velocity ($v$) states for newly pruned parameters.
Rigorous Empirical Evaluation: Paired 5-seed batch schedules comparing Taylor Saliency against Global Magnitude Pruning, complete with automated CSV logging and error-bar plotting.


---

## Project Structure

```text
tredence-casestudy/
│
├── engine.py              # Core zero-dependency autodiff engine & Tensor DAG
├── nn_model.py            # Neural network layers, MLP definitions, and activations
├── self_prune.py          # PrunedLinear layer, MaskedAdam, & Taylor Saliency logic
├── run_self_prune.py      # Single-run training script with baseline comparison
├── pareto_eval.py         # 5-seed Pareto sweep, CSV export, & plotting
├── requirements.txt       # Minimal dependencies (numpy, matplotlib)
└── results/               # Generated evaluation artifacts
    ├── pareto_results.csv # Raw sweep metrics across 5 seeds
    └── sparsity_accuracy.png # High-resolution Pareto frontier plot


---

## Installation

Clone the repository and install dependencies using `pip`:

```bash
git clone [https://github.com/DanishAnjum/aquachallenge-casestudy.git](https://github.com/DanishAnjum/aquachallenge-casestudy.git)
cd aquachallenge-casestudy
pip install -r requirements.txt

Run MLP
python train_mlp.py

Run Single Training & Baseline Experiment
python run_self_prune.py

Run 5-Seed Pareto Sweep & Generate Visualizations
python pareto_eval.py

Outputs will be saved directly to the results/ directory.



