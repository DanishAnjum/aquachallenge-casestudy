---
# Design Document: Self-Pruning Neural Network Architecture

## 1. Importance Criterion Derivation (Taylor Saliency)
Magnitude-only pruning ($|w$) assumes small weights are unimportant, but a small weight with a massive gradient can be critical to the loss manifold. 
Using a First-Order Taylor expansion, the change in loss $\Delta L$ resulting from removing connection $w_{ij}$ is approximated as:
$$\Delta L_{ij} \approx |L(w_{ij} = 0) - L(w_{ij})| \approx \left| w_{ij} \cdot \frac{\partial L}{\partial w_{ij}} \right|$$
This saliency score captures both parameter magnitude and gradient sensitivity simultaneously. If either the weight magnitude or its gradient sensitivity is near zero, removing the connection incurs minimal loss increase.

---

## 2. Masked-Weight Gradient Treatment & Optimizer Moments
During the forward pass, weights are masked directly (`W * mask`), ensuring pruned connections contribute true zeros rather than near-zero values. 

To prevent Adam's first ($m$) and second ($v$) moment estimators from corrupting when gradients are zeroed out for pruned weights, our `MaskedAdam` optimizer multiplies incoming gradients by the binary mask (`grad = p.grad * mask`). This freezes historical momentum for dead connections (`mask == 0`). If a connection is subsequently revived (mask returns to `1`), its momentum state is safely preserved without erratic update spikes.

---

## 3. Autodiff Bottlenecks & Optimization
The primary bottleneck in a NumPy-based autodiff engine is memory allocation overhead during broadcasting and intermediate array creation (`_unbroadcast` and element-wise temporaries). 

In a production setting, this would be optimized by:
- Fusing element-wise operations into contiguous C/CUDA kernels to eliminate kernel launch and round-trip memory overhead.
- Implementing memory pooling / buffer reuse for gradient accumulators.

---

## 4. Serving at Scale in a Multi-Tenant Inference Service (AQUA)
To achieve actual wall-clock speedups and reduce GPU memory bandwidth pressure in a high-throughput multi-tenant inference service like AQUA, sparse models cannot rely on dense matrix multiplication masked by zeros (`x @ (W * mask)`), as hardware will still execute the full dense FLOP count.

Instead, production serving requires:
1. **Sparse Tensor Formats:** Compiling weights into sparse formats (such as Compressed Sparse Row/Column, CSR/CSC) or block-sparse tensor kernels.
2. **Hardware-Aware Kernels:** Utilizing specialized sparse tensor hardware acceleration (such as NVIDIA Ampere/Hopper Sparse Tensor Cores) that natively skips zero-value computations to deliver true wall-clock latency reduction and throughput scaling.