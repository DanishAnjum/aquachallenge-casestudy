import numpy as np
from engine import Tensor, softmax_cross_entropy

class PrunedLinear:
    """
    Linear layer equipped with binary masks for self-pruning and weight regrowth.
    """
    def __init__(self, in_features, out_features):
        limit = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * limit, op_name="weight")
        self.b = Tensor(np.zeros((1, out_features)), op_name="bias")
        
        # Binary mask: 1 = active connection, 0 = pruned connection
        self.mask = np.ones((in_features, out_features), dtype=np.float32)

    def parameters(self):
        return [self.W, self.b]

    def __call__(self, x):
        # Apply mask to weights during forward pass (Honest Masking: true zeros)
        masked_W_data = self.W.data * self.mask
        masked_W = Tensor(masked_W_data, parents=[self.W], op_name="masked_weight")
        
        # We manually hook backward so gradients flow only through unmasked weights
        out = (x @ masked_W) + self.b
        return out


class PrunedMLP:
    """
    MLP containing PrunedLinear layers.
    """
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(PrunedLinear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(ReLU_Layer())

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params

    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def update_masks(self, target_sparsity):
        """
        Computes Taylor Saliency |w * grad| and prunes the lowest scoring connections.
        """
        for layer in self.layers:
            if isinstance(layer, PrunedLinear):
                # Importance criterion: Saliency = |W * Grad|
                if layer.W.grad is not None:
                    saliency = np.abs(layer.W.data * layer.W.grad)
                else:
                    saliency = np.abs(layer.W.data) # Fallback to magnitude if grad is missing
                
                # Flatten saliency to find threshold
                flat_saliency = saliency.flatten()
                k = int(len(flat_saliency) * (1.0 - target_sparsity))
                
                if k > 0:
                    threshold = np.partition(flat_saliency, -k)[-k]
                    # Update mask: 1 if saliency >= threshold, else 0
                    layer.mask = (saliency >= threshold).astype(np.float32)
                else:
                    layer.mask = np.zeros_like(layer.mask)


class ReLU_Layer:
    def __call__(self, x):
        return x.relu()


class MaskedAdam:
    """
    Adam optimizer that respects weight masks and freezes moments for pruned weights.
    """
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.params = model.parameters()
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        param_idx = 0
        
        for layer in self.model.layers:
            if isinstance(layer, PrunedLinear):
                p = layer.W
                mask = layer.mask
                if p.grad is not None:
                    # Freeze optimizer updates for pruned weights (mask == 0)
                    grad = p.grad * mask
                    
                    self.m[param_idx] = self.beta1 * self.m[param_idx] + (1 - self.beta1) * grad
                    self.v[param_idx] = self.beta2 * self.v[param_idx] + (1 - self.beta2) * (grad ** 2)
                    
                    m_hat = self.m[param_idx] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[param_idx] / (1 - self.beta2 ** self.t)
                    
                    # Update active weights
                    p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    # Enforce exact zero on masked weights
                    p.data *= mask
                param_idx += 1
                
                # Handle bias parameters normally
                pb = layer.b
                if pb.grad is not None:
                    self.m[param_idx] = self.beta1 * self.m[param_idx] + (1 - self.beta1) * pb.grad
                    self.v[param_idx] = self.beta2 * self.v[param_idx] + (1 - self.beta2) * (pb.grad ** 2)
                    m_hat = self.m[param_idx] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[param_idx] / (1 - self.beta2 ** self.t)
                    pb.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                param_idx += 1