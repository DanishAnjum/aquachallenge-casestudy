import numpy as np
from engine import Tensor, softmax_cross_entropy

class PrunedLinear:
    """
    Linear layer with honest zero-masking, proper autodiff backward scaling, 
    and permanent pruning support.
    """
    def __init__(self, in_features, out_features):
        limit = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * limit, op_name="weight")
        self.b = Tensor(np.zeros((1, out_features)), op_name="bias")
        
        # Binary mask: 1 = active, 0 = permanently pruned
        self.mask = np.ones((in_features, out_features), dtype=np.float32)
        self.saliency_accumulator = np.zeros((in_features, out_features), dtype=np.float32)
        self.saliency_steps = 0

    def parameters(self):
        return [self.W, self.b]

    def __call__(self, x):
        # 1. Honest forward masking
        masked_data = self.W.data * self.mask
        
        # 2. Create active_W Tensor with custom backward closure applying the chain rule: grad_W = grad_out * mask
        active_W = Tensor(masked_data, parents=[self.W], op_name="masked_weight")
        
        def _backward():
            if self.W.grad is not None and active_W.grad is not None:
                self.W.grad += active_W.grad * self.mask
            elif active_W.grad is not None:
                self.W.grad = active_W.grad * self.mask
                
        active_W._backward = _backward
        
        return (x @ active_W) + self.b


class PrunedMLP:
    """
    Multi-Layer Perceptron supporting exact global top-k permanent pruning.
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

    def accumulate_saliency(self):
        """Accumulates Taylor Saliency |W * grad| across mini-batches."""
        for layer in self.layers:
            if isinstance(layer, PrunedLinear) and layer.W.grad is not None:
                if not np.isfinite(layer.W.grad).all():
                    raise ValueError("NaN or Inf detected in gradients!")
                saliency = np.abs(layer.W.data * layer.W.grad)
                layer.saliency_accumulator += saliency
                layer.saliency_steps += 1

    def apply_global_permanent_pruning(self, target_sparsity):
        """
        Enforces exact global top-k permanent pruning with stable tie-breaking,
        zero-saliency ranking inclusion, and accurate floor budget enforcement.
        """
        # Issue 5: Target sparsity validation
        if not (0.0 <= target_sparsity < 1.0):
            raise ValueError(f"Target sparsity must be between 0.0 and 1.0 (got {target_sparsity}).")

        weight_refs = []
        active_scores_list = []
        
        for layer in self.layers:
            if isinstance(layer, PrunedLinear):
                if layer.saliency_steps > 0:
                    avg_saliency = layer.saliency_accumulator / layer.saliency_steps
                else:
                    avg_saliency = np.abs(layer.W.data)
                
                # Issue 3: Finite-value checks for computed saliency before ranking
                if not np.isfinite(avg_saliency).all():
                    raise ValueError("NaN or Inf detected in computed saliency scores!")
                
                weight_refs.append(layer)
                
                scores = np.where(layer.mask == 1.0, avg_saliency, np.inf)
                active_scores_list.append(scores)

        if not weight_refs:
            return

        flat_scores = np.concatenate([s.flatten() for s in active_scores_list])
        active_indices = np.where(flat_scores != np.inf)[0]
        total_active_count = len(active_indices)
        
        if total_active_count == 0:
            return

        # Issue 6: Budget rounding replaced with np.floor to prevent exceeding target budget
        total_original_weights = sum(l.mask.size for l in weight_refs)
        target_active_weights = int(np.floor(total_original_weights * (1.0 - target_sparsity)))
        num_to_prune = total_active_count - target_active_weights
        
        if num_to_prune <= 0:
            return

        active_scores = flat_scores[active_indices]
        sorted_order = np.argsort(active_scores, kind='stable')
        sorted_active_indices = active_indices[sorted_order]
        
        indices_to_prune = sorted_active_indices[:num_to_prune]

        mask_transitions = []
        flat_cursor = 0
        
        for layer in weight_refs:
            num_elements = layer.mask.size
            layer_flat_mask = layer.mask.flatten()
            
            layer_indices_to_prune = indices_to_prune[
                (indices_to_prune >= flat_cursor) & (indices_to_prune < flat_cursor + num_elements)
            ] - flat_cursor
            
            new_layer_mask = layer_flat_mask.copy()
            newly_pruned_flat = np.zeros_like(layer_flat_mask, dtype=bool)
            
            if len(layer_indices_to_prune) > 0:
                new_layer_mask[layer_indices_to_prune] = 0.0
                newly_pruned_flat[layer_indices_to_prune] = True
            
            new_mask = new_layer_mask.reshape(layer.mask.shape)
            newly_pruned = newly_pruned_flat.reshape(layer.mask.shape)
            
            layer.mask = new_mask
            
            # Issue 4: Immediately zero out underlying parameter values in storage
            layer.W.data *= layer.mask
            
            mask_transitions.append((layer, newly_pruned))
            
            layer.saliency_accumulator.fill(0.0)
            layer.saliency_steps = 0
            flat_cursor += num_elements

        return mask_transitions


class ReLU_Layer:
    def __call__(self, x):
        return x.relu()


class MaskedAdam:
    """
    Adam optimizer with explicit state clearing for permanently pruned weights.
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
        self.param_to_idx = {id(p): i for i, p in enumerate(self.params)}
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        
        for layer in self.model.layers:
            if isinstance(layer, PrunedLinear):
                p = layer.W
                mask = layer.mask
                
                if p.grad is not None:
                    if not np.isfinite(p.grad).all():
                        raise ValueError("NaN or Inf detected in weights or gradients!")
                    
                    p.grad *= mask
                    
                    idx = self.param_to_idx[id(p)]
                    grad = p.grad
                    
                    self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
                    self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)
                    
                    m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
                    
                    p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                    # Enforce parameter storage zeroing
                    p.data *= mask

                pb = layer.b
                if pb.grad is not None:
                    idx = self.param_to_idx[id(pb)]
                    grad = pb.grad
                    
                    self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
                    self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)
                    
                    m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
                    
                    pb.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset_optimizer_state_for_pruned(self, newly_pruned_mask, param_tensor):
        """Clears Adam momentum and velocity for newly pruned weights."""
        idx = self.param_to_idx[id(param_tensor)]
        self.m[idx][newly_pruned_mask] = 0.0
        self.v[idx][newly_pruned_mask] = 0.0


# Issue 9: Finite-difference gradient correctness test for custom backward function
def test_pruned_linear_gradient_correctness():
    print("Running Finite-Difference Gradient Correctness Test for PrunedLinear...")
    np.random.seed(42)
    layer = PrunedLinear(in_features=3, out_features=2)
    layer.mask[0, 0] = 0.0  # Mask out one connection
    layer.mask[1, 1] = 1.0  # Keep active
    layer.W.data *= layer.mask

    x_val = np.random.randn(2, 3).astype(np.float32)
    x = Tensor(x_val)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    # Analytic gradient
    analytic_grad = layer.W.grad.copy()

    # Numerical gradient via finite differences (epsilon perturbation)
    eps = 1e-5
    numerical_grad = np.zeros_like(layer.W.data)
    for i in range(layer.W.data.shape[0]):
        for j in range(layer.W.data.shape[1]):
            orig = layer.W.data[i, j]
            
            # Forward perturb
            layer.W.data[i, j] = orig + eps
            out_pos = (x_val @ (layer.W.data * layer.mask)).sum() + layer.b.data.sum()
            
            # Backward perturb
            layer.W.data[i, j] = orig - eps
            out_neg = (x_val @ (layer.W.data * layer.mask)).sum() + layer.b.data.sum()
            
            layer.W.data[i, j] = orig
            numerical_grad[i, j] = (out_pos - out_neg) / (2 * eps)

    # Verify masked weight gets exact zero analytic gradient
    assert analytic_grad[0, 0] == 0.0, "Masked weight gradient must be zero!"
    
    # Verify active weight matches finite difference approximation
    diff = np.abs(analytic_grad[1, 1] - numerical_grad[1, 1])
    assert diff < 1e-4, f"Gradient mismatch! Analytic: {analytic_grad[1, 1]}, Numerical: {numerical_grad[1, 1]}"
    print("✓ PrunedLinear finite-difference gradient correctness test passed successfully!")

if __name__ == "__main__":
    test_pruned_linear_gradient_correctness()
