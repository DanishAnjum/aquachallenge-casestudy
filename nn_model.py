import numpy as np
from engine import Tensor, softmax_cross_entropy

class Linear:
    """
    Fully connected (Linear) layer: y = x @ W + b
    """
    def __init__(self, in_features, out_features):
        # He (Kaiming) Initialization for ReLU stability
        # Std dev = sqrt(2 / in_features)
        limit = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * limit, op_name="weight")
        self.b = Tensor(np.zeros((1, out_features)), op_name="bias")
        
    def parameters(self):
        return [self.W, self.b]

    def __call__(self, x):
        return (x @ self.W) + self.b


class ReLU:
    """
    ReLU activation layer wrapper.
    """
    def __call__(self, x):
        return x.relu()


class MLP:
    """
    A small Multi-Layer Perceptron (MLP) classifier.
    """
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2: # Add ReLU for hidden layers
                self.layers.append(ReLU())
                
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


class Adam:
    """
    Adam Optimizer implemented from scratch in NumPy.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        # Initialize historical moment vectors (m and v) as zeros for each parameter
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0 # timestep counter

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = np.zeros_like(p.data)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)