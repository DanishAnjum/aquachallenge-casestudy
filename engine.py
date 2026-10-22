import numpy as np

class Tensor:
    """
    A wrap around NumPy arrays that tracks computation history (parents) 
    and computes gradients via reverse-mode automatic differentiation.
    """
    def __init__(self, data, parents=None, op_name="", requires_grad=True):
        # Ensure data is a NumPy float array
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        
        # Graph tracking
        self.parents = parents if parents is not None else []
        self.op_name = op_name
        self.requires_grad = requires_grad
        
        # Internal function stored during forward pass to compute gradients during backward()
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self.op_name})"

    def backward(self, grad_output=None):
        """
        Executes reverse-mode autodiff through topological sorting.
        """
        if not self.requires_grad:
            return

        if grad_output is None:
            if self.data.shape == ():
                grad_output = np.array(1.0, dtype=np.float32)
            else:
                if self.data.size == 1:
                    grad_output = np.ones_like(self.data)
                else:
                    raise RuntimeError("Grad output can only be implicitly created for scalar outputs")

        # Accumulate gradients
        self.grad += grad_output

        # Build topological sort of the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Traverse backwards in reverse topological order
        for v in reversed(topo):
            v._backward()

    # ==========================================
    # Helper for handling broadcasting in backward
    # ==========================================
    @staticmethod
    def _unbroadcast(grad, target_shape):
        """
        Sums out broadcasted dimensions so grad matches target_shape.
        """
        if grad.shape == target_shape:
            return grad
        
        grad_ndim = grad.ndim
        target_ndim = len(target_shape)
        
        # 1. Sum out leading extra dimensions
        for _ in range(grad_ndim - target_ndim):
            grad = grad.sum(axis=0)
            
        # 2. Sum out dimensions where target shape had size 1
        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
                
        # Ensure exact shape match
        return np.reshape(grad, target_shape)

    # ==========================================
    # Differentiable Arithmetic Operations
    # ==========================================
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, parents=[self, other], op_name="add")

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data - other.data, parents=[self, other], op_name="sub")

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(-out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return Tensor(other, requires_grad=False) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, parents=[self, other], op_name="mul")

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data / other.data, parents=[self, other], op_name="div")

        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad / other.data, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(out.grad * (-self.data / (other.data ** 2)), other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        out = Tensor(self.data @ other.data, parents=[self, other], op_name="matmul")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # ==========================================
    # Reductions
    # ==========================================
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), parents=[self], op_name="sum")

        def _backward():
            if self.requires_grad:
                # Reconstruct broadcast shape for sum backward
                grad = out.grad
                if not keepdims and axis is not None:
                    # Expand dimensions back for proper broadcasting
                    axes = [axis] if isinstance(axis, int) else axis
                    shape = list(self.data.shape)
                    for ax in sorted(axes):
                        shape[ax] = 1
                    grad = np.reshape(grad, shape)
                self.grad += np.broadcast_to(grad, self.data.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        num_elements = self.data.size if axis is None else (
            np.prod([self.data.shape[a] for a in axis]) if isinstance(axis, tuple) else self.data.shape[axis]
        )
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / num_elements)

    # ==========================================
    # Non-Linearities & Activations
    # ==========================================
    def relu(self):
        out = Tensor(np.maximum(0, self.data), parents=[self], op_name="relu")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, parents=[self], op_name="tanh")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * (1 - t ** 2)
        out._backward = _backward
        return out

# ==========================================
# Stable Softmax + Cross-Entropy Loss
# ==========================================
def softmax_cross_entropy(logits, targets):
    """
    Numerically stable softmax cross-entropy loss.
    logits: Tensor of shape (N, C)
    targets: NumPy array or Tensor of shape (N,) containing class indices
    """
    # Numerical stability shift: subtract max along classes
    shift_logits = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exps = np.exp(shift_logits)
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    
    N = logits.data.shape[0]
    target_probs = probs[np.arange(N), targets]
    loss_val = -np.mean(np.log(target_probs + 1e-12))
    
    out = Tensor(loss_val, parents=[logits], op_name="softmax_ce")

    def _backward():
        if logits.requires_grad:
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1.0
            grad = grad / N
            logits.grad += grad * out.grad
            
    out._backward = _backward
    return out