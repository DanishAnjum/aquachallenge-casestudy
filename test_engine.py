import numpy as np
from engine import Tensor, softmax_cross_entropy

def test_broadcasting_and_backprop():
    print("Running Part 1 verification tests...")
    
    # 1. Test Broadcasting Backprop (N, D) + (D,)
    x_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_data = np.array([10.0, 20.0], dtype=np.float32)
    
    x = Tensor(x_data)
    b = Tensor(b_data)
    
    y = x + b
    loss = y.sum()
    loss.backward()
    
    # Expected gradients: d(loss)/dx = [[1, 1], [1, 1]], d(loss)/db = [2, 2]
    assert np.allclose(x.grad, np.ones_like(x_data))
    assert np.allclose(b.grad, np.array([2.0, 2.0]))
    print("✓ Broadcasting gradient test passed!")

if __name__ == "__main__":
    test_broadcasting_and_backprop()