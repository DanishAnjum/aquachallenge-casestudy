import numpy as np
from engine import Tensor, softmax_cross_entropy
from self_prune import PrunedMLP, MaskedAdam

def run_pruning_experiment():
    print("Starting Self-Pruning Training Run...")
    
    np.random.seed(42)
    X = np.random.randn(500, 16).astype(np.float32)
    y = np.dot(X[:, :3], np.array([[1.5], [-2.0], [0.8]])) + np.random.randn(500, 1) * 0.5
    y = np.squeeze((y > 0).astype(np.int64) + (y > 1).astype(np.int64)) % 3
    
    model = PrunedMLP([16, 32, 3])
    optimizer = MaskedAdam(model, lr=0.01)
    
    epochs = 20
    target_final_sparsity = 0.80 # Target 80% sparsity
    
    for epoch in range(epochs):
        # Cubic sparsity schedule ramp
        if epoch < 15:
            current_sparsity = target_final_sparsity * (1.0 - (1.0 - epoch / 15.0) ** 3)
        else:
            current_sparsity = target_final_sparsity
            
        # Update pruning masks every few epochs
        if epoch % 3 == 0:
            model.update_masks(current_sparsity)
            
        # Mini-batch training step
        logits = model(Tensor(X))
        loss = softmax_cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate active sparsity
        total_weights = 0
        zero_weights = 0
        for layer in model.layers:
            if hasattr(layer, 'mask'):
                total_weights += layer.mask.size
                zero_weights += np.sum(layer.mask == 0)
        achieved_sparsity = zero_weights / total_weights
        
        print(f"Epoch {epoch+1:02d} | Loss: {loss.data.item():.4f} | Target Sparsity: {current_sparsity*100:.1f}% | Actual Sparsity: {achieved_sparsity*100:.1f}%")

    print("✓ Self-pruning run completed successfully!")

if __name__ == "__main__":
    run_pruning_experiment()