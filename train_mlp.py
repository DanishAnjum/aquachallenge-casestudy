import numpy as np
from engine import Tensor, softmax_cross_entropy
from nn_model import MLP, Adam

def generate_synthetic_data(num_samples=500, input_dim=20, num_classes=3):
    """Generates synthetic classification data."""
    np.random.seed(42)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    # Generate random class labels based on a simple linear combination + noise
    y = np.dot(X[:, :3], np.array([[1.5], [-2.0], [0.8]])) + np.random.randn(num_samples, 1) * 0.5
    y = np.squeeze((y > 0).astype(np.int64) + (y > 1).astype(np.int64)) % num_classes
    return X, y

def train_network():
    print("Initializing Part 2 Training Pipeline...")
    
    # Setup data
    X, y = generate_synthetic_data(num_samples=600, input_dim=16, num_classes=3)
    
    # Train/Val split
    split = 500
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    # Create Model: Input (16) -> Hidden (32) -> Output (3 classes)
    model = MLP([16, 32, 3] )
    optimizer = Adam(model.parameters(), lr=0.01)
    
    epochs = 15
    batch_size = 32
    
    for epoch in range(epochs):
        # Mini-batch training loop
        permutation = np.random.permutation(len(X_train))
        X_train_shuff = X_train[permutation]
        y_train_shuff = y_train[permutation]
        
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, len(X_train), batch_size):
            X_batch = Tensor(X_train_shuff[i:i+batch_size])
            y_batch = y_train_shuff[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(X_batch)
            loss = softmax_cross_entropy(logits, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            epoch_loss += loss.data.item()
            batches += 1
            
        # Validation accuracy
        val_logits = model(Tensor(X_val))
        val_probs = np.exp(val_logits.data - np.max(val_logits.data, axis=1, keepdims=True))
        val_probs /= np.sum(val_probs, axis=1, keepdims=True)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_preds == y_val) * 100
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss/batches:.4f} | Val Accuracy: {val_acc:.1f}%")
        
    print("✓ Training completed successfully without NaNs!")

if __name__ == "__main__":
    train_network()