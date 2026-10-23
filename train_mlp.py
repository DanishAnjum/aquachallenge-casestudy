import numpy as np
import matplotlib.pyplot as plt
from engine import Tensor, softmax_cross_entropy
from nn_model import MLP, Adam

def generate_synthetic_data(num_samples=600, input_dim=16, num_classes=3):
    """Generates synthetic classification data."""
    np.random.seed(42)
    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = np.dot(X[:, :3], np.array([[1.5], [-2.0], [0.8]])) + np.random.randn(num_samples, 1) * 0.5
    y = np.squeeze((y > 0).astype(np.int64) + (y > 1).astype(np.int64)) % num_classes
    return X, y

def train_network():
    print("Training Part 2 MLP & generating learning curves...")
    
    X, y = generate_synthetic_data()
    split = 500
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    model = MLP([16, 32, 3])
    optimizer = Adam(model.parameters(), lr=0.01)
    
    epochs = 20
    batch_size = 32
    
    history_loss = []
    history_acc = []
    
    for epoch in range(epochs):
        permutation = np.random.permutation(len(X_train))
        X_train_shuff = X_train[permutation]
        y_train_shuff = y_train[permutation]
        
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, len(X_train), batch_size):
            X_batch = Tensor(X_train_shuff[i:i+batch_size])
            y_batch = y_train_shuff[i:i+batch_size]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = softmax_cross_entropy(logits, y_batch)
            if not np.isfinite(loss.data).all():
                raise FloatingPointError("Non-finite loss detected")
            loss.backward()
            for p in model.parameters():
                if not np.isfinite(p.grad).all():
                    raise FloatingPointError("Non-finite gradient detected")
            optimizer.step()
            for p in model.parameters():
                if not np.isfinite(p.data).all():
                    raise FloatingPointError("Non-finite parameter detected")
            
            epoch_loss += loss.data.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        
        # Validation evaluation
        val_logits = model(Tensor(X_val))
        val_probs = np.exp(val_logits.data - np.max(val_logits.data, axis=1, keepdims=True))
        val_probs /= np.sum(val_probs, axis=1, keepdims=True)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_preds == y_val) * 100
        
        history_loss.append(avg_loss)
        history_acc.append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.1f}%")
        
    # Plot and save learning curve
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history_loss, marker='o', color='b')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history_acc, marker='s', color='g')
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    print("✓ Training complete! Learning curve saved as 'learning_curve.png'.")

if __name__ == "__main__":
    train_network()
