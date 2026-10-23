import numpy as np
from engine import Tensor, softmax_cross_entropy
from self_prune import PrunedMLP, MaskedAdam

def run_pruning_experiment():
    print("=" * 60)
    print("Running Enterprise-Grade Self-Pruning vs. Fair Dense Baseline Experiment")
    print("=" * 60)
    
    # Reproducible dataset & shared permutation seed
    np.random.seed(42)
    X = np.random.randn(600, 16).astype(np.float32)
    y = np.dot(X[:, :3], np.array([[1.5], [-2.0], [0.8]])) + np.random.randn(600, 1) * 0.5
    y = np.squeeze((y > 0).astype(np.int64) + (y > 1).astype(np.int64)) % 3
    
    X_train, y_train = X[:500], y[:500]
    X_val, y_val = X[500:], y[500:]
    
    epochs = 25
    batch_size = 32
    
    # Generate identical epoch permutations for both models to ensure a completely fair baseline comparison (Issue 1)
    np.random.seed(100)
    epoch_permutations = [np.random.permutation(len(X_train)) for _ in range(epochs)]

    # ==========================================
    # 1. TRAIN DENSE BASELINE (0% Sparsity)
    # ==========================================
    print("\n[Baseline] Training Dense MLP (0% Sparsity with identical batch schedule)...")
    np.random.seed(42)
    dense_model = PrunedMLP([16, 32, 3])
    dense_optimizer = MaskedAdam(dense_model, lr=0.01)
    
    for epoch in range(epochs):
        perm = epoch_permutations[epoch]
        X_tr_shuff = X_train[perm]
        y_tr_shuff = y_train[perm]
        
        for i in range(0, len(X_train), batch_size):
            X_b = Tensor(X_tr_shuff[i:i+batch_size])
            y_b = y_tr_shuff[i:i+batch_size]
            dense_optimizer.zero_grad()
            loss = softmax_cross_entropy(dense_model(X_b), y_b)
            loss.backward()
            dense_optimizer.step()
            
    val_logits_dense = dense_model(Tensor(X_val))
    val_probs_d = np.exp(val_logits_dense.data - np.max(val_logits_dense.data, axis=1, keepdims=True))
    val_probs_d /= np.sum(val_probs_d, axis=1, keepdims=True)
    dense_val_acc = np.mean(np.argmax(val_probs_d, axis=1) == y_val) * 100
    print(f"-> Dense Baseline Validation Accuracy: {dense_val_acc:.2f}%\n")

    # ==========================================
    # 2. TRAIN SELF-PRUNING MODEL
    # ==========================================
    print("[Pruning] Training Self-Pruning MLP...")
    np.random.seed(42)
    model = PrunedMLP([16, 32, 3])
    optimizer = MaskedAdam(model, lr=0.01)
    
    target_final_sparsity = 0.85
    warmup_epochs = 5
    
    for epoch in range(epochs):
        perm = epoch_permutations[epoch]
        X_train_shuff = X_train[perm]
        y_train_shuff = y_train[perm]
        
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, len(X_train), batch_size):
            X_batch = Tensor(X_train_shuff[i:i+batch_size])
            y_batch = y_train_shuff[i:i+batch_size]
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = softmax_cross_entropy(logits, y_batch)
            
            if not np.isfinite(loss.data):
                raise ValueError(f"NaN or Inf loss detected at epoch {epoch+1}!")
                
            loss.backward()
            
            if epoch >= warmup_epochs:
                model.accumulate_saliency()
                
            optimizer.step()
            
            epoch_loss += loss.data.item()
            batches += 1
            
        avg_loss = epoch_loss / batches

        if epoch >= warmup_epochs:
            pruning_progress = (epoch - warmup_epochs) / (epochs - warmup_epochs - 1)
            current_target_sparsity = target_final_sparsity * (1.0 - (1.0 - pruning_progress) ** 3)
            
            if epoch % 2 == 0:
                transitions = model.apply_global_permanent_pruning(current_target_sparsity)
                if transitions:
                    for layer, newly_pruned in transitions:
                        optimizer.reset_optimizer_state_for_pruned(newly_pruned, layer.W)
        else:
            current_target_sparsity = 0.0

        val_logits = model(Tensor(X_val))
        val_probs = np.exp(val_logits.data - np.max(val_logits.data, axis=1, keepdims=True))
        val_probs /= np.sum(val_probs, axis=1, keepdims=True)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = np.mean(val_preds == y_val) * 100

        total_weights = 0
        zero_weights = 0
        for layer in model.layers:
            if hasattr(layer, 'mask'):
                total_weights += layer.mask.size
                zero_weights += np.sum(layer.mask == 0)
        achieved_sparsity = zero_weights / total_weights if total_weights > 0 else 0.0

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:5.1f}% | Target Sparsity: {current_target_sparsity*100:4.1f}% | Achieved Sparsity: {achieved_sparsity*100:4.1f}%")

    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY & BASELINE COMPARISON:")
    print(f"- Dense Baseline Validation Accuracy: {dense_val_acc:.2f}%")
    print(f"- Pruned Model Validation Accuracy:  {val_acc:.2f}%")
    print(f"- Accuracy Difference:             {val_acc - dense_val_acc:+.2f}%")
    print(f"- Target Sparsity:                 {target_final_sparsity*100:.2f}%")
    print(f"- Achieved Sparsity:               {achieved_sparsity*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_pruning_experiment()
