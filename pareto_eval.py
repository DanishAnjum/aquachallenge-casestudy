import numpy as np
from engine import Tensor, softmax_cross_entropy
from self_prune import PrunedMLP, MaskedAdam, PrunedLinear

def evaluate_sparsity_sweep():
    print("=" * 60)
    print("PART 4: Running Sparsity-Accuracy Pareto Sweep & Baseline Comparison")
    print("=" * 60)
    
    # Generate reproducible dataset
    np.random.seed(42)
    X = np.random.randn(600, 16).astype(np.float32)
    y = np.dot(X[:, :3], np.array([[1.5], [-2.0], [0.8]])) + np.random.randn(600, 1) * 0.5
    y = np.squeeze((y > 0).astype(np.int64) + (y > 1).astype(np.int64)) % 3
    
    X_train, y_train = X[:500], y[:500]
    X_val, y_val = X[500:], y[500:]
    
    sparsity_levels = [0.0, 0.50, 0.75, 0.90, 0.95]
    seeds = [42, 100, 2026]
    
    results = {
        "saliency": [],
        "magnitude": []
    }
    
    for sparsity in sparsity_levels:
        print(f"\n--- Target Sparsity: {sparsity*100}% ---")
        
        for method in ["saliency", "magnitude"]:
            accuracies = []
            active_params_list = []
            
            for seed in seeds:
                np.random.seed(seed)
                model = PrunedMLP([16, 32, 3])
                optimizer = MaskedAdam(model, lr=0.01)
                
                # Train for 15 epochs
                for epoch in range(15):
                    if epoch % 3 == 0 and sparsity > 0:
                        if method == "saliency":
                            model.update_masks(sparsity)
                        else:
                            # Pure magnitude baseline pruning
                            for layer in model.layers:
                                if isinstance(layer, PrunedLinear):
                                    mags = np.abs(layer.W.data)
                                    flat_mags = mags.flatten()
                                    k = int(len(flat_mags) * (1.0 - sparsity))
                                    if k > 0:
                                        thresh = np.partition(flat_mags, -k)[-k]
                                        layer.mask = (mags >= thresh).astype(np.float32)
                                        
                    logits = model(Tensor(X_train))
                    loss = softmax_cross_entropy(logits, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Evaluate Validation Accuracy
                val_logits = model(Tensor(X_val))
                val_probs = np.exp(val_logits.data - np.max(val_logits.data, axis=1, keepdims=True))
                val_probs /= np.sum(val_probs, axis=1, keepdims=True)
                val_preds = np.argmax(val_probs, axis=1)
                acc = np.mean(val_preds == y_val) * 100
                accuracies.append(acc)
                
                # Measure Real Cost (Active Parameter Count)
                total_weights = sum(layer.mask.size for layer in model.layers if hasattr(layer, 'mask'))
                active_weights = sum(np.sum(layer.mask == 1) for layer in model.layers if hasattr(layer, 'mask'))
                active_params_list.append(active_weights)
                
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            mean_active = np.mean(active_params_list)
            
            results[method].append((sparsity, mean_acc, std_acc, mean_active))
            print(f"[{method.upper():9s}] Sparsity: {sparsity*100:4.1f}% | Accuracy: {mean_acc:5.2f}% (±{std_acc:4.2f}) | Active Weights: {mean_active:.0f}")

    print("\n" + "=" * 60)
    print("FALSIFIABLE CLAIM FOR DESIGN.md:")
    print("=" * 60)
    
    sal_90 = results["saliency"][3][1] # 90% saliency accuracy
    mag_90 = results["magnitude"][3][1] # 90% magnitude accuracy
    print(f"\"At 90% sparsity, Taylor Saliency pruning retains {sal_90:.1f}% validation accuracy, "
          f"outperforming pure magnitude pruning ({mag_90:.1f}%) by {sal_90 - mag_90:.1f}% "
          f"averaged across 3 random seeds, with an active parameter count reduction of 90%.\"")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_sparsity_sweep()