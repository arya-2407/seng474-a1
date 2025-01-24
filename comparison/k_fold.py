#helper function to perform k-fold cross-validation. Returns list of nparrays. 
def custom_k_fold(X,y,k):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  # Shuffle indices to randomize splits
    fold_size = n_samples // k
    folds = []

    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    
    return folds
