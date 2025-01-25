import numpy as np
from sklearn.metrics import accuracy_score

#helper function to perform k-fold cross-validation. Returns list of nparrays. 
def custom_k_fold(X,y,k):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)  
    fold_size = n_samples // k
    folds = []

    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    
    return folds

# Helper function to compute mean validation error
def k_fold_cross_validation(X, y, model, k=5):
    folds = custom_k_fold(X, y, k)  
    errors = []

    for train_idx, test_idx in folds:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)  
        y_pred = model.predict(X_test)  
        error = 1 - accuracy_score(y_test, y_pred)  
        errors.append(error)
    
    return np.mean(errors)  # Return average validation error

