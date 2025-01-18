import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import _tree

def prune_tree(tree, X_val, y_val):
    """
    Prune a decision tree by iteratively removing nodes if it improves validation accuracy.
    """
    def prune_node(node):
        """
        Recursively prune the tree starting from the leaves.
        """
        if tree.tree_.children_left[node] != _tree.TREE_LEAF:
            # Recursively prune left and right children
            prune_node(tree.tree_.children_left[node])
            prune_node(tree.tree_.children_right[node])

            # Backup original predictions
            left_child, right_child = tree.tree_.children_left[node], tree.tree_.children_right[node]
            original_values = tree.tree_.value[node].copy()

            # Make the node a leaf by setting its children to -1
            tree.tree_.children_left[node] = _tree.TREE_LEAF
            tree.tree_.children_right[node] = _tree.TREE_LEAF

            # Compute validation accuracy after pruning
            y_val_pred = tree.predict(X_val)
            pruned_accuracy = accuracy_score(y_val, y_val_pred)

            # Restore the original node if pruning didn't improve accuracy
            if pruned_accuracy < prune_tree.best_val_accuracy:
                tree.tree_.children_left[node] = left_child
                tree.tree_.children_right[node] = right_child
                tree.tree_.value[node] = original_values
            else:
                prune_tree.best_val_accuracy = pruned_accuracy  # Update best accuracy

    # Fix: Use `tree.predict(X_val)` instead of `clf.predict(X_val)`
    prune_tree.best_val_accuracy = accuracy_score(y_val, tree.predict(X_val))  # Store best accuracy
    prune_node(0)  # Start pruning from the root
