import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('Generated Classification Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))
print("Class distribution:", np.bincount(y))

# Write the code for Q2 a) and b) below. Show your results.

# Part (a): Train-Test Split (70%-30%) and Evaluation
print("\nPART (A): Train-Test Split Evaluation")

# Convert to pandas for consistency with our DecisionTree implementation
X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
y_series = pd.Series(y)

# Split the data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.3, random_state=42, stratify=y_series
)

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")

# Train decision tree with both criteria
criteria = ["information_gain", "gini_index"]

for criterion in criteria:
    print(f"\n--- Decision Tree with {criterion.upper()} ---")
    
    # Create and train the decision tree
    dt = DecisionTree(criterion=criterion, max_depth=5)
    dt.fit(X_train, y_train)
    
    # Make predictions
    y_pred = dt.predict(X_test)
    
    # Calculate metrics
    test_accuracy = accuracy(y_pred, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Per-class precision and recall
    unique_classes = np.unique(y_test)
    print("\nPer-class Metrics:")
    print("Class | Precision | Recall")
    print("--- | --- | ---")
    
    for cls in unique_classes:
        prec = precision(y_pred, y_test, cls)
        rec = recall(y_pred, y_test, cls)
        print(f"  {cls}   |   {prec:.4f}   | {rec:.4f}")
    
    # Display the tree structure
    print(f"\nDecision Tree Structure ({criterion}):")
    dt.plot()
    print()

# Part (b): 5-Fold Cross-Validation with Nested CV for Optimal Depth
print("\nPART (B): 5-Fold Cross-Validation with Nested CV for Optimal Depth")

def cross_validate_depth(X, y, criterion, depths, k_folds=5):
    """
    Perform nested cross-validation to find optimal depth
    """
    kf_outer = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    outer_scores = []
    optimal_depths = []
    
    for fold, (train_idx, test_idx) in enumerate(kf_outer.split(X)):
        print(f"\nOuter Fold {fold + 1}:")
        
        # Split data for outer fold
        X_train_outer = X.iloc[train_idx]
        y_train_outer = y.iloc[train_idx]
        X_test_outer = X.iloc[test_idx]
        y_test_outer = y.iloc[test_idx]
        
        # Inner cross-validation for hyperparameter tuning
        kf_inner = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        depth_scores = {}
        for depth in depths:
            inner_scores = []
            
            for train_inner_idx, val_idx in kf_inner.split(X_train_outer):
                X_train_inner = X_train_outer.iloc[train_inner_idx]
                y_train_inner = y_train_outer.iloc[train_inner_idx]
                X_val = X_train_outer.iloc[val_idx]
                y_val = y_train_outer.iloc[val_idx]
                
                # Train and evaluate
                dt = DecisionTree(criterion=criterion, max_depth=depth)
                dt.fit(X_train_inner, y_train_inner)
                y_pred_val = dt.predict(X_val)
                
                inner_scores.append(accuracy(y_pred_val, y_val))
            
            depth_scores[depth] = np.mean(inner_scores)
        
        # Find optimal depth for this outer fold
        optimal_depth = max(depth_scores, key=depth_scores.get)
        optimal_depths.append(optimal_depth)
        
        print(f"  Depth scores: {depth_scores}")
        print(f"  Optimal depth: {optimal_depth}")
        
        # Train final model on full outer training set with optimal depth
        dt_final = DecisionTree(criterion=criterion, max_depth=optimal_depth)
        dt_final.fit(X_train_outer, y_train_outer)
        y_pred_outer = dt_final.predict(X_test_outer)
        
        outer_score = accuracy(y_pred_outer, y_test_outer)
        outer_scores.append(outer_score)
        print(f"  Outer fold accuracy: {outer_score:.4f}")
    
    return outer_scores, optimal_depths

# Test different depths
depths_to_test = [1, 2, 3, 4, 5, 6, 7, 8]

for criterion in criteria:
    print(f"\nNested CV Results for {criterion.upper()}")
    
    outer_scores, optimal_depths = cross_validate_depth(
        X_df, y_series, criterion, depths_to_test, k_folds=5
    )
    
    print(f"\nSummary for {criterion}:")
    print(f"Cross-validation scores: {[f'{score:.4f}' for score in outer_scores]}")
    print(f"Mean CV accuracy: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    print(f"Optimal depths per fold: {optimal_depths}")
    print(f"Most frequent optimal depth: {max(set(optimal_depths), key=optimal_depths.count)}")

# Final Analysis: Train with best depth on full dataset
print(f"\nFINAL ANALYSIS: Training with Best Depth on Full Dataset")

# Simple cross-validation to find overall best depth
def simple_cv_for_depth(X, y, criterion, depths, k_folds=5):
    """Simple cross-validation to find best depth"""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    depth_scores = {}
    for depth in depths:
        scores = []
        for train_idx, val_idx in kf.split(X):
            X_train_cv = X.iloc[train_idx]
            y_train_cv = y.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_val_cv = y.iloc[val_idx]
            
            dt = DecisionTree(criterion=criterion, max_depth=depth)
            dt.fit(X_train_cv, y_train_cv)
            y_pred_cv = dt.predict(X_val_cv)
            scores.append(accuracy(y_pred_cv, y_val_cv))
        
        depth_scores[depth] = np.mean(scores)
    
    return depth_scores

for criterion in criteria:
    print(f"\nFinding best depth for {criterion}:")
    depth_scores = simple_cv_for_depth(X_df, y_series, criterion, depths_to_test)
    
    best_depth = max(depth_scores, key=depth_scores.get)
    best_score = depth_scores[best_depth]
    
    print(f"Depth scores: {depth_scores}")
    print(f"Best depth: {best_depth} (CV accuracy: {best_score:.4f})")
    
    # Train final model with best depth
    print(f"\nFinal model with depth {best_depth}:")
    dt_final = DecisionTree(criterion=criterion, max_depth=best_depth)
    dt_final.fit(X_df, y_series)
    dt_final.plot()

print(f"\nEXPERIMENT COMPLETED")

