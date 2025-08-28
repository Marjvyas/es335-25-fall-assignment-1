"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal
import pandas as pd
import numpy as np


@dataclass
class Node:
    """
    A node in the decision tree.
    """
    feature: str = None
    threshold: float = None
    value: float = None
    samples: int = 0
    left: 'Node' = None
    right: 'Node' = None
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    """
    A decision tree classifier/regressor using Information Gain (entropy) for splitting.
    """
    
    def __init__(self, criterion = "information_gain", max_depth: int = 5):
        """
        Initialize the decision tree.
        
        Parameters:
        -----------
        criterion : str
            The function to measure the quality of a split. Only "information_gain" is implemented.
            If "gini_index" is passed, the tree will not be built (since it's optional).
        max_depth : int
            The maximum depth of the tree.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_classification = None
        self.X_train = None
        self.y_train = None
        
    def entropy(self, y: pd.Series) -> float:
        """
        Calculate the entropy for a given set of class labels.
        
        Parameters:
        -----------
        y : pd.Series
            The target values
            
        Returns:
        --------
        float
            The entropy value
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def gini_index(self, y: pd.Series) -> float:
        """
        Calculate the Gini index for a given set of class labels.
        
        Parameters:
        -----------
        y : pd.Series
            The target values
            
        Returns:
        --------
        float
            The Gini index value
        """
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def information_gain(self, parent: pd.Series, left_child: pd.Series, right_child: pd.Series) -> float:
        """
        Calculate the information gain from splitting the parent into left and right children.
        
        Parameters:
        -----------
        parent : pd.Series
            The parent node's target values
        left_child : pd.Series
            The left child's target values
        right_child : pd.Series
            The right child's target values
            
        Returns:
        --------
        float
            The information gain value
        """
        if len(parent) == 0:
            return 0.0
        
        if self.is_classification:
            parent_entropy = self.entropy(parent)
            
            n = len(parent)
            n_left = len(left_child)
            n_right = len(right_child)
            
            if n_left == 0 or n_right == 0:
                return 0.0
            
            weighted_entropy = (n_left / n) * self.entropy(left_child) + (n_right / n) * self.entropy(right_child)
            return parent_entropy - weighted_entropy
        else:
            # For regression, use variance reduction (MSE reduction)
            parent_var = np.var(parent) if len(parent) > 0 else 0.0
            
            n = len(parent)
            n_left = len(left_child)
            n_right = len(right_child)
            
            if n_left == 0 or n_right == 0:
                return 0.0
            
            left_var = np.var(left_child) if len(left_child) > 0 else 0.0
            right_var = np.var(right_child) if len(right_child) > 0 else 0.0
            
            weighted_var = (n_left / n) * left_var + (n_right / n) * right_var
            return parent_var - weighted_var
    
    def gini_gain(self, parent: pd.Series, left_child: pd.Series, right_child: pd.Series) -> float:
        """
        Calculate the Gini gain from splitting the parent into left and right children.
        
        Parameters:
        -----------
        parent : pd.Series
            The parent node's target values
        left_child : pd.Series
            The left child's target values
        right_child : pd.Series
            The right child's target values
            
        Returns:
        --------
        float
            The Gini gain value
        """
        if len(parent) == 0:
            return 0.0
        
        if self.is_classification:
            parent_gini = self.gini_index(parent)
            
            n = len(parent)
            n_left = len(left_child)
            n_right = len(right_child)
            
            if n_left == 0 or n_right == 0:
                return 0.0
            
            weighted_gini = (n_left / n) * self.gini_index(left_child) + (n_right / n) * self.gini_index(right_child)
            return parent_gini - weighted_gini
        else:
            # For regression, use variance reduction (same as information gain)
            parent_var = np.var(parent) if len(parent) > 0 else 0.0
            
            n = len(parent)
            n_left = len(left_child)
            n_right = len(right_child)
            
            if n_left == 0 or n_right == 0:
                return 0.0
            
            left_var = np.var(left_child) if len(left_child) > 0 else 0.0
            right_var = np.var(right_child) if len(right_child) > 0 else 0.0
            
            weighted_var = (n_left / n) * left_var + (n_right / n) * right_var
            return parent_var - weighted_var
    
    def find_best_split(self, X: pd.DataFrame, y: pd.Series):
        """
        Find the best feature and threshold to split on using the specified criterion.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target values
            
        Returns:
        --------
        tuple
            (best_feature, best_threshold, best_left_indices, best_right_indices, best_gain)
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        for feature in X.columns:
            feature_values = X[feature]
            
            # Handle discrete features
            if feature_values.dtype == 'object' or feature_values.dtype.name == 'category':
                unique_values = feature_values.unique()
                for value in unique_values:
                    left_indices = feature_values == value
                    right_indices = ~left_indices
                    
                    if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                        continue
                    
                    left_y = y[left_indices]
                    right_y = y[right_indices]
                    
                    # Use the appropriate criterion
                    if self.criterion == "information_gain":
                        gain = self.information_gain(y, left_y, right_y)
                    else:  # gini_index
                        gain = self.gini_gain(y, left_y, right_y)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = value
                        best_left_indices = left_indices
                        best_right_indices = right_indices
            
            # Handle continuous/real features
            else:
                sorted_values = np.sort(feature_values.unique())
                # Use midpoints between consecutive unique values as potential thresholds
                for i in range(len(sorted_values) - 1):
                    threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
                    
                    left_indices = feature_values <= threshold
                    right_indices = feature_values > threshold
                    
                    if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                        continue
                    
                    left_y = y[left_indices]
                    right_y = y[right_indices]
                    
                    # Use the appropriate criterion
                    if self.criterion == "information_gain":
                        gain = self.information_gain(y, left_y, right_y)
                    else:  # gini_index
                        gain = self.gini_gain(y, left_y, right_y)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold
                        best_left_indices = left_indices
                        best_right_indices = right_indices
        
        return best_feature, best_threshold, best_left_indices, best_right_indices, best_gain
    
    def build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target values
        depth : int
            Current depth of the tree
            
        Returns:
        --------
        Node
            The root node of the built tree
        """
        node = Node()
        node.samples = len(y)
        
        # Determine leaf node value
        if self.is_classification:
            # Most common class for classification
            node.value = y.mode().iloc[0] if len(y) > 0 else 0
        else:
            # Mean value for regression
            node.value = np.mean(y) if len(y) > 0 else 0.0
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            len(y.unique()) == 1 or 
            len(y) < 2):
            return node
        
        # Find the best split
        best_feature, best_threshold, left_indices, right_indices, best_gain = self.find_best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            return node
        
        # Create the split
        node.feature = best_feature
        node.threshold = best_threshold
        
        # Create child nodes
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        
        node.left = self.build_tree(X_left, y_left, depth + 1)
        node.right = self.build_tree(X_right, y_right, depth + 1)
        
        return node
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Build a decision tree from the training set (X, y).
        
        Parameters:
        -----------
        X : pd.DataFrame
            The input samples.
        y : pd.Series
            The target values.
        """
            
        # Reset indices to ensure alignment
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Store training data for plotting
        self.X_train = X
        self.y_train = y
        
        # Determine if this is a classification or regression task
        if (y.dtype == 'object' or 
            y.dtype.name == 'category' or 
            (y.dtype in ['int64', 'float64'] and len(y.unique()) <= 10)):
            self.is_classification = True
        else:
            self.is_classification = False
        
        # Build the tree
        self.root = self.build_tree(X, y)
        
    def _predict_single(self, node: Node, sample: pd.Series):
        """
        Predict a single sample using the decision tree.
        
        Parameters:
        -----------
        node : Node
            Current node in the tree
        sample : pd.Series
            Single sample to predict
            
        Returns:
        --------
        The predicted value
        """
        if node.is_leaf():
            return node.value
        
        feature_value = sample[node.feature]
        
        # Handle discrete features (exact match)
        if isinstance(node.threshold, str) or self.X_train[node.feature].dtype == 'object':
            if feature_value == node.threshold:
                return self._predict_single(node.left, sample)
            else:
                return self._predict_single(node.right, sample)
        
        # Handle continuous features (threshold comparison)
        else:
            if feature_value <= node.threshold:
                return self._predict_single(node.left, sample)
            else:
                return self._predict_single(node.right, sample)
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict class or regression value for samples in X.
        
        Parameters:
        -----------
        X : pd.DataFrame
            The input samples.
            
        Returns:
        --------
        pd.Series
            The predicted values.
        """
        # Return empty predictions if implementation was skipped
        if self.root is None:
            
            print("Decision tree has not been fitted yet.")
            # Return default predictions (zeros)
            return pd.Series([0] * len(X), index=X.index)
        
        predictions = []
        for _, sample in X.iterrows():
            prediction = self._predict_single(self.root, sample)
            predictions.append(prediction)
        
        return pd.Series(predictions, index=X.index)
        
    def _plot_tree_text(self, node: Node, depth: int = 0, prefix: str = "", is_left: bool = True) -> None:
        """
        Recursively print the tree structure in a nice format.
        
        Parameters:
        -----------
        node : Node
            Current node to plot
        depth : int
            Current depth
        prefix : str
            Prefix for the current line
        is_left : bool
            Whether this is a left child
        """
        if node is None:
            return
        
        if node.is_leaf():
            if self.is_classification:
                print(f"{prefix}{'├─' if is_left else '└─'} class: {node.value} (samples: {node.samples})")
            else:
                print(f"{prefix}{'├─' if is_left else '└─'} value: {node.value:.3f} (samples: {node.samples})")
        else:
            # Print the condition
            if isinstance(node.threshold, str):
                condition = f"{node.feature} == '{node.threshold}'"
            else:
                condition = f"{node.feature} ≤ {node.threshold:.3f}"
            
            print(f"{prefix}{'├─' if is_left else '└─'} {condition} (samples: {node.samples})")
            
            # Prepare prefix for children
            child_prefix = prefix + ("│   " if is_left else "    ")
            
            # Plot children
            if node.left:
                self._plot_tree_text(node.left, depth + 1, child_prefix, True)
            if node.right:
                self._plot_tree_text(node.right, depth + 1, child_prefix, False)
        
    def plot(self) -> None:
        """
        Function to plot the tree
        
        Output Example:
        ├── X₀ ≤ 6.0
        │   ├── X₁ ≤ 4.0
        │   │   ├── class_A
        │   │   └── class_B
        │   └── X₁ ≤ 2.0
        │       ├── class_B
        │       └── class_A
        └── class_B
        """
            
        if self.root is None:
            print("Tree has not been fitted yet.")
            return
        
        # First create graphical version with auto-generated filename
        try:
            import graphviz
            import os
            import time
            
            # Generate filename based on data types and criterion
            input_type = "discrete" if any(self.X_train.dtypes == 'object') else "real"
            output_type = "discrete" if self.is_classification else "real"
            
            # Use timestamp to ensure unique filenames for each run
            timestamp = str(int(time.time() * 1000))[-6:]
            filename = f"tree_{input_type}_input_{output_type}_output_{self.criterion}_{timestamp}"
            png_filename = f"{filename}.png"
            
            # Create graphviz representation
            dot = graphviz.Digraph(comment='Decision Tree')
            dot.attr(rankdir='TB')
            dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        
            def add_nodes(node: Node, node_id: str = "0"):
                if node is None:
                    return
                
                if node.is_leaf():
                    if self.is_classification:
                        label = f"Class: {node.value}\\nSamples: {node.samples}"
                        color = 'lightgreen'
                    else:
                        label = f"Value: {node.value:.3f}\\nSamples: {node.samples}"
                        color = 'lightblue'
                else:
                    if isinstance(node.threshold, str):
                        condition = f"{node.feature} == '{node.threshold}'"
                    else:
                        condition = f"{node.feature} ≤ {node.threshold:.3f}"
                    
                    label = f"{condition}\\nSamples: {node.samples}"
                    color = 'lightcoral'
                
                dot.node(node_id, label, fillcolor=color)
                
                if not node.is_leaf():
                    left_id = f"{node_id}_L"
                    right_id = f"{node_id}_R"
                    
                    if node.left:
                        add_nodes(node.left, left_id)
                        dot.edge(node_id, left_id, label='True', color='green')
                    
                    if node.right:
                        add_nodes(node.right, right_id)
                        dot.edge(node_id, right_id, label='False', color='red')
            
            add_nodes(self.root)
            
            # Render to PNG
            dot.render(filename, format='png', cleanup=True)
            print(f"Tree visualization saved as: {png_filename}")
            
        except ImportError:
            print("Graphviz not available for image generation.")
        except Exception as e:
            print(f"Could not create tree image: {e}")
        
        # Always show text representation
        print("\nDecision Tree Structure:")
        print("=" * 50)
        if self.root.is_leaf():
            if self.is_classification:
                print(f"Root (Leaf): class {self.root.value} (samples: {self.root.samples})")
            else:
                print(f"Root (Leaf): value {self.root.value:.3f} (samples: {self.root.samples})")
        else:
            if isinstance(self.root.threshold, str):
                condition = f"{self.root.feature} == '{self.root.threshold}'"
            else:
                condition = f"{self.root.feature} ≤ {self.root.threshold:.3f}"
            print(f"Root: {condition} (samples: {self.root.samples})")
            
            # Plot children with proper formatting
            if self.root.left:
                print("│")
                self._plot_tree_text(self.root.left, 1, "", True)
            if self.root.right:
                print("│")
                self._plot_tree_text(self.root.right, 1, "", False)
        
        print("=" * 50)
