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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

try:
    import graphviz
    import os
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

np.random.seed(42)


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        # Convert discrete features to one-hot encoding if needed
        self.original_X = X.copy()
        
        # Check if any column is categorical or non-numeric
        categorical_cols = []
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                categorical_cols.append(col)
        
        if categorical_cols:
            X = one_hot_encoding(X)
        
        # Determine if this is a regression or classification problem
        self.is_regression = check_ifreal(y)
        
        # Store feature names after encoding
        self.feature_names = X.columns.tolist()
        
        # Build the tree
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int):
        """
        Recursively build the decision tree with enhanced statistics tracking
        """
        n_samples = len(y)
        
        # Calculate node statistics
        if self.is_regression:
            node_value = y.mean()
            # Calculate MSE for this node
            mse = np.mean((y - node_value) ** 2) if len(y) > 0 else 0.0
        else:
            node_value = y.mode()[0] if len(y) > 0 else 0
            # Calculate Gini impurity
            if len(y) == 0:
                gini = 0.0
            else:
                proportions = y.value_counts(normalize=True)
                gini = 1.0 - sum(p**2 for p in proportions)
        
        # Base cases - stopping criteria
        if depth >= self.max_depth or len(y.unique()) == 1 or len(X.columns) == 0 or n_samples <= 1:
            if self.is_regression:
                return {
                    'type': 'leaf',
                    'value': node_value,
                    'samples': n_samples,
                    'mse': 0.0,  # Pure leaf has 0 MSE
                    'is_leaf': True
                }
            else:
                return {
                    'type': 'leaf', 
                    'value': node_value,
                    'samples': n_samples,
                    'gini': 0.0,  # Pure leaf has 0 Gini
                    'is_leaf': True
                }
        
        # Choose the best criterion based on problem type
        if self.is_regression:
            criterion = "mse"
        else:
            criterion = self.criterion
        
        # Find the best attribute to split on
        current_features = pd.Series(X.columns)
        result = opt_split_attribute(X, y, criterion, current_features)
        
        if result is None or result[0] is None:
            if self.is_regression:
                return {
                    'type': 'leaf',
                    'value': node_value,
                    'samples': n_samples,
                    'mse': mse,
                    'is_leaf': True
                }
            else:
                return {
                    'type': 'leaf',
                    'value': node_value,
                    'samples': n_samples,
                    'gini': gini,
                    'is_leaf': True
                }
        
        best_attribute, split_value = result
        
        # Create a node with statistics
        if self.is_regression:
            node = {
                'attribute': best_attribute,
                'split_value': split_value,
                'children': {},
                'is_leaf': False,
                'samples': n_samples,
                'mse': mse,
                'value': node_value
            }
        else:
            node = {
                'attribute': best_attribute,
                'split_value': split_value,
                'children': {},
                'is_leaf': False,
                'samples': n_samples,
                'gini': gini,
                'value': node_value
            }
        
        # Split data based on feature type
        if split_value is not None:
            # Real-valued feature: binary split
            X_left, y_left = split_data(X, y, best_attribute, f'<={split_value}')
            X_right, y_right = split_data(X, y, best_attribute, f'>{split_value}')
            
            if len(y_left) > 0:
                X_left_reduced = X_left.drop(columns=[best_attribute])
                node['children'][f'<={split_value}'] = self._build_tree(X_left_reduced, y_left, depth + 1)
            else:
                if self.is_regression:
                    node['children'][f'<={split_value}'] = {
                        'type': 'leaf',
                        'value': node_value,
                        'samples': 0,
                        'mse': 0.0,
                        'is_leaf': True
                    }
                else:
                    node['children'][f'<={split_value}'] = {
                        'type': 'leaf',
                        'value': node_value,
                        'samples': 0,
                        'gini': 0.0,
                        'is_leaf': True
                    }
            
            if len(y_right) > 0:
                X_right_reduced = X_right.drop(columns=[best_attribute])
                node['children'][f'>{split_value}'] = self._build_tree(X_right_reduced, y_right, depth + 1)
            else:
                if self.is_regression:
                    node['children'][f'>{split_value}'] = {
                        'type': 'leaf',
                        'value': node_value,
                        'samples': 0,
                        'mse': 0.0,
                        'is_leaf': True
                    }
                else:
                    node['children'][f'>{split_value}'] = {
                        'type': 'leaf',
                        'value': node_value,
                        'samples': 0,
                        'gini': 0.0,
                        'is_leaf': True
                    }
        else:
            # Discrete feature: split for each unique value
            unique_values = X[best_attribute].unique()
            
            for value in unique_values:
                X_subset, y_subset = split_data(X, y, best_attribute, value)
                
                if len(y_subset) == 0:
                    # If no data points, create leaf with current node's value
                    if self.is_regression:
                        node['children'][value] = {
                            'type': 'leaf',
                            'value': node_value,
                            'samples': 0,
                            'mse': 0.0,
                            'is_leaf': True
                        }
                    else:
                        node['children'][value] = {
                            'type': 'leaf',
                            'value': node_value,
                            'samples': 0,
                            'gini': 0.0,
                            'is_leaf': True
                        }
                else:
                    # Remove the current attribute from further consideration
                    X_subset_reduced = X_subset.drop(columns=[best_attribute])
                    node['children'][value] = self._build_tree(X_subset_reduced, y_subset, depth + 1)
        
        return node

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        # Apply same preprocessing as in fit
        categorical_cols = []
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                categorical_cols.append(col)
        
        if categorical_cols:
            X = one_hot_encoding(X)
        
        # Make predictions for each row
        predictions = []
        for idx in range(len(X)):
            row = X.iloc[idx]
            prediction = self._predict_single(row, self.tree)
            predictions.append(prediction)
        
        return pd.Series(predictions)
    
    def _predict_single(self, row, node):
        """
        Predict for a single row - updated to handle enhanced node structure
        """
        # Handle different node formats
        if not isinstance(node, dict):
            # Old format - just a value
            return node
        
        # Check if it's a leaf node in new format
        if node.get('is_leaf', False) or node.get('type') == 'leaf':
            return node.get('value', 0)
        
        # Check if it's an internal node without 'attribute' (old format leaf stored as dict)
        if 'attribute' not in node:
            return node.get('value', 0)
        
        # It's an internal node
        attribute = node['attribute']
        split_value = node.get('split_value', None)
        
        # Check if attribute exists in the row
        if attribute not in row.index:
            # If attribute doesn't exist, return majority/mean from children
            child_values = [child for child in node['children'].values() 
                          if not isinstance(child, dict)]
            if child_values:
                if self.is_regression:
                    return np.mean(child_values)
                else:
                    return pd.Series(child_values).mode()[0]
            else:
                # If no leaf children, pick first child and traverse
                first_child = list(node['children'].values())[0]
                return self._predict_single(row, first_child)
        
        attribute_value = row[attribute]
        
        if split_value is not None:
            # Real-valued feature: use binary split
            if attribute_value <= split_value:
                key = f'<={split_value}'
            else:
                key = f'>{split_value}'
            
            if key in node['children']:
                return self._predict_single(row, node['children'][key])
            else:
                # Fallback
                if self.is_regression:
                    return 0.0
                else:
                    return 0
        else:
            # Discrete feature: exact match
            if attribute_value in node['children']:
                return self._predict_single(row, node['children'][attribute_value])
            else:
                # If we haven't seen this value, return majority/mean from children
                child_values = []
                for child in node['children'].values():
                    if isinstance(child, dict):
                        # Need to traverse further to get leaf values
                        continue
                    else:
                        child_values.append(child)
                
                if child_values:
                    if self.is_regression:
                        return np.mean(child_values)
                    else:
                        return pd.Series(child_values).mode()[0]
                else:
                    # Default fallback
                    if self.is_regression:
                        return 0.0
                    else:
                        return list(node['children'].values())[0]

    def plot(self) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        def _plot_tree(node, indent="", attribute_value=""):
            if isinstance(node, dict):
                # Check if it's a leaf node in new format
                if node.get('is_leaf', False) or node.get('type') == 'leaf':
                    # Leaf node
                    leaf_value = node.get('value', 'Unknown')
                    if attribute_value:
                        print(f"{indent}{attribute_value}: {leaf_value}")
                    else:
                        print(f"{indent}{leaf_value}")
                elif 'attribute' in node:
                    # Internal node
                    attribute = node['attribute']
                    if attribute_value:
                        print(f"{indent}{attribute_value}: ?({attribute})")
                    else:
                        print(f"{indent}?({attribute})")
                    
                    # Plot children
                    for value, child in node['children'].items():
                        _plot_tree(child, indent + "  ", str(value))
                else:
                    # Old format leaf node stored as dict
                    leaf_value = node.get('value', node)
                    if attribute_value:
                        print(f"{indent}{attribute_value}: {leaf_value}")
                    else:
                        print(f"{indent}{leaf_value}")
            else:
                # Old format leaf node
                if attribute_value:
                    print(f"{indent}{attribute_value}: {node}")
                else:
                    print(f"{indent}{node}")
        
        if hasattr(self, 'tree'):
            _plot_tree(self.tree)
        else:
            print("Tree not fitted yet!")

    def create_graph(self, filename=None, feature_names=None):
        """
        Create a graphviz visualization of the decision tree
        
        Parameters:
        filename: str, name for the output file (without extension)
        feature_names: list, names of the features
        
        Returns:
        graphviz.Source object
        """
        if not GRAPHVIZ_AVAILABLE:
            print("Graphviz not available. Please install: pip install graphviz")
            return None
            
        if not hasattr(self, 'tree'):
            print("Tree not fitted yet!")
            return None
            
        if feature_names is None:
            if hasattr(self, 'feature_names_'):
                feature_names = self.feature_names_
            else:
                feature_names = [f'feature_{i}' for i in range(len(self.tree.get('children', {})))]
        
        # Create DOT format string
        dot_lines = ['digraph Tree {']
        dot_lines.append('node [shape=box, style="filled,rounded", color="black", fontname="helvetica"];')
        dot_lines.append('edge [fontname="helvetica"];')
        
        node_counter = [0]  # Use list to allow modification in nested function
        
        def _add_node_to_dot(node, parent_id=None, edge_label=""):
            current_id = node_counter[0]
            node_counter[0] += 1
            
            if isinstance(node, dict):
                # Check if it's a leaf node
                if node.get('is_leaf', False) or node.get('type') == 'leaf':
                    # Leaf node with actual statistics
                    samples_count = node.get('samples', 0)
                    leaf_value = node.get('value', 0)
                    
                    if self.is_regression:
                        mse_value = node.get('mse', 0.0)
                        node_info = f"squared_error = {mse_value:.3f}\\nsamples = {samples_count}\\nvalue = {leaf_value:.1f}"
                        
                        # Color based on value range
                        if leaf_value < 20:
                            color = "#fff2e6"  # Very light orange
                        elif leaf_value < 30:
                            color = "#ffe6cc"  # Light orange
                        else:
                            color = "#ffcc99"  # Darker orange
                    else:
                        gini_value = node.get('gini', 0.0)
                        class_value = int(leaf_value)
                        node_info = f"gini = {gini_value:.3f}\\nsamples = {samples_count}\\nvalue = [{samples_count if class_value else 0}, {samples_count if not class_value else 0}]"
                        
                        # Color based on class
                        if class_value == 0:
                            color = "#ffe6cc"  # Light orange for class 0
                        else:
                            color = "#e6ffe6"  # Light green for class 1
                    
                    dot_lines.append(f'{current_id} [label="{node_info}", fillcolor="{color}"];')
                    
                else:
                    # Internal node with actual statistics
                    attribute = node['attribute']
                    split_value = node.get('split_value', None)
                    samples_count = node.get('samples', 0)
                    node_value = node.get('value', 0)
                    
                    if split_value is not None:
                        # Real-valued split
                        condition_text = f"{attribute} <= {split_value:.1f}"
                    else:
                        # Discrete split
                        condition_text = f"{attribute}"
                    
                    if self.is_regression:
                        mse_value = node.get('mse', 0.0)
                        node_info = f"{condition_text}\\nsquared_error = {mse_value:.3f}\\nsamples = {samples_count}\\nvalue = {node_value:.1f}"
                    else:
                        gini_value = node.get('gini', 0.0)
                        node_info = f"{condition_text}\\ngini = {gini_value:.3f}\\nsamples = {samples_count}\\nvalue = [{int(samples_count/2)}, {int(samples_count/2)}]"
                    
                    color = "#ffffff"  # White background for internal nodes
                    dot_lines.append(f'{current_id} [label="{node_info}", fillcolor="{color}"];')
                    
                    # Add edges to children
                    for child_value, child_node in node['children'].items():
                        child_id = _add_node_to_dot(child_node, current_id, str(child_value))
                        
                        # Format edge label
                        if split_value is not None:
                            if child_value.startswith('<='):
                                edge_text = "True"
                            else:
                                edge_text = "False"
                        else:
                            edge_text = str(child_value)
                        
                        dot_lines.append(f'{current_id} -> {child_id} [label="{edge_text}"];')
                
            else:
                # Old format leaf node (fallback)
                if self.is_regression:
                    node_info = f"squared_error = 0.0\\nsamples = 1\\nvalue = {float(node):.1f}"
                    color = "#ffe6cc"
                else:
                    node_info = f"gini = 0.0\\nsamples = 1\\nvalue = [{int(node)}]"
                    color = "#ffe6cc" if node == 0 else "#e6ffe6"
                
                dot_lines.append(f'{current_id} [label="{node_info}", fillcolor="{color}"];')
            
            return current_id
        
        # Build the graph
        _add_node_to_dot(self.tree)
        dot_lines.append('}')
        
        dot_data = '\n'.join(dot_lines)
        
        # Create graphviz object
        graph = graphviz.Source(dot_data)
        graph.format = 'pdf'
        
        # Save if filename provided
        if filename:
            # Create figures directory if it doesn't exist
            figures_dir = "figures/decision-trees"
            os.makedirs(figures_dir, exist_ok=True)
            
            graph.render(f"{figures_dir}/{filename}", cleanup=True)
            print(f"Decision tree saved as {figures_dir}/{filename}.pdf")
        
        return graph

    def export_graphviz(self, feature_names=None):
        """
        Export decision tree in DOT format similar to sklearn's export_graphviz
        
        Parameters:
        feature_names: list, names of the features
        
        Returns:
        str: DOT format string
        """
        if not GRAPHVIZ_AVAILABLE:
            print("Graphviz not available. Please install: pip install graphviz")
            return None
            
        if not hasattr(self, 'tree'):
            return "Tree not fitted yet!"
            
        graph = self.create_graph(feature_names=feature_names)
        return graph.source if graph else None
