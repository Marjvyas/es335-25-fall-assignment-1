"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X, prefix_sep='_')

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_categorical_dtype(y)


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    if len(Y) == 0:
        return 0
    
    value_counts = Y.value_counts()
    probabilities = value_counts / len(Y)
    
    # Calculate entropy
    entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    return entropy_val


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    if len(Y) == 0:
        return 0
    
    value_counts = Y.value_counts()
    probabilities = value_counts / len(Y)
    
    # Calculate Gini index
    gini_val = 1 - np.sum(probabilities ** 2)
    return gini_val


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    Using formula: Gain(S,A) ≡ Entropy(S) − Σ(v∈Values(A)) (|Sv| · Entropy(Sv)/|S|)
    """
    if len(Y) == 0:
        return 0
    
    # Calculate initial entropy/impurity of S
    if criterion == "information_gain":
        initial_entropy = entropy(Y)
    elif criterion == "gini_index":
        initial_entropy = gini_index(Y)
    elif criterion == "mse":
        initial_entropy = np.var(Y)
    else:
        raise ValueError("Invalid criterion")
    
    # Calculate the weighted sum: Σ(v∈Values(A)) (|Sv| · Entropy(Sv)/|S|)
    weighted_sum = 0
    total_size = len(Y)  # |S|
    unique_values = attr.unique()  # Values(A)
    
    for value in unique_values:  # for each v ∈ Values(A)
        subset_mask = attr == value
        subset_Y = Y[subset_mask]  # Sv
        
        if len(subset_Y) == 0:
            continue
            
        subset_size = len(subset_Y)  # |Sv|
        
        # Calculate Entropy(Sv) based on criterion
        if criterion == "information_gain":
            subset_entropy = entropy(subset_Y)
        elif criterion == "gini_index":
            subset_entropy = gini_index(subset_Y)
        elif criterion == "mse":
            subset_entropy = np.var(subset_Y)
        
        # Add (|Sv| · Entropy(Sv)/|S|) to the sum
        weighted_sum += (subset_size * subset_entropy) / total_size
    
    # Information gain = Entropy(S) - weighted_sum
    return initial_entropy - weighted_sum


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: tuple (attribute to split upon, split_value for real features or None for discrete)
    """
    best_gain = -np.inf
    best_attribute = None
    best_split_value = None
    
    for feature in features:
        if feature in X.columns:
            if check_ifreal(X[feature]):
                # For real-valued features, find the best threshold
                unique_vals = sorted(X[feature].unique())
                for i in range(len(unique_vals) - 1):
                    threshold = (unique_vals[i] + unique_vals[i + 1]) / 2
                    
                    # Create binary split based on threshold
                    left_mask = X[feature] <= threshold
                    right_mask = X[feature] > threshold
                    
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    
                    # Calculate information gain for this threshold
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    
                    if criterion == "information_gain":
                        initial_entropy = entropy(y)
                        left_entropy = entropy(y_left) if len(y_left) > 0 else 0
                        right_entropy = entropy(y_right) if len(y_right) > 0 else 0
                        
                        weighted_entropy = (len(y_left) * left_entropy + len(y_right) * right_entropy) / len(y)
                        gain = initial_entropy - weighted_entropy
                    
                    elif criterion == "gini_index":
                        initial_gini = gini_index(y)
                        left_gini = gini_index(y_left) if len(y_left) > 0 else 0
                        right_gini = gini_index(y_right) if len(y_right) > 0 else 0
                        
                        weighted_gini = (len(y_left) * left_gini + len(y_right) * right_gini) / len(y)
                        gain = initial_gini - weighted_gini
                    
                    elif criterion == "mse":
                        # For MSE, we want to minimize the weighted MSE
                        if len(y_left) > 0:
                            left_mse = ((y_left - y_left.mean()) ** 2).mean()
                        else:
                            left_mse = 0
                        
                        if len(y_right) > 0:
                            right_mse = ((y_right - y_right.mean()) ** 2).mean()
                        else:
                            right_mse = 0
                        
                        # For MSE, gain is the reduction in MSE
                        initial_mse = ((y - y.mean()) ** 2).mean()
                        weighted_mse = (len(y_left) * left_mse + len(y_right) * right_mse) / len(y)
                        gain = initial_mse - weighted_mse
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_attribute = feature
                        best_split_value = threshold
            else:
                # For discrete features, use the existing information_gain function
                gain = information_gain(y, X[feature], criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = feature
                    best_split_value = None
    
    return best_attribute, best_split_value


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon (threshold for real, category for discrete)

    return: splitted data(Input and output)
    """
    if check_ifreal(X[attribute]):
        # For real-valued features, split based on threshold
        if isinstance(value, str) and value.startswith('<='):
            threshold = float(value[2:])
            mask = X[attribute] <= threshold
        elif isinstance(value, str) and value.startswith('>'):
            threshold = float(value[1:])
            mask = X[attribute] > threshold
        else:
            # For backwards compatibility
            mask = X[attribute] == value
    else:
        # For discrete features, exact match
        mask = X[attribute] == value
    
    return X[mask], y[mask]
