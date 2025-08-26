from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # Reset indices to ensure proper comparison
    y_hat_reset = y_hat.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    return (y_hat_reset == y_reset).sum() / len(y_reset)


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size
    
    # Reset indices to ensure proper comparison
    y_hat_reset = y_hat.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    true_positives = ((y_hat_reset == cls) & (y_reset == cls)).sum()
    predicted_positives = (y_hat_reset == cls).sum()
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size
    
    # Reset indices to ensure proper comparison
    y_hat_reset = y_hat.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    true_positives = ((y_hat_reset == cls) & (y_reset == cls)).sum()
    actual_positives = (y_reset == cls).sum()
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size
    
    # Reset indices to ensure proper subtraction
    y_hat_reset = y_hat.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    return np.sqrt(np.mean((y_hat_reset - y_reset) ** 2))


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size
    
    # Reset indices to ensure proper subtraction
    y_hat_reset = y_hat.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    
    return np.mean(np.abs(y_hat_reset - y_reset))
