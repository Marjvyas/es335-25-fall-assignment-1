import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

print("Original dataset shape:", data.shape)
print("First few rows:")
print(data.head())
print("\nDataset info:")
print(data.info())
print("\nMissing values per column:")
print(data.isnull().sum())

# Clean the above data by removing redundant columns and rows with junk values
print("\nDATA CLEANING")

# Remove the 'car name' column as it's just an identifier
data = data.drop('car name', axis=1)

# Handle missing values in 'horsepower' (marked as '?' in the dataset)
print(f"Unique values in horsepower: {data['horsepower'].unique()[:10]}...")
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
print(f"Missing values in horsepower after conversion: {data['horsepower'].isnull().sum()}")

# Remove rows with missing horsepower values
data = data.dropna()
print(f"Dataset shape after removing missing values: {data.shape}")

# Separate features and target
X = data.drop('mpg', axis=1)
y = data['mpg']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target statistics:")
print(y.describe())

print("\nPART (A): USAGE OF CUSTOM DECISION TREE FOR AUTO MPG REGRESSION")

print("Original dataset shape:", data.shape)
print("First few rows:")
print(data.head())
print("\nDataset info:")
print(data.info())
print("\nMissing values per column:")
print(data.isnull().sum())

# Clean the above data by removing redundant columns and rows with junk values
print("\nDATA CLEANING")

# Remove the 'car name' column as it's just an identifier
data = data.drop('car name', axis=1)

# Handle missing values in 'horsepower' (marked as '?' in the dataset)
print(f"Unique values in horsepower: {data['horsepower'].unique()[:10]}...")
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
print(f"Missing values in horsepower after conversion: {data['horsepower'].isnull().sum()}")

# Remove rows with missing horsepower values
data = data.dropna()
print(f"Dataset shape after removing missing values: {data.shape}")

# Separate features and target
X = data.drop('mpg', axis=1)
y = data['mpg']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target statistics:")
print(y.describe())

print("\n" + "="*80)
print("PART (A): USAGE OF CUSTOM DECISION TREE FOR AUTO MPG REGRESSION")
print("="*80)

# Split the data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train our custom decision tree for regression
print("\nTraining custom Decision Tree (max_depth=5)...")
start_time = time.time()
custom_tree = DecisionTree(criterion="mse", max_depth=5)
custom_tree.fit(X_train, y_train)
custom_train_time = time.time() - start_time

# Make predictions
print("Making predictions...")
y_pred_custom = custom_tree.predict(X_test)

# Calculate performance metrics
custom_rmse = rmse(y_pred_custom, y_test)
custom_mae = mae(y_pred_custom, y_test)

print(f"\nCustom Decision Tree Results:")
print(f"Training time: {custom_train_time:.4f} seconds")
print(f"RMSE: {custom_rmse:.4f}")
print(f"MAE: {custom_mae:.4f}")

# Display tree structure
print(f"\nDecision Tree Structure:")
custom_tree.plot()

# Create graphical visualization
print(f"\nCreating graphical visualization...")
try:
    # Create feature names list
    feature_names = list(X.columns)
    
    # Create and save the graph
    graph = custom_tree.create_graph(
        filename="auto-mpg-custom-tree", 
        feature_names=feature_names
    )
    
    if graph:
        print("Graphical tree visualization created successfully!")
        print("PDF saved at: figures/decision-trees/auto-mpg-custom-tree.pdf")
        
        # Also try to view the tree inline (if possible)
        try:
            # This will try to display the tree directly
            graph.view(cleanup=True)
            print("Tree visualization opened in default viewer!")
        except Exception as view_error:
            print(f"Could not auto-open visualization: {view_error}")
            print("Please manually open the PDF file to see the graphical tree structure")
    else:
        print("Could not create graphical visualization (graphviz may not be installed)")
        
except Exception as e:
    print(f"Error creating graph: {e}")
    print("Tree visualization available in text format only")

print("\nPART (B): COMPARISON WITH SCIKIT-LEARN DECISION TREE")

# Train scikit-learn decision tree with same parameters
print("Training scikit-learn Decision Tree (max_depth=5)...")
start_time = time.time()
sklearn_tree = DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=42)
sklearn_tree.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time

# Make predictions
y_pred_sklearn = sklearn_tree.predict(X_test)

# Calculate performance metrics
sklearn_rmse = np.sqrt(np.mean((y_pred_sklearn - y_test) ** 2))
sklearn_mae = np.mean(np.abs(y_pred_sklearn - y_test))

print(f"\nScikit-learn Decision Tree Results:")
print(f"Training time: {sklearn_train_time:.4f} seconds")
print(f"RMSE: {sklearn_rmse:.4f}")
print(f"MAE: {sklearn_mae:.4f}")

print("\nPERFORMANCE COMPARISON")

comparison_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'Training Time (s)'],
    'Custom Tree': [custom_rmse, custom_mae, custom_train_time],
    'Scikit-learn Tree': [sklearn_rmse, sklearn_mae, sklearn_train_time],
    'Difference (Custom - SKLearn)': [
        custom_rmse - sklearn_rmse,
        custom_mae - sklearn_mae,
        custom_train_time - sklearn_train_time
    ]
})

print(comparison_df.to_string(index=False, float_format='%.4f'))

# Plot predictions comparison
plt.figure(figsize=(15, 5))

# Reset indices to ensure alignment
y_test_reset = y_test.reset_index(drop=True)
y_pred_custom_reset = y_pred_custom.reset_index(drop=True)
y_pred_sklearn_reset = pd.Series(y_pred_sklearn).reset_index(drop=True)

plt.subplot(1, 3, 1)
plt.scatter(y_test_reset, y_pred_custom_reset, alpha=0.6, color='blue')
plt.plot([y_test_reset.min(), y_test_reset.max()], [y_test_reset.min(), y_test_reset.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title(f'Custom Decision Tree\nRMSE: {custom_rmse:.3f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.scatter(y_test_reset, y_pred_sklearn_reset, alpha=0.6, color='green')
plt.plot([y_test_reset.min(), y_test_reset.max()], [y_test_reset.min(), y_test_reset.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title(f'Scikit-learn Decision Tree\nRMSE: {sklearn_rmse:.3f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
residuals_custom = y_test_reset - y_pred_custom_reset
residuals_sklearn = y_test_reset - y_pred_sklearn_reset
plt.scatter(y_pred_custom_reset, residuals_custom, alpha=0.6, color='blue', label='Custom Tree')
plt.scatter(y_pred_sklearn_reset, residuals_sklearn, alpha=0.6, color='green', label='Scikit-learn Tree')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted MPG')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis: Feature importance (for scikit-learn tree)
print(f"\nFeature Importance (Scikit-learn Tree):")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': sklearn_tree.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))

# Test with different max_depths
print(f"\nDEPTH ANALYSIS: TESTING DIFFERENT MAX_DEPTHS")

depths = [1, 2, 3, 4, 5, 6, 7, 8, 10]
custom_rmses = []
sklearn_rmses = []

for depth in depths:
    # Custom tree
    custom_tree_depth = DecisionTree(criterion="mse", max_depth=depth)
    custom_tree_depth.fit(X_train, y_train)
    custom_pred = custom_tree_depth.predict(X_test)
    custom_rmse_depth = rmse(custom_pred, y_test)
    custom_rmses.append(custom_rmse_depth)
    
    # Scikit-learn tree
    sklearn_tree_depth = DecisionTreeRegressor(criterion='squared_error', max_depth=depth, random_state=42)
    sklearn_tree_depth.fit(X_train, y_train)
    sklearn_pred = sklearn_tree_depth.predict(X_test)
    sklearn_rmse_depth = np.sqrt(np.mean((sklearn_pred - y_test) ** 2))
    sklearn_rmses.append(sklearn_rmse_depth)

# Plot depth analysis
plt.figure(figsize=(10, 6))
plt.plot(depths, custom_rmses, 'bo-', label='Custom Decision Tree', linewidth=2, markersize=8)
plt.plot(depths, sklearn_rmses, 'go-', label='Scikit-learn Decision Tree', linewidth=2, markersize=8)
plt.xlabel('Maximum Depth')
plt.ylabel('RMSE')
plt.title('RMSE vs Maximum Depth Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(depths)
plt.show()

# Summary table for different depths
depth_comparison = pd.DataFrame({
    'Max Depth': depths,
    'Custom Tree RMSE': custom_rmses,
    'Scikit-learn RMSE': sklearn_rmses,
    'Difference': [c - s for c, s in zip(custom_rmses, sklearn_rmses)]
})
print("\nRMSE Comparison Across Different Depths:")
print(depth_comparison.to_string(index=False, float_format='%.4f'))

print(f"\nEXPERIMENT COMPLETED")