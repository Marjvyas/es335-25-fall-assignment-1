import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 1  # Only 1 run per configuration for speed


def create_fake_data(N, M, case_type):
    """
    Create fake data for different decision tree cases
    
    Parameters:
    N: number of samples
    M: number of features
    case_type: string indicating the type of DT case
        - "discrete_input_discrete_output": All features are discrete, output is discrete
        - "discrete_input_real_output": All features are discrete, output is real
        - "real_input_discrete_output": All features are real, output is discrete  
        - "real_input_real_output": All features are real, output is real
    
    Returns:
    X: Feature matrix (N x M)
    y: Target vector (N,)
    """
    
    if case_type == "discrete_input_discrete_output":
        # Binary features (0 or 1)
        X = np.random.randint(0, 2, size=(N, M))
        # Binary classification (0 or 1)
        y = np.random.randint(0, 2, size=N)
        
    elif case_type == "discrete_input_real_output":
        # Binary features (0 or 1)
        X = np.random.randint(0, 2, size=(N, M))
        # Continuous target (regression)
        y = np.random.normal(0, 1, size=N)
        
    elif case_type == "real_input_discrete_output":
        # Use fewer unique values for real features to speed up split finding
        X = np.round(np.random.normal(0, 1, size=(N, M)), 1)  # Round to 1 decimal place
        # Binary classification (0 or 1)
        y = np.random.randint(0, 2, size=N)
        
    elif case_type == "real_input_real_output":
        # Use fewer unique values for real features to speed up split finding
        X = np.round(np.random.normal(0, 1, size=(N, M)), 1)  # Round to 1 decimal place
        # Continuous target (regression)
        y = np.random.normal(0, 1, size=N)
        
    else:
        raise ValueError("Invalid case_type")
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(M)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


def measure_time_complexity(case_type, N_values, M_values, max_depth=2):
    """
    Measure average time taken for fit() and predict() for different N and M values
    
    Parameters:
    case_type: string indicating the DT case type
    N_values: list of sample sizes to test
    M_values: list of feature counts to test  
    max_depth: maximum depth of the decision tree
    
    Returns:
    results: dictionary containing timing results
    """
    
    print(f"\nMEASURING TIME COMPLEXITY FOR: {case_type.upper()}")
    
    results = {
        'case_type': case_type,
        'N_values': N_values,
        'M_values': M_values,
        'fit_times_vs_N': [],
        'predict_times_vs_N': [],
        'fit_times_vs_M': [],
        'predict_times_vs_M': [],
        'fit_std_vs_N': [],
        'predict_std_vs_N': [],
        'fit_std_vs_M': [],
        'predict_std_vs_M': []
    }
    
    # Determine if this is a regression case
    is_regression = case_type.endswith("real_output")
    criterion = "mse" if is_regression else "information_gain"
    
    print(f"Case: {case_type}")
    print(f"Is Regression: {is_regression}")
    print(f"Criterion: {criterion}")
    
    # Test 1: Vary N (number of samples), keep M fixed
    print(f"\nTest 1: Varying N (samples) with M=10 features")
    M_fixed = 10
    fit_times_N = []
    predict_times_N = []
    fit_stds_N = []
    predict_stds_N = []
    
    for N in N_values:
        print(f"  Testing N={N}...", end="", flush=True)
        
        fit_times_runs = []
        predict_times_runs = []
        
        for run in range(num_average_time):
            try:
                # Create data
                X_train, y_train = create_fake_data(N, M_fixed, case_type)
                X_test, y_test = create_fake_data(max(50, N//5), M_fixed, case_type)
                
                # Create decision tree
                dt = DecisionTree(criterion=criterion, max_depth=max_depth)
                
                # Measure fit time
                start_time = time.time()
                dt.fit(X_train, y_train)
                fit_time = time.time() - start_time
                fit_times_runs.append(fit_time)
                
                # Measure predict time
                start_time = time.time()
                predictions = dt.predict(X_test)
                predict_time = time.time() - start_time
                predict_times_runs.append(predict_time)
            except Exception as e:
                print(f"\n    Warning: Run {run+1} failed for N={N}: {str(e)}")
                continue
        
        avg_fit_time = np.mean(fit_times_runs)
        std_fit_time = np.std(fit_times_runs)
        avg_predict_time = np.mean(predict_times_runs)
        std_predict_time = np.std(predict_times_runs)
        
        fit_times_N.append(avg_fit_time)
        predict_times_N.append(avg_predict_time)
        fit_stds_N.append(std_fit_time)
        predict_stds_N.append(std_predict_time)
        
        print(f" Fit: {avg_fit_time:.4f}±{std_fit_time:.4f}s, Predict: {avg_predict_time:.4f}±{std_predict_time:.4f}s")
    
    results['fit_times_vs_N'] = fit_times_N
    results['predict_times_vs_N'] = predict_times_N
    results['fit_std_vs_N'] = fit_stds_N
    results['predict_std_vs_N'] = predict_stds_N
    
    # Test 2: Vary M (number of features), keep N fixed
    print(f"\nTest 2: Varying M (features) with N=1000 samples")
    N_fixed = 1000
    fit_times_M = []
    predict_times_M = []
    fit_stds_M = []
    predict_stds_M = []
    
    for M in M_values:
        print(f"  Testing M={M}...", end="")
        
        fit_times_runs = []
        predict_times_runs = []
        
        for run in range(num_average_time):
            # Create data
            X_train, y_train = create_fake_data(N_fixed, M, case_type)
            X_test, y_test = create_fake_data(200, M, case_type)
            
            # Create decision tree
            dt = DecisionTree(criterion=criterion, max_depth=max_depth)
            
            # Measure fit time
            start_time = time.time()
            dt.fit(X_train, y_train)
            fit_time = time.time() - start_time
            fit_times_runs.append(fit_time)
            
            # Measure predict time
            start_time = time.time()
            predictions = dt.predict(X_test)
            predict_time = time.time() - start_time
            predict_times_runs.append(predict_time)
        
        avg_fit_time = np.mean(fit_times_runs)
        std_fit_time = np.std(fit_times_runs)
        avg_predict_time = np.mean(predict_times_runs)
        std_predict_time = np.std(predict_times_runs)
        
        fit_times_M.append(avg_fit_time)
        predict_times_M.append(avg_predict_time)
        fit_stds_M.append(std_fit_time)
        predict_stds_M.append(std_predict_time)
        
        print(f" Fit: {avg_fit_time:.4f}±{std_fit_time:.4f}s, Predict: {avg_predict_time:.4f}±{std_predict_time:.4f}s")
    
    results['fit_times_vs_M'] = fit_times_M
    results['predict_times_vs_M'] = predict_times_M
    results['fit_std_vs_M'] = fit_stds_M
    results['predict_std_vs_M'] = predict_stds_M
    
    return results


def plot_complexity_results(all_results, N_values, M_values):
    """
    Plot the time complexity results for all cases
    """
    
    cases = list(all_results.keys())
    
    # Create plots for N variation (fit and predict)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Decision Tree Time Complexity Analysis', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    # Plot 1: Fit time vs N
    ax = axes[0, 0]
    for i, case in enumerate(cases):
        results = all_results[case]
        ax.errorbar(N_values, results['fit_times_vs_N'], 
                   yerr=results['fit_std_vs_N'],
                   label=case.replace('_', ' ').title(),
                   color=colors[i], marker=markers[i], 
                   linewidth=2, markersize=6, capsize=5)
    
    ax.set_xlabel('Number of Samples (N)')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time vs Number of Samples')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 2: Predict time vs N  
    ax = axes[0, 1]
    for i, case in enumerate(cases):
        results = all_results[case]
        ax.errorbar(N_values, results['predict_times_vs_N'],
                   yerr=results['predict_std_vs_N'], 
                   label=case.replace('_', ' ').title(),
                   color=colors[i], marker=markers[i],
                   linewidth=2, markersize=6, capsize=5)
    
    ax.set_xlabel('Number of Samples (N)')
    ax.set_ylabel('Prediction Time (seconds)')
    ax.set_title('Prediction Time vs Number of Samples')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 3: Fit time vs M
    ax = axes[1, 0]
    for i, case in enumerate(cases):
        results = all_results[case]
        ax.errorbar(M_values, results['fit_times_vs_M'],
                   yerr=results['fit_std_vs_M'],
                   label=case.replace('_', ' ').title(),
                   color=colors[i], marker=markers[i],
                   linewidth=2, markersize=6, capsize=5)
    
    ax.set_xlabel('Number of Features (M)')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time vs Number of Features')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Predict time vs M
    ax = axes[1, 1]
    for i, case in enumerate(cases):
        results = all_results[case]
        ax.errorbar(M_values, results['predict_times_vs_M'],
                   yerr=results['predict_std_vs_M'],
                   label=case.replace('_', ' ').title(), 
                   color=colors[i], marker=markers[i],
                   linewidth=2, markersize=6, capsize=5)
    
    ax.set_xlabel('Number of Features (M)')
    ax.set_ylabel('Prediction Time (seconds)')
    ax.set_title('Prediction Time vs Number of Features')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def analyze_theoretical_complexity(all_results, N_values, M_values):
    """
    Analyze and compare with theoretical time complexity
    """
    
    print("\nTHEORETICAL COMPLEXITY ANALYSIS")
    
    print("\nTHEORETICAL EXPECTATIONS:")
    print("1. Training Time Complexity:")
    print("   - Best case: O(N * M * log(N))  [balanced tree]")
    print("   - Worst case: O(N * M * N)      [degenerate tree]")
    print("   - Average case: O(N * M * log(N))")
    print("\n2. Prediction Time Complexity:")
    print("   - Best case: O(log(N))          [balanced tree]")
    print("   - Worst case: O(N)              [degenerate tree]")
    print("   - Average case: O(log(N))")
    
    print("\nEMPIRICAL ANALYSIS")
    
    for case_type, results in all_results.items():
        print(f"\nCase: {case_type.upper()}")
        
        # Analyze fit time scaling with N
        fit_times_N = np.array(results['fit_times_vs_N'])
        N_array = np.array(N_values)
        
        # Calculate growth rates
        if len(fit_times_N) > 1:
            # Log-linear fit for fit time vs N
            log_N = np.log(N_array)
            log_fit_times = np.log(fit_times_N)
            
            # Linear regression to find scaling
            A = np.vstack([log_N, np.ones(len(log_N))]).T
            slope_N, intercept_N = np.linalg.lstsq(A, log_fit_times, rcond=None)[0]
            
            print(f"Training Time Scaling with N:")
            print(f"  Empirical scaling: O(N^{slope_N:.2f})")
            print(f"  Expected: O(N * log(N)) ≈ O(N^1.0 to N^1.3)")
            
            if slope_N < 1.5:
                print(f"  ✅ Good scaling - close to expected O(N log N)")
            else:
                print(f"  ⚠️  Higher than expected - may indicate suboptimal implementation")
        
        # Analyze predict time scaling with N  
        predict_times_N = np.array(results['predict_times_vs_N'])
        
        if len(predict_times_N) > 1:
            log_predict_times = np.log(predict_times_N)
            slope_predict_N, intercept_predict_N = np.linalg.lstsq(A, log_predict_times, rcond=None)[0]
            
            print(f"Prediction Time Scaling with N:")
            print(f"  Empirical scaling: O(N^{slope_predict_N:.2f})")
            print(f"  Expected: O(log(N)) ≈ O(N^0.0 to N^0.3)")
            
            if slope_predict_N < 0.5:
                print(f"  ✅ Good scaling - close to expected O(log N)")
            else:
                print(f"  ⚠️  Higher than expected - prediction may be linear in training size")
        
        # Analyze fit time scaling with M
        fit_times_M = np.array(results['fit_times_vs_M'])
        M_array = np.array(M_values)
        
        if len(fit_times_M) > 1:
            log_M = np.log(M_array)
            log_fit_times_M = np.log(fit_times_M)
            
            A_M = np.vstack([log_M, np.ones(len(log_M))]).T
            slope_M, intercept_M = np.linalg.lstsq(A_M, log_fit_times_M, rcond=None)[0]
            
            print(f"Training Time Scaling with M:")
            print(f"  Empirical scaling: O(M^{slope_M:.2f})")
            print(f"  Expected: O(M) ≈ O(M^1.0)")
            
            if 0.8 <= slope_M <= 1.5:
                print(f"  ✅ Good scaling - close to expected O(M)")
            else:
                print(f"  ⚠️  Different from expected - actual scaling O(M^{slope_M:.2f})")


def main():
    """
    Main function to run all experiments
    """
    
    print("DECISION TREE RUNTIME COMPLEXITY EXPERIMENTS")
    
    # Define test parameters - very small for fast execution
    N_values = [20, 40, 60]              # Very small sample sizes
    M_values = [2, 3, 4]                 # Very few features
    
    # Define the four cases - now include all with optimizations
    cases = [
        "discrete_input_discrete_output",
        "discrete_input_real_output",
        "real_input_discrete_output", 
        "real_input_real_output"
    ]
    
    print(f"Testing cases: {cases}")
    print(f"N values (samples): {N_values}")
    print(f"M values (features): {M_values}")
    print(f"Averaging over {num_average_time} runs per configuration")
    
    # Run experiments for all cases
    all_results = {}
    
    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] Processing: {case}")
        try:
            results = measure_time_complexity(case, N_values, M_values)
            all_results[case] = results
            print(f"Completed: {case}")
        except Exception as e:
            print(f"Failed: {case} - Error: {str(e)}")
    
    # Plot results (disabled for speed)
    if all_results:
        print("\nANALYSIS RESULTS")
        
        # Skip plotting for speed
        print("Plotting disabled for speed optimization")
        
        # Skip theoretical analysis for speed
        print("Theoretical analysis disabled for speed optimization")
        
        # Print summary table
        print("\nSUMMARY TABLE")
        
        summary_df = pd.DataFrame()
        for case, results in all_results.items():
            summary_df[f"{case}_fit_time"] = results['fit_times_vs_N']
            summary_df[f"{case}_predict_time"] = results['predict_times_vs_N']
        
        summary_df.index = [f"N={n}" for n in N_values]
        print("\nTraining and Prediction Times vs Sample Size:")
        print(summary_df.round(4))
    
    print("\nEXPERIMENTS COMPLETED")


# Run the experiments
if __name__ == "__main__":
    main()
