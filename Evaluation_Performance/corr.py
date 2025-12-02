import pandas as pd
import numpy as np

# Define the file path for the detailed predictions CSV
file_path = './result_I20R60_latest/predictions_detailed.csv'
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

def calc_cross_sectional_corr(group):
    """
    Calculates cross-sectional correlation coefficients for a single time slice (day).
    
    Pearson: Measures linear correlation (often used for replicating Table 2 results).
    Spearman: Measures rank correlation (used to assess stock selection ability/rank performance).
    """
    # Calculate Pearson correlation (Prob_Up vs Ret_60d)
    # Prob_Up: Predicted probability of up movement
    # Ret_60d: Actual return over the next 60 days
    pearson = group['Prob_Up'].corr(group['Ret_60d'], method='pearson')
    
    # Calculate Spearman rank correlation (Prob_Up vs Ret_60d)
    spearman = group['Prob_Up'].corr(group['Ret_60d'], method='spearman')
    
    return pd.Series({'Pearson': pearson, 'Spearman': spearman})

def calc_daily_accuracy(group):
    """
    Calculates the prediction accuracy for a single time slice (day).
    Definition from paper: If Predicted 'up' probability (Prob_Up) > 0.5, it is predicted as 1 (Up).
    """
    # Generate 0/1 prediction label based on Prob_Up > 0.5 threshold
    calc_prediction = (group['Prob_Up'] > 0.5).astype(int)
    
    # Ensure True_Label is also an integer type for comparison
    true_label = group['True_Label'].astype(int)
    
    # Calculate the proportion of correct predictions (Accuracy)
    return (true_label == calc_prediction).mean()

# -------------------------------------------------------
# Execution Block
# -------------------------------------------------------

print("Calculating metrics...")

# 2. Calculate Cross-sectional Correlations
# include_groups=False is used to suppress a pandas FutureWarning
daily_correlations = df.groupby('Date').apply(calc_cross_sectional_corr, include_groups=False)

# Average the daily correlations over the entire time series
avg_correlations = daily_correlations.mean()

# 3. Calculate Accuracy
daily_accuracy = df.groupby('Date').apply(calc_daily_accuracy, include_groups=False)

# Average the daily accuracy over the entire time series
final_accuracy = daily_accuracy.mean()

# -------------------------------------------------------
# Output Results
# -------------------------------------------------------

print("-" * 30)
print("Paper Replication Results")
print("-" * 30)
print(f"Out-of-Sample Accuracy (Acc.): {final_accuracy:.2%}")
print(f"Avg Cross-sectional Pearson Corr.: {avg_correlations['Pearson']:.4f} (Target for Table 2)")
print(f"Avg Cross-sectional Spearman Corr.: {avg_correlations['Spearman']:.4f} (Rank Capability)")
print("-" * 30)