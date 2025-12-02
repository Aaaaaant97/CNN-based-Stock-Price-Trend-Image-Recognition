import pandas as pd
import numpy as np


file_path = './result_I20R60_latest/predictions_detailed.csv'
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])

def calc_cross_sectional_corr(group):
    """
    Calculates cross-sectional correlation coefficients for a single time slice (day).
    
    Pearson: Measures linear correlation (Replication standard for Table 2).
    Spearman: Measures rank correlation (Rank capability / Used in Table 12).
    """
    # Calculate Pearson correlation (Prob_Up vs Ret_60d)
    pearson = group['Prob_Up'].corr(group['Ret_60d'], method='pearson')
    
    # Calculate Spearman rank correlation (Prob_Up vs Ret_60d)
    spearman = group['Prob_Up'].corr(group['Ret_60d'], method='spearman')
    
    return pd.Series({'Pearson': pearson, 'Spearman': spearman})

def calc_daily_accuracy(group):
    """
    Calculates the prediction accuracy for a single time slice (day).
    Paper definition: Predicted 'up' probability > 50% is classified as up (1).
    """
    # Robust approach: dynamically generate 0/1 prediction labels based on Prob_Up > 0.5 threshold
    calc_prediction = (group['Prob_Up'] > 0.5).astype(int)
    
    # Ensure True_Label is an integer type for comparison
    true_label = group['True_Label'].astype(int)
    
    # Calculate the proportion of correct predictions (Accuracy)
    return (true_label == calc_prediction).mean()

def calculate_decile_table_with_metrics(df):
    """
    Calculates the annualized decile performance table, including H-L long-short portfolio and Turnover.
    Input df must contain: Date, StockID, Ret_20d (for returns), Prob_Up
    """
    # 1. Data Preprocessing
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Stores decile returns: [{'Date':..., 'Decile':..., 'Return':...}, ...]
    period_stats = []
    
    # Stores holdings information for Turnover calculation: {Date: DataFrame_of_Top_Decile}
    holdings_dict = {}
    
    # Sorts by date to ensure correct order for Turnover calculation
    unique_dates = sorted(df['Date'].unique())
    
    # 2. Iterate by date (simulating periodic rebalancing)
    for date in unique_dates:
        group = df[df['Date'] == date].copy()
        
        # Skip if too few stocks to divide into 10 deciles
        if len(group) < 10:
            continue
            
        # --- Decile Grouping Logic ---
        # Use qcut to divide stocks into 10 groups (0-9)
        # Decile 0 = Low Prob, Decile 9 = High Prob
        try:
            group['Decile_Rank'] = pd.qcut(group['Prob_Up'], 10, labels=False, duplicates='drop')
        except ValueError:
            # Fallback for many ties: use rank to ensure groups are created
            group['Decile_Rank'] = pd.qcut(group['Prob_Up'].rank(method='first'), 10, labels=False)
        
        # --- Calculate Average Return for Each Group ---
        # Calculate the average return for each group (equal weight)
        decile_returns = group.groupby('Decile_Rank')['Ret_20d'].mean()
        
        for d_rank, ret in decile_returns.items():
            period_stats.append({
                'Date': date,
                'Decile': d_rank + 1,  # Convert to 1-10
                'Return': ret
            })
            
        # --- Save Decile 10 holdings for Turnover calculation ---
        # Get stocks in Decile 10 (Rank 9) and their returns for the period
        top_group = group[group['Decile_Rank'] == 9][['StockID', 'Ret_20d']]
        holdings_dict[date] = top_group
            
    # 3. Tidy data: Convert to pivot table (Index=Date, Columns=Decile)
    metrics_df = pd.DataFrame(period_stats)
    # Columns of pivot_df are 1, 2, ..., 10
    pivot_df = metrics_df.pivot(index='Date', columns='Decile', values='Return')
    
    # --- New Logic: Calculate H-L (Long-Short Strategy) ---
    # H-L = Decile 10 - Decile 1
    pivot_df['H-L'] = pivot_df[10] - pivot_df[1]
    
    # 4. Calculate Annualized Metrics (Includes Decile 1-10 and H-L)
    # Annualization factor K (Assuming 20 days per period, approx. 12 periods per year)
    K = 12
    
    ann_return = pivot_df.mean() * K
    ann_volatility = pivot_df.std() * np.sqrt(K)
    sharpe_ratio = ann_return / ann_volatility
    
    # Assemble performance table
    summary_df = pd.DataFrame({
        'Ann. Return': ann_return,
        'Ann. Volatility': ann_volatility,
        'Sharpe Ratio': sharpe_ratio
    })
    
    # --- New Logic: Calculate Turnover (Focusing on Decile 10) ---
    turnover_values = []
    sorted_dates = sorted(holdings_dict.keys())
    
    for i in range(len(sorted_dates) - 1):
        t = sorted_dates[i]
        t_plus_1 = sorted_dates[i+1]
        
        # Get Decile 10 holdings at time t and t+1
        # holdings_t includes the stock returns (Ret_20d) during the period from t to t+1
        holdings_t = holdings_dict[t].set_index('StockID')
        holdings_next = holdings_dict[t_plus_1].set_index('StockID')
        
        # Union of all stocks involved
        all_stocks = set(holdings_t.index).union(set(holdings_next.index))
        
        turnover_t = 0
        
        # Total portfolio return at time t (used for calculating drift)
        # Assuming equal weight, portfolio return at time t is the average return of Decile 10 for that period
        port_ret_t = holdings_t['Ret_20d'].mean()
        
        # Initial weight of individual stock at time t
        n_t = len(holdings_t)
        w_t_initial = 1.0 / n_t if n_t > 0 else 0
        
        # Target weight at time t+1
        n_next = len(holdings_next)
        w_next_target = 1.0 / n_next if n_next > 0 else 0
        
        for stock in all_stocks:
            # 1. Calculate drifted weight at time t (Drifted Weight)
            # w_{i,t} * (1 + r_{i,t+1}) / (1 + R_{p,t+1})
            if stock in holdings_t.index:
                r_i = holdings_t.loc[stock, 'Ret_20d']
                w_t_drifted = w_t_initial * (1 + r_i) / (1 + port_ret_t)
            else:
                w_t_drifted = 0
            
            # 2. Get target weight at time t+1
            w_next = w_next_target if stock in holdings_next.index else 0
            
            # 3. Accumulate absolute difference
            turnover_t += abs(w_next - w_t_drifted)
            
        turnover_values.append(turnover_t)
    
    # Calculate average turnover rate
    avg_turnover = np.mean(turnover_values) if turnover_values else 0
    
    return summary_df, avg_turnover

# -------------------------------------------------------
# Execution Block
# -------------------------------------------------------

print("Calculating metrics...")

# 1. Calculate Cross-sectional Correlations
# include_groups=False is used to suppress a pandas FutureWarning
daily_correlations = df.groupby('Date').apply(calc_cross_sectional_corr, include_groups=False)

# Average the daily correlations over the entire time series
avg_correlations = daily_correlations.mean()

# 2. Calculate Accuracy
daily_accuracy = df.groupby('Date').apply(calc_daily_accuracy, include_groups=False)

# Average the daily accuracy over the entire time series
final_accuracy = daily_accuracy.mean()

# 3. Calculate Decile Performance and Turnover
# Ensure StockID is available for this part of the calculation.
# NOTE: The provided example path is hardcoded and kept as is, but it might not match the main file path.
df_decile = pd.read_csv('./result_I20R20/predictions_detailed.csv')
df_decile['Date'] = pd.to_datetime(df_decile['Date']) # Ensure date is datetime for sorting
result_table, turnover = calculate_decile_table_with_metrics(df_decile)


# -------------------------------------------------------
# Output Results
# -------------------------------------------------------

print("-" * 30)
print("Paper Replication Results (Correlation & Accuracy)")
print("-" * 30)
print(f"Out-of-Sample Accuracy (Acc.): {final_accuracy:.2%}")
print(f"Avg Cross-sectional Pearson Corr.: {avg_correlations['Pearson']:.4f} (Target for Table 2)")
print(f"Avg Cross-sectional Spearman Corr.: {avg_correlations['Spearman']:.4f} (Rank Capability)")
print("-" * 30)

print("\n" + "-" * 30)
print("Decile Portfolio Performance (Ann. Metrics)")
print("-" * 30)

# Print performance table
print("--- Performance Table ---")
print(result_table)

# Print turnover rate
print(f"\nAverage Period Turnover (Decile 10): {turnover:.2%}")

# Export decile results
result_table.to_csv('annual_ret_vol_hl.csv')