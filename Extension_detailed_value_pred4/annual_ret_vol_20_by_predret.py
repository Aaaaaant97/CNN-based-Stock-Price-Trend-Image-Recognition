import pandas as pd
import numpy as np

def calculate_decile_table_with_metrics(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    period_stats = []

    holdings_dict = {}

    unique_dates = sorted(df['Date'].unique())

    for date in unique_dates:
        group = df[df['Date'] == date].copy()

        if len(group) < 10:
            continue
            
        # groupby 10 groups based on Predicted_Ret
        try:
            group['Decile_Rank'] = pd.qcut(group['Predicted_Ret'], 10, labels=False, duplicates='drop')
        except ValueError:
            group['Decile_Rank'] = pd.qcut(group['Predicted_Ret'].rank(method='first'), 10, labels=False)

        decile_returns = group.groupby('Decile_Rank')['True_Ret'].mean()
        
        for d_rank, ret in decile_returns.items():
            period_stats.append({
                'Date': date,
                'Decile': d_rank + 1, 
                'Return': ret
            })

        top_group = group[group['Decile_Rank'] == 9][['StockID', 'True_Ret']]
        holdings_dict[date] = top_group
            

    metrics_df = pd.DataFrame(period_stats)
    pivot_df = metrics_df.pivot(index='Date', columns='Decile', values='Return')
    
    pivot_df['H-L'] = pivot_df[10] - pivot_df[1]

    K = 12
    
    ann_return = pivot_df.mean() * K
    ann_volatility = pivot_df.std() * np.sqrt(K)
    sharpe_ratio = ann_return / ann_volatility
    
    summary_df = pd.DataFrame({
        'Ann. Return': ann_return,
        'Ann. Volatility': ann_volatility,
        'Sharpe Ratio': sharpe_ratio
    })
    
    turnover_values = []
    sorted_dates = sorted(holdings_dict.keys())
    
    for i in range(len(sorted_dates) - 1):
        t = sorted_dates[i]
        t_plus_1 = sorted_dates[i+1]
        holdings_t = holdings_dict[t].set_index('StockID')
        holdings_next = holdings_dict[t_plus_1].set_index('StockID')
        all_stocks = set(holdings_t.index).union(set(holdings_next.index))
        
        turnover_t = 0
        
        port_ret_t = holdings_t['True_Ret'].mean()
        
        n_t = len(holdings_t)
        w_t_initial = 1.0 / n_t if n_t > 0 else 0
        
        n_next = len(holdings_next)
        w_next_target = 1.0 / n_next if n_next > 0 else 0
        
        for stock in all_stocks:
            if stock in holdings_t.index:
                r_i = holdings_t.loc[stock, 'True_Ret']
                w_t_drifted = w_t_initial * (1 + r_i) / (1 + port_ret_t)
            else:
                w_t_drifted = 0
            w_next = w_next_target if stock in holdings_next.index else 0
            turnover_t += abs(w_next - w_t_drifted)
            
        turnover_values.append(turnover_t)
    avg_turnover = np.mean(turnover_values) if turnover_values else 0
    
    return summary_df, avg_turnover

df = pd.read_csv('./result_I20R20_noearlystop_ret/predictions_reg_Ret_20d.csv')

result_table, turnover = calculate_decile_table_with_metrics(df)

print("--- Performance Table ---")
print(result_table)

print(f"\nAverage Monthly Turnover (Decile 10): {turnover:.2%}")

result_table.to_csv('annual_ret_byret.csv')