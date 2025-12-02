import pandas as pd
import numpy as np

def calculate_value_weighted_portfolio(df):
    """
    calculate the value-weighted portfolio performance metrics and turnover for I20R20 strategy.   
    """
    df = df.copy()

    required_cols = ['Date', 'StockID', 'Ret_20d', 'Prob_Up', 'MarketCap']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column: {col}")
            
    df['Date'] = pd.to_datetime(df['Date'])

    period_stats = []
    holdings_dict = {}  

    all_dates = sorted(df['Date'].unique())
    rebalance_dates = all_dates # Monthly rebalancing (Step=1)
    
    print(f"Strategy: Value-Weighted (I20R20)")
    print(f"Rebalancing Frequency: Monthly (Step=1)")
    print(f"Total Rebalance Periods: {len(rebalance_dates)}")
    

    for date in rebalance_dates:
        group = df[df['Date'] == date].copy()
        
        if len(group) < 10:
            continue
            
        group['Decile_Rank'] = pd.qcut(group['Prob_Up'].rank(method='first'), 10, labels=False)
        group_total_cap = group.groupby('Decile_Rank')['MarketCap'].transform('sum')
        group['Weight'] = group['MarketCap'] / group_total_cap
        group['Weighted_Ret'] = group['Weight'] * group['Ret_20d']

        vw_ret = group.groupby('Decile_Rank')['Weighted_Ret'].sum()
        
        for d_rank, ret in vw_ret.items():
            period_stats.append({
                'Date': date,
                'Decile': d_rank + 1,
                'Return': ret
            })
            
        top_group = group[group['Decile_Rank'] == 9][['StockID', 'Ret_20d', 'Weight']]
        holdings_dict[date] = top_group

    metrics_df = pd.DataFrame(period_stats)
    if metrics_df.empty:
        return None, 0
        
    pivot_df = metrics_df.pivot(index='Date', columns='Decile', values='Return')
    pivot_df['H-L'] = pivot_df[10] - pivot_df[1]

    K = 12
    ann_return = pivot_df.mean() * K
    ann_volatility = pivot_df.std() * np.sqrt(K)
    sharpe_ratio = ann_return / ann_volatility
    
    result_table = pd.DataFrame({
        'Ann. Return': ann_return,
        'Ann. Volatility': ann_volatility,
        'Sharpe Ratio': sharpe_ratio
    })
    
    # --- Turnover Calculation ---
    turnover_values = []
    sorted_dates = sorted(holdings_dict.keys())
    
    for i in range(len(sorted_dates) - 1):
        t_current = sorted_dates[i]
        t_next = sorted_dates[i+1]
        
        holdings_t = holdings_dict[t_current].set_index('StockID')
        holdings_next = holdings_dict[t_next].set_index('StockID')
        
        all_stocks = set(holdings_t.index).union(set(holdings_next.index))
        
        turnover_t = 0
        
        port_ret_t = (holdings_t['Weight'] * holdings_t['Ret_20d']).sum()
        
        for stock in all_stocks:
            # drifted weight calculation
            if stock in holdings_t.index:
                w_t_initial = holdings_t.loc[stock, 'Weight']
                r_i = holdings_t.loc[stock, 'Ret_20d'] # [修改] 使用 Ret_20d
                w_t_drifted = w_t_initial * (1 + r_i) / (1 + port_ret_t)
            else:
                w_t_drifted = 0
            
            # next period weight
            if stock in holdings_next.index:
                w_next = holdings_next.loc[stock, 'Weight']
            else:
                w_next = 0
            
            # turnover contribution
            turnover_t += abs(w_next - w_t_drifted)
            
        turnover_values.append(turnover_t)
    
    avg_turnover = np.mean(turnover_values) if turnover_values else 0
    avg_monthly_turnover = avg_turnover / 1.0
    
    return result_table, avg_monthly_turnover

if __name__ == "__main__":
    file_path = './result_I20R20/predictions_detailed.csv' 
    
    try:
        df = pd.read_csv(file_path)
        if 'MarketCap' not in df.columns:
             print("Error: Input CSV must contain 'MarketCap' column.")
        elif 'Ret_20d' not in df.columns:
             print("Error: Input CSV must contain 'Ret_20d' column.")
        else:
            perf_table, turnover = calculate_value_weighted_portfolio(df)
            
            print("\n=== I20R20 Value-Weighted Portfolio Performance ===")
            print(perf_table)
            print("\n=== VW Turnover Analysis ===")
            print(f"Monthly Equivalent Turnover: {turnover:.2%}")
            
            perf_table.to_csv('I20R20_VW_ret_vol_hl.csv')
            
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")