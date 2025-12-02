import pandas as pd
import numpy as np

def calculate_performance_metrics(file_path):
    df = pd.read_csv(file_path, dtype={'Date': str})

    cols_to_numeric = ['True_Ret', 'Predicted_Ret']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=cols_to_numeric)


    def calc_group_metrics(group):
        # MSE calculation
        mse = ((group['True_Ret'] - group['Predicted_Ret']) ** 2).mean()
        # IC calculation (Pearson correlation)
        if len(group) > 1:
            corr = group['True_Ret'].corr(group['Predicted_Ret'])
        else:
            corr = np.nan
            
        return pd.Series({'MSE': mse, 'IC': corr})

    print("calculating daily metrics...")
    daily_metrics = df.groupby('Date').apply(calc_group_metrics)

    avg_metrics = daily_metrics.mean()

    print("\n" + "="*40)
    print("Performance Metrics Summary")
    print("="*40)
    print(f"Average MSE:       {avg_metrics['MSE']:.8f}")
    print(f"Average IC (Corr): {avg_metrics['IC']:.8f}")
    print("="*40)
    return daily_metrics, avg_metrics

file_path = './result_I20R20_noearlystop_ret_latest/predictions_reg_Ret_20d.csv' 

import os
if os.path.exists(file_path):
    daily_metrics, avg_metrics = calculate_performance_metrics(file_path)
else:
    print(f"Cannot find file: {file_path}, please check the path.")