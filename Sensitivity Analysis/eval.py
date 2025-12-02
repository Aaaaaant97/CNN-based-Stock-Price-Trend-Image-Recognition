import torch
from torch.utils.data import DataLoader, Dataset
from dataset import get_years_dataset, train_val_split
from utils import convert_to_python_types
import yaml
from model import CNN20
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F 
import os
import numpy as np
import pandas as pd
import argparse
import yaml
from scipy.stats import pearsonr, spearmanr
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}   


def load_model(model_path, model_name, device, config_path='./config.yaml'):
    config = yaml.safe_load(open(config_path, 'r'))
    model = CNN20(**config)
    model.load_state_dict(torch.load(os.path.join(model_path, f"{model_name}.pth"), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def load_dataset(dir, start=2000, end=2020, ret_days=5, batch_size=128, num_workers=4):
    test_dataset = get_years_dataset(dir, start, end, ret_days)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )  
    return test_loader

def get_pred_metrics(model, test_loader, device):
    TP = TN = FP = FN = 0
    all_predictions = []
    loss = 0
    for i, (images, labels) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
        loss += nn.CrossEntropyLoss()(outputs, labels).item() * images.shape[0]
        predictions = F.softmax(outputs, dim=1)
        all_predictions.append(predictions.cpu().numpy()) 
        _, pred = torch.max(predictions, dim=1)
        TP += ((pred == labels) & (labels == 1)).sum()
        TN += ((pred == labels) & (labels == 0)).sum()
        FP += ((pred != labels) & (labels == 0)).sum()
        FN += ((pred != labels) & (labels == 1)).sum()
    final_predictions = np.concatenate(all_predictions, axis=0)
    TOT = TP + TN + FP + FN
    accuracy = (TP + TN) / TOT
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "loss": loss / len(test_loader.dataset), "TP": TP, "TN": TN, "FP": FP, "FN": FN}
    return final_predictions, convert_to_python_types(metrics)

def get_labels_df(dir, start=2000, end=2020, ret_days=5):
    dfs = []
    with torch.no_grad():
        for year in range(start, end):
            labels_path = os.path.join(dir, f'20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather')
            labels_df = pd.read_feather(labels_path)
            labels = labels_df[f'Ret_{ret_days}d']
            missing = labels.isna()
           #print(f"Year {year}: Found {missing.sum()} missing labels.")
            filtered_df = labels_df[~missing].copy()
            if 'year' not in filtered_df.columns:
                filtered_df['year'] = year
            dfs.append(filtered_df)
            #print(f"Year {year}: Added {len(filtered_df)} records with non-missing return values.")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Successfully created combined DataFrame with {len(df)} total records")
    return df

def get_returns(df, final_predictions, ret_days=5, till_date=None, risk_free_rate=0.02):
    df_with_pred = df.copy()
    df_with_pred[f'pred_{ret_days}d'] = final_predictions[:, 1]
    df_with_pred['StockID'] = df_with_pred['StockID'].astype('category')
    portfolio_returns = []
    annualization_factor = 252 / ret_days
    for date, date_group in df_with_pred.groupby('Date'):
        #print(f"Date: {date}")
        #print(date_group[['StockID', f'Ret_{ret_days}d', f'pred_{ret_days}d']].head(10))
        date_group['decile'] = pd.qcut(date_group[f'pred_{ret_days}d'], 10, labels=False) + 1
        value_weighted_returns = {}
        for decile, decile_group in date_group.groupby('decile'):
                total_market_cap = decile_group['MarketCap'].abs().sum()
                decile_group['weight'] = decile_group['MarketCap'].abs() / total_market_cap
                value_weighted_returns[decile] = (decile_group[f"Ret_{ret_days}d"] * decile_group['weight']).sum()
        value_decile_returns = pd.Series(value_weighted_returns)
        decile_returns = date_group.groupby('decile')[f'Ret_{ret_days}d'].mean()
        hl_return = decile_returns.iloc[9] - decile_returns.iloc[0]
        value_hl_return = value_decile_returns.get(10, 0) - value_decile_returns.get(1, 0)

        result = {'Date': date}
        for i in range(1, 11):
            result[f'Equal-Weight_D{i}'] = decile_returns.get(i, np.nan)
            result[f'Value-Weight_D{i}'] = value_decile_returns.get(i, np.nan)
        result['Equal-Weight_H-L'] = hl_return
        result['Value-Weight_H-L'] = value_hl_return
        portfolio_returns.append(result)
    portfolio_returns = pd.DataFrame(portfolio_returns)  
    avg_returns = {col: portfolio_returns[col].mean() for col in portfolio_returns.columns if col != 'Date'} # returns per periodic
    volatility = {col: portfolio_returns[col].std() * np.sqrt(annualization_factor) for col in portfolio_returns.columns if col != 'Date'} # annualized volatility 
    avg_annualized_returns = {col: (1 + ret) ** annualization_factor - 1 for col, ret in avg_returns.items()} # annualized returns
    annualized_sharpe_ratio = {f'Sharpe_Ratio_{col}': (avg_annualized_returns[col] - risk_free_rate) / volatility[col] for col in portfolio_returns.columns if col != 'Date'} # annualized sharpe ratio 
    return convert_to_python_types(avg_annualized_returns), convert_to_python_types(annualized_sharpe_ratio)

def main():
    parser = argparse.ArgumentParser(description='Test ML model and get metrics')
    parser.add_argument('-model_path', type=str, default='./model', required=True, help='Path to model file')
    parser.add_argument('-config_path', type=str, default='./config.yaml', required=True, help='Path to config file')
    parser.add_argument('-model_name', type=str, default='baseline_I20R5', required=True, help='Model name')
    parser.add_argument('-data', type=str, default='./monthly_20d', required=True, help='Path to test data')
    parser.add_argument('-output', type=str, default='./output', help='Path to save predictions')
    parser.add_argument('-metric', type=str, choices=['None', 'all'], 
                        default='all', help='Metric to calculate')
    parser.add_argument('-ret_days', type=int, default=5, help='Number of days to predict')
    parser.add_argument('-device', type=str, default='cuda', help='Device to use for testing')
    parser.add_argument('-batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('-num_workers', type=int, default=4, help='Number of workers for testing')
    parser.add_argument('-year_split', type=int, default=1999, help='Year to start testing data')
    parser.add_argument('-year_end', type=int, default=2019, help='Year to end testing data')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    model = load_model(args.model_path, args.model_name, args.device, args.config_path)
    test_loader = load_dataset(args.data, args.year_split+1, args.year_end+1, args.ret_days, args.batch_size, args.num_workers)
    final_predictions, metrics = get_pred_metrics(model, test_loader, args.device)
    df = get_labels_df(args.data, args.year_split+1, args.year_end+1, args.ret_days)
    
    
    avg_annualized_returns, sharpe_ratio = get_returns(df, final_predictions, args.ret_days)
    with open(os.path.join(args.output, f'{args.model_name}.yaml'), 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
        #yaml.dump(avg_returns, f, default_flow_style=False)
        yaml.dump(sharpe_ratio, f, default_flow_style=False)
        yaml.dump(avg_annualized_returns, f, default_flow_style=False)
    
if __name__ == "__main__":
    main()



