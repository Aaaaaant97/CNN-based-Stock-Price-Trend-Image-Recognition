from torch.utils.data import DataLoader
import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from dataset import get_years_dataset, train_val_split
from model import CNN20
from train_utils import trainer
import argparse
import os

def plot_loss(train_loss, val_loss, num_epochs, model_name, output):
    import matplotlib.pyplot as plt
    plt.semilogy(range(num_epochs + 1), train_loss, label='Train Loss')
    plt.semilogy(range(num_epochs + 1), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(f'{output}/{model_name}_losses.png')

def model_from_config(config):
    model = CNN20(**config)
    return model

def main():
    parser = argparse.ArgumentParser(description='Test ML model and get metrics')
    parser.add_argument('-model_path', type=str, default='./model', required=True, help='Path to model file')
    parser.add_argument('-data', type=str, default='./monthly_20d', required=True, help='Path to data')
    parser.add_argument('-output', type=str, default='./output', help='Path to save predictions')
    parser.add_argument('-ckpt_path', type=str, default='./ckpt')
    parser.add_argument('-config_path', type=str, required=True, help='Path to config file, if provided, it will override the default parameters')

    parser.add_argument('-seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('-device', type=str, default='cuda', help='Device to use for testing')
    parser.add_argument('-batch_size', type=int, default=128, help='Batch size for testing')
    parser.add_argument('-num_workers', type=int, default=4, help='Number of workers for testing')
    parser.add_argument('-model_name', type=str, default='baseline_I20R5', required=True, help='Model name')
    parser.add_argument('-learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('-num_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('-tolerence', type=float, default=0, help='Tolerance for early stopping')
    parser.add_argument('-patience', type=int, default=2, help='Patience for early stopping')

    parser.add_argument('-ret_days', type=int, default=5, help='Number of days to predict')
    parser.add_argument('-year_start', type=int, default=1993, help='Year to start')
    parser.add_argument('-year_split', type=int, default=1999, help='Year to split')
    parser.add_argument('-year_end', type=int, default=2019, help='Year to end')
    parser.add_argument('-ratio', type=float, default=0.7, help='Train/validation split ratio')
    parser.add_argument('-chronological', action='store_true', help='Use chronological split for train/validation')
    args = parser.parse_args()
    seed = args.seed
    device = args.device
    if seed:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
    model_path = args.model_path
    data_path = args.data
    ckpt_path = args.ckpt_path 
    config_path = args.config_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    model_name = args.model_name
    ret_days = args.ret_days
    learning_rate = args.learning_rate
    tolerence = args.tolerence
    patience = args.patience
    num_epochs = args.num_epochs
    year_start = args.year_start
    year_split = args.year_split
    year_end = args.year_end
    ratio = args.ratio
    batch_size = args.batch_size
    num_workers = args.num_workers
    chronological = args.chronological

    if args.config_path:
        model_configs = yaml.safe_load(open(args.config_path, 'r'))
        model = model_from_config(model_configs)
    model = model_from_config(model_configs).to(device)

    train_dataset, val_dataset = train_val_split(get_years_dataset(data_path, year_start, year_split + 1, ret_days), ratio, chronological)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )   

    total_training_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_training_parameters += param.numel()
    print(f"[INFO]Number of training parameters: {total_training_parameters}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model, metrics_df = trainer(model, train_loader, val_loader, criterion, optimizer, num_epochs, device,
                                                      patience, tolerence, ckpt_path, model_name)
    torch.save(model.state_dict(), os.path.join(model_path, f'{model_name}.pth'))
    metrics_df.to_csv(f'{args.output}/{model_name}_metrics.csv', index=False)
    print(f"[INFO]Training losses saved to {args.output}/{model_name}_metrics.csv")
if __name__=="__main__":
    main()

