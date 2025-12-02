import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  

import pandas as pd
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import numpy as np


# import Reggression model
from models.baseline_extend import CNN_REGRESSION 
from EXTdataset import CRSP20
from tools import set_seed, get_current_path, save_checkpoint, load_checkpoint, \
    EarlyStopping, plot_curve, setup_logger


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

def train(data_path, target_col="Ret_20d"):
    # Set random seed
    set_seed(42)

    # create output path
    current_path = get_current_path()
    output_path = os.path.join(current_path, "result")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger_path = os.path.join(output_path, f'train_reg_{target_col}.log')
    logger = setup_logger(log_file=logger_path, log_info=f"Train Regression ({target_col})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"******************** Start Training Regression for {target_col} ********************")
    
    data_path = os.path.join(current_path, data_path)
    dataset = CRSP20(data_path, split="train", target_col=target_col)
    
    print("Dataset length: ", dataset.len)
    train_val_ratio = 0.7
    train_dataset, val_dataset = random_split(dataset, \
        [int(dataset.len*train_val_ratio), dataset.len-int(dataset.len*train_val_ratio)], \
        generator=torch.Generator().manual_seed(42))
    print("Train dataset length: ", len(train_dataset))
    print("Val dataset length: ", len(val_dataset))
    del dataset

    train_batch_size = 128
    eval_batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)

    # def model
    # regression model with output size 1
    model = CNN_REGRESSION().to(device)
    model.apply(init_weights)
    
    # key modifications for regression: MSELoss
    loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    start_epoch = 0
    end_epoch = 100
    train_loss = []
    eval_loss = []
    best_loss = float('inf')
    best_model_save_path = os.path.join(output_path, f"best_model_reg_{target_col}.pth")
    latest_model_save_path = os.path.join(output_path, f"latest_model_reg_{target_col}.pth")

    use_early_stopping = True
    epoch_early_stopping = 0
    patience_early_stopping = 3 
    early_stopping = EarlyStopping(patience=patience_early_stopping)

    if os.path.exists(latest_model_save_path):
        model, optimizer, start_epoch, train_loss, eval_loss = load_checkpoint(latest_model_save_path, model, optimizer)
    
    logger.info("****************** Start Training ****************************")
    logger.info(f"Target: {target_col}")
    logger.info(f"Model: {model}")
    logger.info(f"Loss Function: MSELoss")

    for epoch in tqdm(range(start_epoch, end_epoch+1)):
        epoch_start = time.time()
        epoch_train_loss = 0.0
        model.train()
        
        for imgs, labels, _, _, _, _ in tqdm(train_dataloader, leave=False):
            imgs = imgs.to(device)
            
            # key modification for regression: labels float and unsqueeze
            labels = labels.float().to(device).unsqueeze(1)
            
            y_pred = model(imgs)
            
            l = loss_fn(y_pred, labels)
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            epoch_train_loss += l.item()

        average_train_loss = epoch_train_loss / len(train_dataloader)
        train_loss.append(average_train_loss)
        
        print(f"Epoch {epoch+1} Train MSE: {average_train_loss:.6f}")

        # Eval
        model.eval()
        epoch_eval_loss = 0.0
        with torch.no_grad():
            for imgs, labels, _, _, _, _ in tqdm(val_dataloader, leave=False):
                imgs = imgs.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                
                y_pred = model(imgs)
                l_eval = loss_fn(y_pred, labels)
                epoch_eval_loss += l_eval.item()
                
            average_eval_loss = epoch_eval_loss / len(val_dataloader)    
            eval_loss.append(average_eval_loss)
            
            if average_eval_loss < best_loss:
                best_loss = average_eval_loss
                save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, train_loss=train_loss, eval_loss=eval_loss, save_path=best_model_save_path)
        
        save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, train_loss=train_loss, eval_loss=eval_loss, save_path=latest_model_save_path)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch + 1}/{end_epoch}], Eval Avg MSE: {average_eval_loss:.6f}, Time: {epoch_time:.2f}s")
        logger.info(f"Epoch [{epoch + 1}], Train MSE: {average_train_loss:.6f}, Eval MSE: {average_eval_loss:.6f}")
        
        if epoch > epoch_early_stopping and use_early_stopping:
            early_stopping(average_eval_loss)
            if early_stopping.early_stop:
                print("Early stopping!")
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
                
    plot_curve(train_loss=train_loss, eval_loss=eval_loss, save_path=output_path)


def test(data_path, model_path, target_col="Ret_20d"):
    print("*********** Start Test Regression ****************")
    current_path = get_current_path()
    model_name = os.path.basename(model_path).split('.')[0]
    logger_path = os.path.join(current_path, f'./result/test_{model_name}.log')
    logger = setup_logger(log_file=logger_path, log_info="Test Regression")
    
    data_path = os.path.join(current_path, data_path)
    model_path = os.path.join(current_path, model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = CRSP20(data_path, split='test', target_col=target_col)
    print("Test dataset length:", len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True)
    model = CNN_REGRESSION().to(device)
    model, _, epoch, _, _ = load_checkpoint(model_path, model)
    model.eval()
    
    logger.info(f"Testing model: {model_name} on target: {target_col}")

    all_preds = []
    all_labels = [] # True return value
    
    all_dates = []
    all_stock_ids = []
    all_raw_rets = []
    all_market_caps = []
    
    total_squared_error = 0.0
    
    with torch.no_grad():
        for imgs, labels, dates, stock_ids, raw_rets, market_caps in tqdm(test_dataloader):
            imgs = imgs.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            y_pred = model(imgs) # output continuous value
            
            # caculate Sum Squared Error
            total_squared_error += torch.sum((y_pred - labels) ** 2).item()
            all_preds.extend(y_pred.cpu().squeeze(1).numpy())
            all_labels.extend(labels.cpu().squeeze(1).numpy())
            
            all_dates.extend(dates)
            all_stock_ids.extend(list(stock_ids))
            all_raw_rets.extend(raw_rets.numpy())
            all_market_caps.extend(market_caps.numpy())
    mse = total_squared_error / len(test_dataset)
    rmse = np.sqrt(mse)
    
    if len(all_preds) > 1:
        corr = np.corrcoef(all_labels, all_preds)[0, 1]
    else:
        corr = 0.0
        
    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Correlation (IC): {corr:.4f}")
    
    logger.info(f"Test MSE: {mse:.6f}")
    logger.info(f"Test RMSE: {rmse:.6f}")
    logger.info(f"Correlation (IC): {corr:.4f}")

    print("Saving predictions to CSV...")
    save_file_path = os.path.join(current_path, "result", f"predictions_reg_{target_col}.csv")
    
    df = pd.DataFrame({
        "Date": all_dates,
        "StockID": all_stock_ids,
        "MarketCap": all_market_caps,
        "True_Ret": all_labels,       
        "Predicted_Ret": all_preds,   
        "Raw_Ret_Ref": all_raw_rets   
    })
    
    df.to_csv(save_file_path, index=False)
    print(f"Predictions saved to: {save_file_path}")

def plot_result(model_path):
    print("***********start plot curve****************")
    current_path = get_current_path()
    model_path = os.path.join(current_path, model_path)
    model = CNN_REGRESSION()
    model, optimizer, start_epoch, train_loss, eval_loss = load_checkpoint(model_path, model)
    plot_curve(train_loss=train_loss, eval_loss=eval_loss, save_path=os.path.join(current_path, "result"))


if __name__ == "__main__":
    data_path = "./img_data/monthly_20d"
    target_column = "Ret_20d"
    model_path_train = f'./result/best_model_reg_{target_column}.pth'
    
    # train
    # train(data_path, target_col=target_column)
    
    # test
    test(data_path, model_path_train, target_col=target_column)
    
    # plot
    plot_result(model_path_train)