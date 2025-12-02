"""
Compared with the baseline version, this code add the function to save detailed test results including predicted probabilities, predicted classes, true labels, dates, stock IDs, original 20-day returns, and market capitalizations to a CSV file. It can be used for further performance analysis.
"""
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

from tools import set_seed, get_current_path, save_checkpoint, load_checkpoint, \
    EarlyStopping, plot_curve, analyze_confusion_matrix, setup_logger
from dataset import CRSP20
from models.baseline import CNN_BASELINE



def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)

def train(data_path):
    # Set random seed
    set_seed(42)

    # create output path
    current_path = get_current_path()
    output_path = os.path.join(current_path, "result")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger_path = os.path.join(output_path, 'train.log')
    logger = setup_logger(log_file=logger_path, log_info = "Train")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("****************************************************start train***************************************************")
    # load data
    data_path = os.path.join(current_path, data_path)
    dataset = CRSP20(data_path)
    print("dataset length: ", dataset.len)
    train_val_ratio = 0.7
    train_dataset, val_dataset = random_split(dataset, \
        [int(dataset.len*train_val_ratio), dataset.len-int(dataset.len*train_val_ratio)], \
        generator=torch.Generator().manual_seed(42))
    print("train_dataset length: ", len(train_dataset))
    print("val_dataset length: ", len(val_dataset))
    del dataset

    train_batch_size = 128
    eval_batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)

    # def model, optimizer, loss
    cnn_baseline = CNN_BASELINE().to(device)
    cnn_baseline.apply(init_weights)
    model = cnn_baseline
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # define train parameters
    start_epoch = 0
    end_epoch = 100
    train_loss = []
    eval_loss = []
    best_loss = 1000000
    best_model_save_path = os.path.join(output_path, "best_model.pth")
    latest_model_save_path = os.path.join(output_path, "latest_model.pth")

    use_early_stopping = True
    epoch_early_stopping = 0
    patience_early_stopping = 2
    early_stopping = EarlyStopping(
        patience=patience_early_stopping,        # tolerance
    )

    if os.path.exists(latest_model_save_path):
        model, optimizer, start_epoch, train_loss, eval_loss = load_checkpoint(latest_model_save_path, model, optimizer)
    
    logger.info("******************start training****************************")
    logger.info(f"Dataset length: train_dataset, {len(train_dataset)}, len(val_dataset), {len(val_dataset)}")
    logger.info(f"model information: {model}")
    logger.info(f"optimizer information: {optimizer}")
    logger.info(f"parameters: train_batch_size = {train_batch_size}, eval_batch_size = {eval_batch_size}, start_epoch = {start_epoch}, end_epoch = {end_epoch}")
    logger.info(f"early stopping: use_early_stopping = {use_early_stopping}, epoch_early_stopping = {epoch_early_stopping}, patience_early_stopping = {patience_early_stopping}")
    

    for epoch in tqdm(range(start_epoch, end_epoch+1)):
        epoch_start = time.time()
        epoch_train_loss = 0.0
        model.train()
        for imgs, labels in tqdm(train_dataloader):
            # for imgs, labels in enumerate(data):
                imgs = imgs.to(device)
                labels = labels.to(device)
                y_pred = model(imgs)
                l = loss(y_pred, labels.long())
                
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                epoch_train_loss += l.item()

        average_train_loss = epoch_train_loss / len(train_dataloader)
        train_loss.append(average_train_loss)
        print("***************************************************start eval****************************************************")
        model.eval()
        epoch_eval_loss = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(val_dataloader):
                # for imgs, labels in enumerate(data):
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    y_pred = model(imgs)
                    l_eval = loss(y_pred, labels.long())
                    epoch_eval_loss += l_eval.item()
            average_eval_loss = epoch_eval_loss / len(val_dataloader)    
            eval_loss.append(average_eval_loss)
            if average_eval_loss < best_loss:
                best_loss = average_eval_loss
                save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, train_loss=train_loss, eval_loss=eval_loss, save_path=best_model_save_path)
        save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, train_loss=train_loss, eval_loss=eval_loss, save_path=latest_model_save_path)
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch + 1}/{end_epoch}], train avg Loss: {average_train_loss:.6f}, eval avg Loss: {average_eval_loss:.6f}, Time: {epoch_time:.2f}s")
        logger.info(f"Epoch = [{epoch + 1}/{end_epoch}], train avg Loss = {average_train_loss:.6f}, eval avg Loss = {average_eval_loss:.6f}, epoch Time = {epoch_time:.2f}s")
        # early stopping
        if epoch > epoch_early_stopping and use_early_stopping:
            early_stopping(average_eval_loss)
            if early_stopping.early_stop:
                print("Early stopping!")
                logger.info(f"Early stopping!, epoch = {epoch + 1}, eval_loss = {average_eval_loss}")
                break
    plot_curve(train_loss=train_loss, eval_loss=eval_loss, save_path=output_path)



# def test(data_path, model_path):
#     print("***********start test****************")
#     current_path = get_current_path()
#     model_name = model_path.split("/")[2].split("_")[0]
#     logger_path = os.path.join(current_path, f'./result/test_{model_name}.log')
#     logger = setup_logger(log_file=logger_path, log_info = "Test")
#     data_path = os.path.join(current_path, data_path)
#     model_path = os.path.join(current_path, model_path)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     test_dataset = CRSP20(data_path, split='test')
#     print("len(test_dataset)", len(test_dataset))
#     test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True)
#     model = CNN_BASELINE().to(device)
#     model, _, epoch, _, _ = load_checkpoint(model_path, model)
#     model.eval()
#     confusion_matrix = np.zeros((2, 2))
#     logger.info("******************start testing****************************")
#     logger.info(f"Dataset length: test_dataset, {test_dataset.len}")
#     logger.info(f"model information: {model}")
#     logger.info(f"epoch: {epoch}")

#     with torch.no_grad():
#         for imgs, labels in tqdm(test_dataloader):
#             imgs = imgs.to(device)
#             labels = labels.to(device)
#             output = model(imgs)
#             _, y_pred = torch.max(output, axis=1)
#             # prediction_correct = torch.sum(y_pred == labels).item()
#             # correct_all += prediction_correct
#             # error_all += (len(labels) - prediction_correct)
#             for i in range(len(labels)):
#                 confusion_matrix[int(labels[i].cpu().item()), int(y_pred[i].cpu().item())] += 1
#         print("confusion_matrix is", confusion_matrix)
#         result = analyze_confusion_matrix(confusion_matrix)
#         logger.info(f"confusion_matrix: {confusion_matrix}")
#         logger.info(f"analyze confusion_matrix result: {result}")

def test(data_path, model_path):
    print("***********start test****************")
    current_path = get_current_path()
    model_name = model_path.split("/")[2].split("_")[0]
    logger_path = os.path.join(current_path, f'./result/test_{model_name}.log')
    logger = setup_logger(log_file=logger_path, log_info="Test")
    data_path = os.path.join(current_path, data_path)
    model_path = os.path.join(current_path, model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = CRSP20(data_path, split='test')
    print("len(test_dataset)", len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, pin_memory=True)
    
    model = CNN_BASELINE().to(device)
    model, _, epoch, _, _ = load_checkpoint(model_path, model)
    model.eval()
    
    confusion_matrix = np.zeros((2, 2))
    logger.info("******************start testing****************************")
    
    # --- save results ---
    all_probs = []      # predicted probabilities for class 1 (up)
    all_preds = []      # predicted classes (0/1)
    all_labels = []     # true labels (0/1)
    
    all_dates = []      # dates
    all_stock_ids = []  # stock IDs
    all_raw_rets = []   # original 20d returns
    all_market_caps = []  # market capitalizations
    # ---------------------------

    with torch.no_grad():
        for imgs, labels, dates, stock_ids, raw_rets, market_caps in tqdm(test_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs)
            probs = F.softmax(output, dim=1)[:, 1]
            _, y_pred = torch.max(output, axis=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_dates.extend(dates) 
            all_stock_ids.extend(list(stock_ids))
            all_raw_rets.extend(raw_rets.numpy())
            all_market_caps.extend(market_caps.numpy())

            for i in range(len(labels)):
                confusion_matrix[int(labels[i].cpu().item()), int(y_pred[i].cpu().item())] += 1
        
        print("confusion_matrix is", confusion_matrix)
        result = analyze_confusion_matrix(confusion_matrix)
        logger.info(f"confusion_matrix: {confusion_matrix}")
        logger.info(f"analyze confusion_matrix result: {result}")

        # --- save as CSV ---
        print("Saving predictions to CSV...")
        save_file_path = os.path.join(current_path, "result", "predictions_detailed.csv")
        
        df = pd.DataFrame({
            "Date": all_dates,           
            "StockID": all_stock_ids,
            "MarketCap": all_market_caps,    
            "Ret_20d": all_raw_rets,     
            "True_Label": all_labels,    
            "Prediction": all_preds,     
            "Prob_Up": all_probs         
        })
        
        df.to_csv(save_file_path, index=False)
        print(f"Predictions saved to: {save_file_path}")

def plot_result(model_path):
    print("***********start plot curve****************")
    current_path = get_current_path()
    model_path = os.path.join(current_path, model_path)
    model = CNN_BASELINE()
    model, optimizer, start_epoch, train_loss, eval_loss = load_checkpoint(model_path, model)
    plot_curve(train_loss=train_loss, eval_loss=eval_loss, save_path=os.path.join(current_path, "result"))

    

if __name__ == "__main__":
    data_path = "./img_data/monthly_20d"
    model_path = './result/best_model.pth'
    # model_path = './result/latest_model.pth'
    # train(data_path)
    test(data_path, model_path)
    # plot_result(model_path)