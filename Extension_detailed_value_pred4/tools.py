import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

import logging

# ---------------------- Logger configuration ----------------------
def setup_logger(log_file, log_info):
    # creat logger
    logger = logging.getLogger(log_info)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Disable default output (to avoid printing to the console)

    log_dir = os.path.dirname(log_file)
    if log_dir != "" and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Only configure file output (remove console processor)
    file_handler = logging.FileHandler(
        filename=log_file,
        mode="a",  # "w" Overwrite old logs, "a" Add old logs
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    # File log format: time - level - file name - line number  -content
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Add file processor to logger (without console processor)
    if not logger.handlers: 
        logger.addHandler(file_handler)
    
    return logger

# # Set random seed
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 单GPU
        torch.cuda.manual_seed_all(seed)  # 多GPU（如DataParallel）
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_current_path():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    return script_dir


def save_checkpoint(model, optimizer, epoch, train_loss, eval_loss, save_path):
    """
    Save the complete training parameters of NeRF
    Args:
        model: model
        optimizer: optimizer
        epoch: current epoch
        train_loss: average train loss for each epoch
        eval_loss: average eval loss for each epoch
        save_path: model save path
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch+1,
        "train_loss": train_loss,
        "eval_loss": eval_loss
    }
    
    torch.save(checkpoint, save_path)
    # print(f"checkpoint has saved to {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load training status (supports breakpoint continuation)
    Args:
        checkpoint_path: checkpoint path
        model: The model with parameters to be loaded (structure must be consistent with when saved)
        optimizer: Optimizer for loading parameters 
        lr_scheduler: Learning rate scheduler for loading parameters
    Returns:
        checkpoint: The complete state dictionary after loading
    """
    # load checkpoint
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    model_state_dict = checkpoint["model_state_dict"]
    # Remove the prefix 'module.' from parameter names
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, strict=False)  # strict=False
    model.to(map_location)
    print(f"The model parameters have been loaded onto the device:{map_location}")
    
    # load optimizer
    if optimizer is not None and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer parameters loaded")
    else:
        print("no optimizer parameters loaded")
    
    start_epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    eval_loss = checkpoint['eval_loss']
    return model, optimizer, start_epoch, train_loss, eval_loss

def plot_curve(train_loss, eval_loss, save_path):
    # plot train_loss
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    
    # plot train_loss
    plt.plot(epochs, train_loss, color="#2E86AB", linewidth=2.5, marker='o', markersize=4, label='Average train Loss')
    plt.plot(epochs, eval_loss, color='#2EAB49', linewidth=2.5, marker='o', markersize=4, label='Average eval Loss')
    
    # add information
    plt.xlabel('Epoch', fontsize=12)  # x
    plt.ylabel('Average CrossEntropyLoss Loss', fontsize=12)  # y
    plt.title('CNN baseline Training Curve', fontsize=14, pad=20)  # title
    plt.legend(fontsize=10)  # legend
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.xticks(range(0, len(epochs)+1, 2), fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    
    loss_path = os.path.join(save_path, "train_curve.png")
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_confusion_matrix(conf_matrix):
    """
    conf_matrix: 2x2 numpy array, 行=真实类别, 列=预测类别
    [[TN, FP],
     [FN, TP]]
    """
    TN, FP = conf_matrix[0]
    FN, TP = conf_matrix[1]

    # 数量
    print(f"TP = {TP}, TN = {TN}, FP = {FP}, FN = {FN}")

    # Accuracy
    accuracy = (TP + TN) / conf_matrix.sum()
    # Precision (对正类)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    # Recall (对正类)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    print(f"Accuracy = {accuracy:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")
    print(f"F1 Score = {f1:.4f}")

    # 比例矩阵
    conf_matrix_ratio = conf_matrix / conf_matrix.sum()
    print("Confusion matrix (proportion):")
    print(conf_matrix_ratio)

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Confusion_matrix_ratio": conf_matrix_ratio
    }


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            # torch.save(model.state_dict(), self.path)
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            # torch.save(model.state_dict(), self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True