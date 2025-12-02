# Project Overview

This repository contains scripts for model prediction, portfolio backtesting, and performance visualization. Below is a description of the key files and their functionalities.

## File Descriptions

### Core & Data

* **`main.py`**
    Compared to the baseline, this script now includes functionality to **save prediction results as CSV files**. This update facilitates downstream model performance evaluation.
* **`dataset.py`**
    Updated to include a **Market Capitalization** column. This addition enables weight allocation for stocks when using Value-Weighted grouping strategies.
* **`models/baseline.py`** and **`tools.py`**
    Same as in the baseline part.

### Metrics & Evaluation

* **`corr.py`**
    Calculates the **average prediction accuracy** and the **average Pearson correlation coefficient**.

### Backtesting (Returns & Volatility)

* **`annual_ret_vol_20.py`**
    Calculates annualized returns, annualized volatility, and Sharpe Ratios using **Equal Weight** allocation.
* **`annual_ret_vol_20_Value.py`**
    Calculates annualized returns, annualized volatility, and Sharpe Ratios using **Value Weight** allocation.

### Visualization

* **`plot_results.py`**
    Generates plots to visualize the results, specifically the grouped annualized returns, volatility, and Sharpe Ratios.