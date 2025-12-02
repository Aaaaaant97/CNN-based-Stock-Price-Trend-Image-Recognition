# Project Overview

This repository contains scripts for model prediction, portfolio backtesting, and performance visualization. Below is a description of the key files and their functionalities.

## File Descriptions

### Core & Data

Use regression models to switch from binary prediction to predicting detailed returns.

* **`main_extend.py`**
* **`EXTdataset.py`**
* **`models/baseline_extend_py`**
* **`tools.py`**


### Metrics & Evaluation

* **`corr_byret.py`**
    Calculates the **Mean Squared Error** and the **average Pearson correlation coefficient**.

### Backtesting (Returns & Volatility)

* **`annual_ret_vol_20_by_predret.py`**
    Calculates annualized returns, annualized volatility, and Sharpe Ratios using **Equal Weight** allocation.
* **`annual_ret_vol_20_Value_by_predret.py`**
    Calculates annualized returns, annualized volatility, and Sharpe Ratios using **Value Weight** allocation.
