import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def line_plot(x, y, x_name, y_name, save_path = None):
    plt.figure(figsize=(16, 8), dpi=100)
    plt.plot(x, y)
    plt.xticks(x, rotation=90) 
    plt.locator_params(axis='x', nbins=50)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def line_plot_predict(x, y, predict_y, idx_start, x_name, y_name, save_path = None):

    plt.figure(figsize=(16, 8), dpi=100)
    plt.plot(x, y)
    plt.plot(x[idx_start:], predict_y)
    plt.axvline(x[idx_start], 0, 1, color="red", alpha=0.5)
    plt.xticks(x, rotation=90) 
    plt.locator_params(axis='x', nbins=50)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
        
def line_plot_predict_cycle(x, y, x_pred_list, y_pred_list, idx_start_list, x_name, y_name, save_path = None):

    plt.figure(figsize=(16, 8), dpi=100)
    plt.plot(x, y)
    for x_pred, y_pred, idx_start in zip(x_pred_list, y_pred_list, idx_start_list):
        num_predict_value = len(y_pred)
        plt.plot(x[idx_start:idx_start+num_predict_value], y_pred, color='red', alpha=0.7)
        # plt.axvline(x[idx_start], 0, 1, color="red", alpha=0.5)

    plt.xticks(x, rotation=90) 
    plt.locator_params(axis='x', nbins=50)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def test_line_plot():  
    root_dir = "."
    data_dir = os.path.join(root_dir, "raw_data", "tw_stock")
    data_path = os.path.join(data_dir, "0050_202201_202209.csv")
    save_plot_dir = os.path.join(root_dir, "plot", "trend")
    ma_days = [1,3,5]
    
    # load data
    print("load data")
    data = pd.read_csv(data_path)
    os.makedirs(save_plot_dir, exist_ok=True)
    
    x = data["date"]
    y = data["closing"]
    x_name = "date"
    y_name = "closing"
    
    # plot
    print("plot")
    for ma_day in ma_days:
        y_ma = y.rolling(ma_day).mean()
        filename = f"0050_ma{ma_day}.png" if ma_day!=1 else f"0050_raw.png"
        save_plot_path = os.path.join(save_plot_dir, filename)
        line_plot(x, y_ma, x_name, y_name, save_plot_path)
    
        idx_start = 120
        predict_y = y_ma[idx_start:].apply(lambda x: x + (np.random.rand()-0.5)*5)
        filename = f"0050_ma{ma_day}_predict.png" if ma_day!=1 else f"0050_raw_predict.png"
        save_plot_path = os.path.join(save_plot_dir, filename)
        line_plot_predict(x, y_ma, predict_y, idx_start, x_name, y_name, save_plot_path)
    
    
if __name__ == "__main__":
    test_line_plot()