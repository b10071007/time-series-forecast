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

def test_line_plot():  
    root_dir = "."
    data_dir = os.path.join(root_dir, "raw_data", "tw_stock")
    data_path = os.path.join(data_dir, "0050_202201_202209.csv")
    save_plot_dir = os.path.join(root_dir, "plot")
    data = pd.read_csv(data_path)
    data.to_csv(data_path, index=False)
    x = data["date"]
    y = data["closing"]
    x_name = "date"
    y_name = "closing"
    
    os.makedirs(save_plot_dir, exist_ok=True)
    save_plot_path = os.path.join(save_plot_dir, "0050.png")
    line_plot(x, y, x_name, y_name, save_plot_path)
    
    idx_start = 120
    predict_y = y[idx_start:].apply(lambda x: x + (np.random.rand()-0.5)*5)
    save_plot_path = os.path.join(save_plot_dir, "0050_predict.png")
    line_plot_predict(x, y, predict_y, idx_start, x_name, y_name, save_plot_path)
    
    
if __name__ == "__main__":
    test_line_plot()