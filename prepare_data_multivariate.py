import os
from datetime import datetime
import numpy as np
import pandas as pd

from data import CycleMaker

#----------------------------------------------------------------------------#

def main():
    # fake
    root_dir = "."
    data_type = "fake" # "tw_stock" or "fake"
    data_dir = os.path.join(root_dir, "raw_data", data_type)
    data_path = os.path.join(data_dir, "fake_data.csv")
    dst_data_dir = os.path.join(root_dir, "dataset", data_type)

    ma_days = [3,5]
    num_higher_left=2
    num_higher_right=2
    
    #----------------------------------------------------------------------------#
    
    os.makedirs(dst_data_dir, exist_ok=True)
    
    data = pd.read_csv(data_path)
    
    x_orig = data["x"]
    x = x_orig
    # x = []
    # date_format = "%Y-%m-%d"
    # for x_i in x_orig:
    #     x.append(datetime.strptime(x_i, date_format))
    # x = pd.Series(x)
    
    y = data["y"]
    
    # for merge feature
    time = x.to_numpy().reshape(len(x), 1)
    feature = y.to_numpy().reshape(len(y), 1)
    
    cycle_maker = CycleMaker()
    for ma_day in ma_days:
        file_name = f"{data_type}_cycle_ma{ma_day}_with_feature.npy"
        dst_data_path = os.path.join(dst_data_dir, file_name)
        y_ma = y.rolling(ma_day).mean()
        y_ma = y_ma.to_numpy().reshape(len(y_ma), 1)
        cycle_maker.save_cycle_data_multivariate(time, feature, y_ma, num_higher_left, num_higher_right, dst_data_path)

if __name__ == "__main__":
    main()