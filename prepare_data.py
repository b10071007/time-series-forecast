import os
from datetime import datetime
import pandas as pd

from data import CycleMaker

#----------------------------------------------------------------------------#

def main():
    root_dir = "."
    data_dir = os.path.join(root_dir, "raw_data", "tw_stock")
    data_path = os.path.join(data_dir, "0050_202201_202209.csv")
    dst_data_dir = os.path.join(root_dir, "dataset", "tw_stock")
    
    ma_days = [1,3,5]
    
    data = pd.read_csv(data_path)
    
    x_orig = data["date"]
    x = []
    date_format = "%Y.%m.%d"
    for x_i in x_orig:
        year, month, day = x_i.split("/")
        year = int(year)+1911
        month = int(month)
        day = int(day)
        x_i_new= "{}.{}.{}".format(year, month, day)
        x.append(datetime.strptime(x_i_new, date_format))
    x = pd.Series(x)

    y = data["closing"]
    
    cycle_maker = CycleMaker()
    for ma_day in ma_days:
        filename = f"0050_cycle_ma{ma_day}.csv" if ma_day!=1 else f"0050_cycle_raw.csv"
        dst_data_path = os.path.join(dst_data_dir, filename)
        y_ma = y.rolling(ma_day).mean()
        cycle_maker.save_cycle_data(x, y_ma, 2, 2, dst_data_path)
        

    # fake
    root_dir = "."
    data_dir = os.path.join(root_dir, "raw_data")
    data_path = os.path.join(data_dir, "fake", "fake_data.csv")
    dst_data_dir = os.path.join(root_dir, "dataset", "fake")
    
    os.makedirs(dst_data_dir, exist_ok=True)
    
    ma_days = [1,3,5]
    
    data = pd.read_csv(data_path)
    
    x_orig = data["x"]
    x = []
    date_format = "%Y-%m-%d"
    for x_i in x_orig:
        x.append(datetime.strptime(x_i, date_format))
    x = pd.Series(x)

    y = data["y"]
    
    cycle_maker = CycleMaker()
    for ma_day in ma_days:
        filename = f"fake_cycle_ma{ma_day}.csv" if ma_day!=1 else f"fake_cycle_raw.csv"
        dst_data_path = os.path.join(dst_data_dir, filename)
        y_ma = y.rolling(ma_day).mean()
        cycle_maker.save_cycle_data(x, y_ma, 2, 2, dst_data_path)
    
if __name__ == "__main__":
    main()