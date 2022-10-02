import os
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------------------#

class CycleMaker():
    def __init__(self):
        pass
    
    def get_drop_point_by_rule(self, x, y, num_higher_left=3, num_higher_right=3):
        '''
        Get sudden drop list by simple rule which is lower than all neighbors
        
        '''
        num_y = len(y)
        drop_idx_list = []
        drop_x_list = []
        for idx in range(num_higher_left, num_y-num_higher_right+1):
            yi = y[idx]
            if np.argmin(y[(idx-num_higher_left):(idx+num_higher_right+1)]) == num_higher_left:
                drop_idx_list.append(idx)
                drop_x_list.append(x[idx])
        return drop_idx_list, drop_x_list
    
    def get_ts_cycle_by_drop_point(self, x, y, drop_idx_list):
        '''
        Collect data into the same cycle until maximum exists
        
        '''
        num_drop_point = len(drop_idx_list)
        cycle_x_list = []
        cycle_y_list = []
        cycle_idx_list = []
        length_list = []
        for idx in range(num_drop_point-1):
            x_temp = x[drop_idx_list[idx]:drop_idx_list[idx+1]]
            y_temp = y[drop_idx_list[idx]:drop_idx_list[idx+1]]
            idx_max = np.argmax(y_temp)
            idx_end_overall = drop_idx_list[idx] + np.argmax(y_temp)
               
            cycle_x_list.append(x_temp[:idx_max+1])
            cycle_y_list.append(y_temp[:idx_max+1])
            cycle_idx_list.append([drop_idx_list[idx], idx_end_overall])
            length_list.append(idx_max+1)
        
        return cycle_x_list, cycle_y_list, cycle_idx_list, length_list
    
    def save_cycle_data(self, x, y, num_higher_left, num_higher_right, dst_data_path):
        drop_idx_list, drop_x_list = self.get_drop_point_by_rule(x, y, num_higher_left, num_higher_right)
        cycle_x_list, cycle_y_list, cycle_idx_list, length_list = self.get_ts_cycle_by_drop_point(x, y, drop_idx_list)
        
        num_cycle = len(cycle_x_list)
        max_length = max(length_list)
        data = np.empty((max_length, num_cycle*2))
        data[:] = np.nan
        data = pd.DataFrame(data)
        
        for idx in range(num_cycle):
            cycle_x = cycle_x_list[idx]
            cycle_y = cycle_y_list[idx]
            cycle_idx = cycle_idx_list[idx]
            length = length_list[idx]
            data.iloc[:length, 2*idx] = cycle_x
            data.iloc[:length, 2*idx+1] = cycle_y
            data = data.rename(columns={2*idx: f"{length}_{cycle_idx[0]}_{cycle_idx[1]}_time", 
                                        2*idx+1: f"{length}_{cycle_idx[0]}_{cycle_idx[1]}_value"})
            
        os.makedirs(os.path.split(dst_data_path)[0], exist_ok=True)
        data.to_csv(dst_data_path, index=False)

def test_CycleMaker():
    root_dir = "."
    data_dir = os.path.join(root_dir, "raw_data", "tw_stock")
    data_path = os.path.join(data_dir, "0050_202201_202209.csv")
    dst_data_path = os.path.join(root_dir, "dataset", "0050_cycle.csv")
    data = pd.read_csv(data_path)
    data.to_csv(data_path, index=False)
    x = data["date"]
    y = data["closing"]
    
    cycle_maker = CycleMaker()
    cycle_maker.save_cycle_data(x, y, 3, 3, dst_data_path)

#------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    test_CycleMaker()