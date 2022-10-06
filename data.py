import os
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------------------#

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2)
        # y = np.expand_dims(y, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
    
class TimeSeriesDatasetMultiVariate(Dataset):
    def __init__(self, x, y):
        # x = np.expand_dims(x, 2)
        # y = np.expand_dims(y, 2)
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class CycleMaker():
    def __init__(self):
        pass
    
    def get_drop_point_by_rule(self, y, num_higher_left=3, num_higher_right=3):
        '''
        Get sudden drop list by simple rule which is lower than all neighbors
        
        '''
        num_y = len(y)
        drop_idx_list = []
        for idx in range(num_higher_left, num_y-num_higher_right+1):
            yi = y[idx]
            if np.argmin(y[(idx-num_higher_left):(idx+num_higher_right+1)]) == num_higher_left:
                drop_idx_list.append(idx)
        return drop_idx_list
    
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
        drop_idx_list = self.get_drop_point_by_rule(x, y, num_higher_left, num_higher_right)
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
            # if length==18:
            #     break
            data.iloc[:length, 2*idx] = cycle_x.to_numpy()
            data.iloc[:length, 2*idx+1] = cycle_y.to_numpy()
            # print(data.iloc[0, 2*idx])
            # print(data.iloc[0, 2*idx+1])
            data = data.rename(columns={2*idx: f"{length}_{cycle_idx[0]}_{cycle_idx[1]}_time", 
                                        2*idx+1: f"{length}_{cycle_idx[0]}_{cycle_idx[1]}_value"})
            
        os.makedirs(os.path.split(dst_data_path)[0], exist_ok=True)
        # print(data)
        data.to_csv(dst_data_path, index=False)
        
    def save_cycle_data_multivariate(self, time, feature, value, num_higher_left, num_higher_right, dst_data_path):
        drop_idx_list = self.get_drop_point_by_rule(value, num_higher_left, num_higher_right)
        cycle_x_list, cycle_y_list, cycle_idx_list, length_list = self.get_ts_cycle_by_drop_point(time, value, drop_idx_list)
        feature_list = []
        for cycle_indices in cycle_idx_list:
            feature_by_indices = feature[cycle_indices[0]:cycle_indices[1]+1]
            feature_list.append(feature_by_indices)
        
        num_cycle = len(cycle_x_list)
        max_length = max(length_list)
        num_feature = feature.shape[1]
        data = np.empty((num_cycle, num_feature+2, max_length))
        data[:] = np.nan
        data = data.astype(str)
        
        for idx in range(num_cycle):
            cycle_x = cycle_x_list[idx]
            cycle_y = cycle_y_list[idx]
            feature = feature_list[idx]
            length = len(cycle_x)
            
            merge_data = np.concatenate([cycle_x, cycle_y.astype(object), feature.astype(object)], axis=1)
            merge_data = merge_data.T
            data[idx, :, :length] = merge_data

        # print(data.shape)
        os.makedirs(os.path.split(dst_data_path)[0], exist_ok=True)
        np.save(dst_data_path, data)

def convert_to_train_data(value, num_input=2, num_output=2):
    if type(value) is pd.Series:
        value = value.to_numpy()
    
    num_value = len(value)
    x = []
    y = []
    for idx in range(num_value-num_input-num_output+1):
        x.append(value[idx:idx+num_input])
        y.append(value[idx+num_input:idx+num_input+num_output])
    x = np.array(x)
    y = np.array(y)
    
    return x, y

def convert_to_train_data_with_feature(feature, value, num_input=2, num_output=2):
    if type(value) is pd.Series:
        value = value.to_numpy()
    
    num_value = len(value)
    x = []
    y = []
    
    
    for idx in range(num_value-num_input-num_output+1):
        x.append(np.stack([value[idx:idx+num_input], feature[idx:idx+num_input]]))
        y.append(value[idx+num_input:idx+num_input+num_output])
    x = np.array(x)
    y = np.array(y)
    
    return x, y
    

def prepare_cycle_data(data, num_input=2, num_output=2, train_ratio=0.7):
    num_cycle = data.shape[1]//2
    min_length = num_input + num_output
    
    idx_list_select = []
    num_cycle_select = 0
    for idx in range(num_cycle):
        length, start_idx, end_idx = data.columns[2*idx].split("_")[:-1]
        length = int(length)
        # if idx==35:
        #     break
        if length>=min_length:
            idx_list_select.append(idx)
            num_cycle_select += 1
            
    idx_split = round(num_cycle_select*train_ratio)
    
    # train
    train_x = []
    train_y = []
    for idx in idx_list_select[:idx_split]:
        length, start_idx, end_idx = data.columns[2*idx].split("_")[:-1]
        length = int(length)
        # date = data.iloc[:length,2*idx]
        value = data.iloc[:length,2*idx+1]
        x, y = convert_to_train_data(value, num_input, num_output)
        train_x.append(x)
        train_y.append(y)
        # if i == 2:
        #     break

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)

    # val
    val_x = []
    val_y = []
    for idx in idx_list_select[idx_split:]:
        length, start_idx, end_idx = data.columns[2*idx].split("_")[:-1]
        length = int(length)
        # date = data.iloc[:length,2*idx]
        value = data.iloc[:length,2*idx+1]
        x, y = convert_to_train_data(value, num_input, num_output)
        val_x.append(x)
        val_y.append(y)
        # if idx==35:
        #     break
    val_x = np.concatenate(val_x) 
    val_y = np.concatenate(val_y)
    
    return train_x, train_y, val_x, val_y

def prepare_cycle_data_with_feature(data, num_input=2, num_output=2, train_ratio=0.7):
    num_cycle = data.shape[0]
    min_length = num_input + num_output
    
    idx_list_select = []
    num_cycle_select = 0
    for idx in range(num_cycle):
        length = np.sum(data[idx,0]!="nan")
        if length>=min_length:
            idx_list_select.append(idx)
            num_cycle_select += 1
            
    idx_split = round(num_cycle_select*train_ratio)
    
    # train
    train_x = []
    train_y = []
    for idx in idx_list_select[:idx_split]:
        length = np.sum(data[idx,0]!="nan")
        time, feature, value  = data[idx, :, :length]
        feature = feature.astype(float)
        value = value.astype(float)
        x, y = convert_to_train_data_with_feature(feature, value, num_input, num_output)
        train_x.append(x)
        train_y.append(y)
        # if i == 2:
        #     break

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)

    # val
    val_x = []
    val_y = []
    for idx in idx_list_select[idx_split:]:
        length = np.sum(data[idx,0]!="nan")
        time, feature, value  = data[idx, :, :length]
        feature = feature.astype(float)
        value = value.astype(float)
        x, y = convert_to_train_data_with_feature(feature, value, num_input, num_output)
        val_x.append(x)
        val_y.append(y)
        # if i == 2:
        #     break
    val_x = np.concatenate(val_x) 
    val_y = np.concatenate(val_y)
    
    return train_x, train_y, val_x, val_y

def test_TimeSeriesDataset():
    root_dir = "."
    data_path = os.path.join(root_dir, "dataset", "tw_stock", "0050_cycle_raw.csv")
    data = pd.read_csv(data_path)
    train_x, train_y, val_x, val_y = prepare_cycle_data(data, num_input=2, num_output=2, train_ratio=0.7)
    
    train_dataset = TimeSeriesDataset(train_x, train_y)
    x, y = train_dataset[0:5]
    
    
    

#------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    test_TimeSeriesDataset()