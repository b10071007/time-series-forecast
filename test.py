import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch

from model import Encoder, Decoder, Seq2Seq
from data import CycleMaker
from plot import line_plot, line_plot_predict_cycle
# from data import TimeSeriesDataset, prepare_cycle_data

def get_mae(y, y_pred):
    return np.mean(np.abs(y-y_pred))

# def mape(y, y_pred):
#     return np.mean(np.abs((y-y_pred)/y))*100
    
def test_long_future(model, value, num_input, num_output):
    '''
    only see the first "num_input" data to predict long future based on predicted value
    '''
    num_value = len(value)
    num_predict_value = num_value - num_input
    num_predict_times = int(np.ceil(num_predict_value / num_output))
    outputs = np.array([])
    with torch.no_grad():
        src = value[:num_input]
        for idx in range(num_predict_times):

            # print("input: ", src)
            src = src.unsqueeze(0).unsqueeze(2)
            output = model.predict(src, num_output)
            # print(" - output: ", output)
            src = output.squeeze(0)
            
            outputs = np.append(outputs, output.detach().cpu().numpy())
        # outputs = np.array(outputs)
        outputs = outputs[:num_predict_value]
    return outputs

def test_short_future(model, value, num_input, num_output):
    '''
    iteratively see "num_input" data to predict short future based on real data
    '''
    num_value = len(value)
    num_predict_value = num_value - num_input
    num_predict_times = int(np.ceil(num_predict_value / num_output))
    outputs = np.array([])
    with torch.no_grad():
        for idx in range(num_predict_times):
            src = value[idx:idx+num_input]
            # print("input: ", src)
            src = src.unsqueeze(0).unsqueeze(2)
            output = model.predict(src, num_output)
            # print(" - output: ", output)
            
            outputs = np.append(outputs, output.detach().cpu().numpy())
        # outputs = np.array(outputs)
        outputs = outputs[:num_predict_value]
    return outputs

def main():
    
    # data 
    root_dir = "."
    data_type = "fake" # "tw_stock" or "fake"
    data_dir = os.path.join(root_dir, "raw_data", data_type)
    if data_type == "tw_stock":
        data_path = os.path.join(data_dir, "0050_202201_202209.csv")
    elif data_type == "fake":
        data_path = os.path.join(data_dir, "fake_data.csv")
    model_path = os.path.join(root_dir, 'best_model.pt') # "last_model.pt" or "best_model.pt"
    save_plot_dir = os.path.join(root_dir, "plot", data_type, "predict")
        
    ma_day = 3
    num_input=2
    num_output=2
    
    # model
    INPUT_DIM = 1
    OUTPUT_DIM = 1
    ENC_EMB_DIM = 8
    DEC_EMB_DIM = 8
    HID_DIM = 16
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(enc, dec, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # load data
    data = pd.read_csv(data_path)
    
    # for stock
    if data_type == "tw_stock":
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
        
    # for fake
    elif data_type == "fake":
        x_orig = data["x"]
        x = []
        date_format = "%Y-%m-%d"
        for x_i in x_orig:
            x.append(datetime.strptime(x_i, date_format))
        x = pd.Series(x)

        y = data["y"]
    
    cycle_maker = CycleMaker()
    y_ma = y.rolling(ma_day).mean()
    
    drop_idx_list, drop_x_list = cycle_maker.get_drop_point_by_rule(x, y_ma, num_higher_left=2, num_higher_right=2)
    
    x = x.to_numpy()
    y_ma = y_ma.to_numpy().astype(np.float32)
    num_data = len(drop_idx_list)-1
    
    mae_list_long = []
    mae_list_short = []
    x_pred_list = []
    y_pred_list_long = []
    y_pred_list_short = []
    idx_start_list = []
    for i in range(num_data-1):
        print(f"[{i+1}/{num_data}]")
        drop_idx_start = drop_idx_list[i]
        drop_idx_end = drop_idx_list[i+1]
        x_by_drop_idx = x[drop_idx_start:drop_idx_end]
        y_by_drop_idx = y_ma[drop_idx_start:drop_idx_end]
        
        y_tensor = torch.tensor(y_by_drop_idx).to(device)
        outputs_long = test_long_future(model, y_tensor, num_input, num_output)
        outputs_short = test_short_future(model, y_tensor, num_input, num_output)
        
        y_np = y_tensor[num_input:].detach().cpu().numpy()
        # print("y_tensor: ", y)
        # print(" - outputs: ", outputs)
        mae_long = get_mae(y_np, outputs_long)
        mae_short = get_mae(y_np, outputs_short)
        mae_list_long.append(mae_long)
        mae_list_short.append(mae_short)
        # print(mae)

        # prepare info for plot
        x_pred_list.append(x_by_drop_idx)
        y_pred_list_long.append(outputs_long)
        y_pred_list_short.append(outputs_short)
        idx_start_list.append(drop_idx_start+num_input)
        

    os.makedirs(save_plot_dir, exist_ok=True)
    x_name = "x"
    y_name = "y"
    
    # save_plot_path = os.path.join(save_plot_dir, "ground_truth.png")
    # line_plot(x, y_ma, x_name, y_name, save_plot_path)
    
    # long result
    print(f"MAE_long={np.mean(mae_list_long)}")
    save_plot_path = os.path.join(save_plot_dir, "predict_long.png")
    line_plot_predict_cycle(x, y_ma, x_pred_list, y_pred_list_long, idx_start_list, x_name, y_name, save_plot_path)
    

    # short result
    print(f"MAE_short={np.mean(mae_list_short)}")
    save_plot_path = os.path.join(save_plot_dir, "predict_short.png")
    line_plot_predict_cycle(x, y_ma, x_pred_list, y_pred_list_short, idx_start_list, x_name, y_name, save_plot_path)

    print("Done")

if __name__ == "__main__":
    main()