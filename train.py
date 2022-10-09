import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Encoder, Decoder, Seq2Seq, init_weights
from data import TimeSeriesDataset, prepare_cycle_data

def train(model, data_loader, optimizer, criterion, clip, device):
    
    model.train()
    
    epoch_loss = 0
    
    for i, (x, y) in enumerate(data_loader):
        
        src = x.to(device)
        trg = y.to(device)
        # print("src: ", src)
        # print("trg: ", src)
        optimizer.zero_grad()
        
        output = model(src, trg)
        # print("output: ", output)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        # output = output[1:].view(-1, output_dim)
        # trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(data_loader)

def validate(model, data_loader, criterion, device):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, (x, y) in enumerate(data_loader):

            src = x.to(device)
            trg = y.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            # output = output[1:].view(-1, output_dim)
            # trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(data_loader)

# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs

def main():
    
    root_dir = "."
    data_type = "fake" # "tw_stock" or "fake"
    data_dir = os.path.join(root_dir, "dataset", data_type)
    if data_type == "tw_stock":
        data_path = os.path.join(data_dir, "0050_cycle_ma3.csv")
    elif data_type == "fake":
        data_path = os.path.join(data_dir, "fake_cycle_ma3.csv")
    exp_dir = os.path.join(root_dir, "exp", data_type, "univariate_20221009")
        
    num_input=2
    num_output=2
    train_ratio=0.7
    
    print("Training data: ", data_path)
    data = pd.read_csv(data_path)
    train_x, train_y, val_x, val_y = prepare_cycle_data(data, num_input, num_output, train_ratio)
    
    INPUT_DIM = 1
    OUTPUT_DIM = 1
    ENC_EMB_DIM = 8
    DEC_EMB_DIM = 8
    HID_DIM = 16
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    N_EPOCHS = 200
    CLIP = 1
    batch_size = 8
    lr=0.02

    train_dataset = TimeSeriesDataset(train_x, train_y)
    val_dataset = TimeSeriesDataset(val_x, val_y)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(enc, dec, device).to(device)
    init_weights(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_ep = 0
    best_valid_loss = float('inf')
    os.makedirs(exp_dir, exist_ok=True)

    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP, device)
        valid_loss = validate(model, val_dataloader, criterion, device)
        
        end_time = time.time()
        
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss <= best_valid_loss:
            best_ep = epoch
            best_valid_loss = valid_loss
            model_path = os.path.join(exp_dir, 'best_model_univariate.pt')
            torch.save(model.state_dict(), model_path)
        
        print(f'Epoch: [{epoch+1}/{N_EPOCHS}] | train loss: {train_loss:.4f} | val loss: {valid_loss:.4f}')

    model_path = os.path.join(exp_dir, 'last_model_univariate.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Best model: ep={best_ep}, val loss={best_valid_loss:.4f}")


if __name__ == "__main__":
    main()