import os
import time
import pandas as pd
import requests, json
from datetime import datetime 

#------------------------------------------------------------------------------------------#

def download_tw_stock(start_year, stock_no, dst_data_dir):
    # download data
    date_now = datetime.now()
    years = [n for n in range(start_year, date_now.year+1)]
    dst_data_dir_by_stock = os.path.join(dst_data_dir, stock_no)
    os.makedirs(dst_data_dir_by_stock, exist_ok=True)
    
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9", 
        "Accept-Encoding": "gzip, deflate, br", 
        "Accept-Language": "zh-TW,zh;q=0.9", 
        "Host": "example.com",  #目標網站
        "Sec-Fetch-Dest": "document", 
        "Sec-Fetch-Mode": "navigate", 
        "Sec-Fetch-Site": "none", 
        "Upgrade-Insecure-Requests": "1", 
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36", #使用者代理
        "Referer": "https://www.google.com/"  #參照位址
    }
    months = []
    data = None
    for year in years:
        month_end = 12
        if year==years:
            month_end = date_now.month
        for mon in range(1, month_end+1):
            date = f'{year}{mon:02}01'
            print(date)
            html = requests.get('https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=%s&stockNo=%s' % (date,stock_no),
                                headers=headers)
            content = json.loads(html.text)
            stock_data = content['data']
            col_name = content['fields']

            data_temp = pd.DataFrame(data=stock_data, columns=col_name)
            if data is None:
                data = data_temp
            else:
                data = pd.concat((data, data_temp))

            # data cleaning
            data.columns = ["date", "trading_volume", "business_volume", "opening", "max", "min", "closing", "spread", "num_transactions"]

            data["trading_volume"] = data["trading_volume"].apply(lambda x: x.replace(",", ""))
            data["business_volume"] = data["business_volume"].apply(lambda x: x.replace(",", ""))

            dst_data_path = os.path.join(dst_data_dir_by_stock, "0050_{year}{mon:02}.csv")
            data.to_csv(dst_data_path, index=False)
            time.sleep(30)
            
def merge_data(dst_data_dir):
    data_name_list = os.listdir(dst_data_dir)
    
    # dst_data_path = os.path.join(raw_data_dir, "0050_{start_year}01_{date_now.year}{date_now.month:02}.csv")
    # data.to_csv(dst_data_path, index=False)
    pass

#------------------------------------------------------------------------------------------#

def main():
    root_dir = "."
    start_year = 2019
    stock_no = '0050'
    dst_data_dir = os.path.join(root_dir, "dataset", "tw_stock")
    
    download_tw_stock(start_year, stock_no, dst_data_dir)
    merge_data(dst_data_dir)
    
    
#------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()