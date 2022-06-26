import numpy as np
from datetime import  datetime
from mongo_db.utils import get_db_collection
from pymongoarrow.api import Schema,find_pandas_all
import pyarrow
import pandas as pd
import torch

class StockTradingEnv():

    def __init__(self,trading_window_size,start_date,end_date,path,number_of_days_trading = 25,assets_number = 25,bias = 5e-5,batch_size = 128,random = True,device = 'cpu',data_file_path = None):
        self.assets_number = assets_number
        self.trading_window_size = trading_window_size
        self.batch_size = batch_size
        self.number_of_days_trading = number_of_days_trading
        self.device = device
        self.all_data, self.future_prices, _  = self.prepare_data(start_date,end_date,path,data_file_path)
        self.weights = torch.ones((self.all_data.shape[0],self.assets_number),  dtype= torch.float, device = self.device) * 1/self.assets_number
        self.bias = bias
        self.random = random
        self.end_tick = self.all_data.shape[0]
        
    
    def sample(self, start, end):
        ran = np.random.geometric(self.bias)
        result = abs(end - ran )

        while ran > (end - start) or result % self.number_of_days_trading == 0:
            ran = np.random.geometric(self.bias)
            result = abs(end - ran)

        return result

    def next_experience_batch(self):
        batch = []
        if self.random:
            for _ in range(self.batch_size):
                batch.append(self.sample(0,self.end_tick - self.batch_size))
        else:
            batch_start = self.sample(0,self.end_tick - self.batch_size)
            batch = [x for x in range(batch_start,batch_start+self.batch_size-1)]
        
        return np.array(batch)

    def get_batch(self):
        indexes = self.next_experience_batch()
        prev_indexes = indexes - 1

        return self.all_data[indexes] , self.future_prices[indexes], self.weights[prev_indexes], indexes

    def set_new_weigts(self,indexes,new_weights):
        self.weights[indexes] = new_weights

    def get_all(self):
        return self.all_data[:], self.future_prices[:], self.weights[:]

    def prepare_data(self,start_date,end_date,file_portfolio,file_data):
        if file_data != None:
            all_data = pd.read_pickle(file_data)
            all_data = all_data[all_data.columns.intersection(['date','Open','High','Low','Close','normalized_Open','symbol'])]
            all_data = all_data[['date','Open','High','Low','Close','normalized_Open','symbol']]
            all_data = all_data[(all_data['date'] >= start_date) & (all_data['date'] <= end_date)]
        else:         
            db_collection = get_db_collection()
            schema = Schema({'date': pyarrow.string(),'Open': float, 'High': float,'Low': float,'Close': float,'normalized_Open':float,'symbol': pyarrow.string()})
            date_filter = {'date':{'$gte':start_date,'$lte':end_date}}
            all_data = find_pandas_all(db_collection, date_filter ,schema=schema)
        all_data = all_data.to_numpy()
        all_dates = sorted(np.unique(all_data[:,0]))
        all_data[:,2] = all_data[:,2]/all_data[:,1]
        all_data[:,3] = all_data[:,3]/all_data[:,1]
        all_data[:,4] = all_data[:,4]/all_data[:,1]
        with open(file_portfolio, "r") as fo:
            x = fo.readlines()

        data_windowed_vectors = []
        future_prices = []
        i = self.trading_window_size

        print(all_dates[i])
        for portfolio in x:
            cur = portfolio.strip().split(",")
            till_date = i + self.number_of_days_trading
            
            if till_date >= len(all_dates):
                break
            c = all_data[(all_data[:,0] >= all_dates[i-self.trading_window_size]) & (all_data[:,0] < all_dates[till_date]) & (np.isin(all_data[:,6],cur))]
            c = np.delete(c, [0,5,6], 1)
            c_grouped = np.array_split(c, self.trading_window_size + self.number_of_days_trading)
            for p in range(self.trading_window_size,self.trading_window_size+self.number_of_days_trading):
                current = np.array(c_grouped[p-self.trading_window_size:p],dtype = np.float16).transpose((2,1,0))
                future_prices.append(np.array(c_grouped[p][:,3], dtype = np.float16))
                data_windowed_vectors.append(current)
            i = i + self.number_of_days_trading
            print(all_dates[till_date])
        return torch.tensor(data_windowed_vectors, dtype= torch.float, device = self.device), torch.tensor(future_prices, dtype= torch.float, device = self.device), all_dates

    