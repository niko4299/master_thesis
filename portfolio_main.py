from portfolio_construct.portfolio_construct import PortfolioConstruct
from pymongoarrow.api import Schema,find_pandas_all
import pyarrow
from mongo_db.utils import get_db_collection
import numpy as np
import pandas as pd
import argparse
import json
from tqdm import tqdm

def calculate_return(current_portfolio,input):
    global portfolio_value
    input = input[np.isin(input[:,5],current_portfolio)]
    input[:,3] = input[:,3]/input[:,1]
    dates = sorted(np.unique(input[:,0]))
    portfolio_return = []
    for date in dates:
        day = input[input[:,0]==date]
        portfolio_return.append(np.sum(np.dot(1/ctx['selected_symbols'],np.log(day[:,3].astype(float)))))
    
    portfolio_value = portfolio_value + np.sum(portfolio_return)
    print(portfolio_value)

def start_trading(db_collection,start_date,end_date, file_data = None):
    if file_data != None:
      all_data = pd.read_pickle(file_data)
      all_data = all_data[all_data.columns.intersection(['date','Open','High','Low','Close','symbol'])]
      all_data = all_data[['date','Open','High','Close','Low','symbol']]
      all_data = all_data[(all_data['date'] >= start_date) & (all_data['date'] <= end_date)]
    else:
      schema = Schema({'date': pyarrow.string(),'Open': float, 'High': float,'Close': float,'Low': float,'symbol': pyarrow.string()})
      date_filter = {'date':{'$gte':start_date,'$lt':end_date}}
      all_data = find_pandas_all(db_collection, date_filter ,schema=schema)

    all_data = all_data.to_numpy()
    all_dates = sorted(np.unique(all_data[:,0])) 
    number_of_all_dates = len(all_dates) - ctx['number_of_trading_days']
    for i in tqdm(range(20,number_of_all_dates, ctx['number_of_trading_days'])):
        till_date = i + ctx['number_of_trading_days']
    
        current_ndarray = all_data[(all_data[:,0] >= all_dates[i-20]) & (all_data[:,0] < all_dates[till_date])]
        to_choose = current_ndarray[current_ndarray[:,0] < all_dates[i]]
        current_portfolio = selector.construct_portfolio(to_choose)
       
        # to_trade = current_ndarray[(current_ndarray[:,0] >= all_dates[i]) & (current_ndarray[:,0] < all_dates[till_date])]
        portfolio_symbols.append(current_portfolio)
        # calculate_return(current_portfolio,to_trade)
        
def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type = str,default='default_configs.json',help='config file')
    parser.add_argument('--data_file_path',type = str,default='mongo_db/total_df.pkl',help='config file')
    return parser.parse_args()

if __name__ == '__main__':
    db_collection = get_db_collection()
    args = get_args()
    with open(args.config_file,"r") as fhandle:
        ctx = json.load(fhandle)
    selector = PortfolioConstruct(ctx['selected_symbols'],20)
    portfolio_value = 1
    portfolio_symbols = []
    
    for phase in ['train', 'val', 'test']:
        start_date = ctx[phase]['start']
        end_date = ctx[phase]['end']
        start_trading(db_collection,start_date,end_date, file_data = args.data_file_path)
        with open(f"portfolio_files/{phase}/portfolios_{start_date}_{end_date}.txt", "w") as fo:
            for one in portfolio_symbols:
                s = ",".join(one) + '\n'
                fo.write(s)
        print('Ended:',phase)

