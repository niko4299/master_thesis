import yfinance as yf
import pandas as pd
import utils
from config import SYMBOLS
import numpy as np

def download_data():
    return yf.download(
            tickers = SYMBOLS,
            period = 'max',
            interval = "1d",
            group_by = 'ticker',
            prepost = True,
            threads = True,
            proxy = None
        )

def process_data(data):
    all_data_sql = []

    for symbol in SYMBOLS:
        df = data[symbol]
        df.dropna(inplace=True)
        df.reset_index(inplace = True)
        df = df[df['Date'] >= '2002-01-02'].reset_index()
        df.drop(columns=['index'], inplace=True)
        df['symbol'] = symbol
        rolled = np.roll(df['Open'].values, shift=1)
        # df['normalized_High'] = df['High'].values / rolled
        df['normalized_Open'] = df['Open'].values / rolled
        # df['normalized_Close'] = df['Close'].values / rolled
        # df['normalized_Low'] = df['Low'].values / rolled
        df.rename(columns= {'Date':'date'}, inplace=True)
        all_data_sql.extend(df.values[1:])

    total_df = pd.DataFrame(all_data_sql, columns=df.columns)
    total_df['date'] = total_df['date'].apply(lambda x: str(x).split(" ")[0])
    grouped = total_df.groupby('date')
    fix_dates = [x for x in grouped.groups if len(grouped.get_group(x)) != len(SYMBOLS)]
    total_df = total_df[~total_df['date'].isin(fix_dates)]
    total_df.reset_index(inplace=True)
    total_df = total_df.sort_values(by=['date','symbol'],ascending=True)
    
    pd.to_pickle(total_df,'mongo_db/total_df.pkl')

    return total_df

if __name__ == "__main__":
    df = process_data(download_data())
    # collection = utils.get_db_collection()
    # collection.drop()
    # collection.insert_many(df.to_dict('records'))

