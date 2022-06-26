import yfinance as yf
import pandas as pd

def download_data(symbols):
    return yf.download(
            tickers = symbols,
            period = 'max',
            interval = "1d",
            group_by = 'ticker',
            prepost = True,
            threads = True,
            proxy = None
        )

def process_save_data(data,symbols):
    for symbol in symbols:
        df = data[symbol]
        df.dropna(inplace=True)
        df.reset_index(inplace = True)
        df = df[df['Date'] >= '2004-01-02'].reset_index()
        print(df.head())
        pd.to_pickle(df,f"index_data/{symbol}.pkl")

if __name__ == "__main__":
    symbols = ['TAN','^GSPC','FEZ','^IXIC','VOO','SPY','IVW','RSP']
    process_save_data(download_data(symbols),symbols)