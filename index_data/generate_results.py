import pandas as pd
import numpy as np
import empyrical
import matplotlib.pyplot as plt

def analyze_index(symbol):
    df = pd.read_pickle(f"index_data/{symbol}.pkl")
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    closes = df['Close'].pct_change().values
    closes = closes[1:]
    df = df.to_numpy()
    #x = df[:,5]/df[:,2]
    a = np.cumprod(1 + closes, axis = 0)
    plt.plot(df[1:,1],a,label = symbol)
    print("INDEX:",symbol,"ROI:",a[-1],"SR:",empyrical.sharpe_ratio(closes,annualization=len(closes)), "CR:", empyrical.calmar_ratio(closes,annualization = len(closes)))

if __name__ == '__main__':
    start_date = "2021-12-21" 
    end_date = "2022-05-15"
    df = pd.read_pickle(f"index_data/VOO.pkl")
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    for symbol in ['VOO','SPY','IVW','RSP']:
        analyze_index(symbol)

    me =[1.0037, 1.0105, 1.0158, 1.0303, 1.0297, 1.0370, 1.0288, 1.0279, 1.0274,
        1.0271, 1.0116, 1.0114, 1.0041, 1.0096, 1.0147, 1.0187, 1.0027, 1.0121,
        0.9962, 0.9784, 0.9689, 0.9614, 0.9770, 0.9797, 0.9671, 0.9593, 0.9757,
        0.9889, 0.9886, 0.9954, 0.9890, 0.9912, 0.9911, 1.0011, 1.0068, 1.0006,
        0.9889, 0.9836, 0.9870, 0.9907, 0.9914, 0.9926, 0.9764, 0.9685, 0.9650,
        0.9888, 0.9992, 0.9901, 0.9990, 1.0010, 1.0147, 1.0150, 1.0082, 1.0064,
        1.0125, 1.0059, 0.9997, 1.0088, 1.0077, 1.0196, 1.0245, 1.0277, 1.0292,
        1.0245, 1.0342, 1.0456, 1.0464, 1.0514, 1.0514, 1.0422, 1.0486, 1.0457,
        1.0469, 1.0552, 1.0678, 1.0677, 1.0606, 1.0596, 1.0664, 1.0632, 1.0621,
        1.0718, 1.0727, 1.0427, 1.0264, 1.0305, 1.0162, 1.0131, 1.0156, 0.9920,
        0.9938, 1.0037, 1.0242, 1.0031, 1.0041, 0.9847, 0.9740, 0.9701, 0.9782,
        0.9865]
    
    plt.plot(df['Date'],me, label = 'LEIDEN-RAT',linewidth=4,color = 'black')
    plt.xlabel("Trading dates", fontsize=50)
    plt.ylabel("Portfolio value", fontsize=50)
    plt.legend(loc="lower left",prop={'size': 50})
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=50)
    plt.show()
