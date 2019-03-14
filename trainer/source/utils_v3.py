import pandas as pd
import numpy as np

import os.path
import math
from sklearn.preprocessing import RobustScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from numba import jit

def getHighLow(time, data):
    day = data[datetime(time.year, time.month, time.day):time]

    high = 0
    low = 1000000
    for i, x in day.iterrows():
        if x['high'] > high: high = x['high']
        if x['low'] < low: low = x['low']
    
    return [high, low]

def getHigh(time, data):
    day = data[datetime(time.year, time.month, time.day):time]
    
    high = 0
    for h in day['high']:
        if h > high: high = h
    return high

def getLow(time, data):
    day = data[datetime(time.year, time.month, time.day):time]
    low = 1000000
    for l in day['low']:
        if l < low: low = l
    return low
    


def get_data_v2(csv_path, time_val, time_freq_str, series_count, nrows=None, test_count=15000, test_only=False, resample=False, preprocessed=False, npz=False, no_high_low=False, near_high_low=False, ahead='1Min', behind='5Min', categorical=False, ohlc=False):
    
    if npz:
        loaded = np.load(csv_path)
        return (loaded['x_train'], loaded['y_train'], loaded['x_test'], loaded['y_test'], list())
    
    savedName = '%d%s%d.csv' % (time_val, time_freq_str, series_count)

    def makedt(x):
        return datetime.fromtimestamp(int(x))

    time_freq = time_freq_str
    time = '%d%s' % (time_val, time_freq)
    seq = series_count

    if preprocessed:
        data = pd.read_csv(csv_path, index_col='time', nrows=nrows)
        data.index = pd.to_datetime(data.index)
    else:
        if os.path.isfile(savedName) and False:
            print("is saved")
            data = pd.read_csv(csv_path, index_col=0, date_parser=makedt)
        else:
            df = pd.read_csv(csv_path, nrows=nrows)
            df = df[df.open != 0]
            df.index = pd.to_datetime(df['time'], unit='s')
            
            print('loaded and indexed')
#             print(df.head())
            
            #df = df.resample(time).mean().dropna()
            #

            if resample:
                gr = df.resample(time)
                #                 gr = df.groupby(pd.Grouper(freq=time))
                
                if ohlc:
                    gdf = gr.agg({
                    'close':'ohlc',
                    'bids':'sum',
                    'offers':'sum'
                    })
                else:  
                    gdf = gr.agg({
                    #'open':'first',
                    #'high':'max',
                    #'low':'min',
                    'close':'last',
                    'bids':'sum',
                    'offers':'sum'
                    })
                df = gdf.copy()
                df = df.dropna()
                #df = df[(df.T != 0).any()]
                
            print('resampled')
#             print(df.head())

            df['date'] = df.index.to_pydatetime()
            df['date_normalized'] = df['date'].apply(lambda x: (x.hour * 10000) + (x.minute * 100) + x.second)

            data = df

    if ohlc:
        data = data[['date_normalized', 'open','high','low', 'close', 'bids', 'offers']]
    else:
        data = data[['date_normalized', 'close', 'bids', 'offers']]
    
    # train = np.empty((0, seq+2, data.shape[1]))
    # test = np.empty((0, seq+2, data.shape[1]))

    i = 0
    empty = list()
    
    price_index = data.columns.get_loc("close")
    count = 0
    
    starttime = pd.Timedelta(behind)
    endtime = pd.Timedelta(ahead)
    total = data.shape[0]
    
    fills = 0
    x_train_l = list()
    y_train_l = list()
    x_test_l = list()
    y_test_l = list()
    
    #@jit(nopython=True)
    def process(i, s, seq, count, test_count, x_train_l, y_train_l, x_test_l, y_test_l):
        # v = s[:seq+1]
        # t = s[-1:]

        # if categorical:
        #     last = v[-1:,price_index]
        #     target = t[0,price_index]
        #     diff_ticks = ((target - last) / 25.0)
        #     if (diff_ticks) > 1:
        #         t[0, price_index] = 1 # buy
        #     elif diff_ticks <  -1:
        #         t[0,price_index] = 2 # sell
        #     else:
        #         t[0,price_index] = 3 # hold



        d = np.vstack((s[:seq+1], s[-1:]))
        if d.shape[0] == seq + 2:
            if count < test_count:
                v = StandardScaler().fit_transform(d)
                x_test_l.append([v[:-1,]])
                y_test_l.append(v[-1:,price_index])
                count += 1
                return (False, count)
            else:
                v = StandardScaler().fit_transform(d)
                x_train_l.append([v[:-1,]])
                y_train_l.append(v[-1:,price_index])

            count += 1 
        else: 
            return (False, count)
        return (True, count)
            
    for rowtime in data.index:
        start = rowtime - starttime
        end = rowtime + endtime

        (filled, new_count) = process(i, data.loc[start:end].values, seq, count, test_count, x_train_l, y_train_l, x_test_l, y_test_l)
        
        count = new_count
        if not filled:
            empty.append(i)
        else: fills += 1
            
        if (i % 50000 == 0):
            print('------')
            print('')
            print('%f' % (i / total * 100))
            print(rowtime)
            print('')
            print('------')
        i += 1

    
    x_train = np.asarray(x_train_l)
    y_train = np.asarray(y_train_l)
    x_test = np.asarray(x_test_l)
    y_test = np.asarray(y_test_l)
    
    
    return (x_train, y_train, x_test, y_test, None)