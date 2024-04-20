import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import pandas_datareader as pdr

class ohlc(object):
    def __init__(self):
        self.ohlc = pd.DataFrame()
        self.rf_rate =  pd.DataFrame()
        self.bm = pd.DataFrame()
        self.vix = pd.DataFrame()
        self.current_date = ""
    
    def load(self):
        folder = os.path.dirname(os.path.realpath(__file__))+os.sep + 'data' + os.sep
        filenames =  [name for name in os.listdir(folder) if not os.path.isdir(folder+name)]
        hdf = [datetime.strptime(os.path.splitext(name)[0], '%Y-%m-%d') for name in filenames if name[-3:] == 'hdf']
        latest_file = sorted(hdf)[-1].strftime("%Y-%m-%d")   
        self.current_date = latest_file
        self.ohlc =  pd.read_hdf(f"{folder}{latest_file}.hdf",key='ohlc',mode="r")
        self.rf_rate = pdr.DataReader('TB3MS', 'fred', start='2010-01-01')['TB3MS']
        self.bm = round(yf.download(tickers= "^GSPC",start= '2010-01-01', interval = "1d",
                     group_by = 'column',proxy = None)['Adj Close'],2)
        self.vix = round(yf.download(tickers= "^VIX",start= '2010-01-01', interval = "1d",
                     group_by = 'column',proxy = None)['Close'],2)