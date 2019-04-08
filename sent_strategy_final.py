#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 08:05:47 2017

@author: DeanMeyer
"""

# load packages

import quandl
import pandas as pd
import numpy as np

#import sys
#sys.path.append('/Users/selenahe/Documents/Sentiment project') # where backtest_senti is 
#from backtest_senti import BacktestEngine

# input 
sp_100 = ['AAPL','ABBV','ABT','ACN','AGN','AIG','ALL','AMGN','AMZN','AXP','BA',
'BAC','BIIB','BK','BLK','BMY','BRK.B','C','CAT','CELG','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DHR','DIS','DOW','DUK','EMR','EXC','F','FB','FDX','FOX','FOXA','GD','GE','GILD','GM','GOOG','GOOGL','GS','HAL','HD','HON','IBM','INTC','JNJ','JPM','KHC','KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON','MRK','MS','MSFT','NEE','NKE','ORCL','OXY','PCLN','PEP','PFE' ,'PG','PM','PYPL','QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP','UPS','USB','UTX','V','VZ','WBA','WFC','WMT','XOM']

sp_500 = ['ABT', 'ABBV', 'ACN', 'ACE', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A',
                  'GAS', 'APD', 'ARG', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO',
                  'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH',
                  'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN',
                  'AZO', 'AVGO', 'AVB', 'AVY', 'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT',
                  'BDX', 'BBBY', 'BRK-B', 'BBY', 'BLX', 'HRB', 'BA', 'BWA', 'BXP', 'BSK', 'BMY',
                  'BRCM', 'BF-B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB', 'COF', 'CAH', 'HSIC',
                  'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK',
                  'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME',
                  'CMS', 'COH', 'KO', 'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX',
                  'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA',
                  'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO', 'DTV', 'DFS', 'DISCA', 'DISCK', 'DG',
                  'DLTR', 'D', 'DOV', 'DOW', 'DPS', 'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN', 'ETN',
                  'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 'EQT',
                  'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM',
                  'FFIV', 'FB', 'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FSIV', 'FLIR', 'FLS',
                  'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GPS', 'GRMN', 'GD',
                  'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL', 'GOOG', 'GWW',
                  'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HCN', 'HP', 'HES',
                  'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 'HCBK', 'HUM', 'HBAN', 'ITW', 'IR', 'INTC',
                  'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT',
                  'JNJ', 'JCI', 'JOY', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM', 'KMI',
                  'KLAC', 'KSS', 'KRFT', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LM', 'LEG', 'LEN', 'LVLT',
                  'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK',
                  'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MHFI', 'MCK',
                  'MJN', 'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU', 'MSFT', 'MHK', 'TAP',
                  'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV',
                  'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI',
                  'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY',
                  'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT',
                  'POM', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI',
                  'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD',
                  'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O',
                  'RHT', 'REGN', 'RF', 'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RLC', 'R',
                  'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIAL', 'SPG',
                  'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS',
                  'SBUX', 'HOT', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT',
                  'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO',
                  'TIF', 'TWX', 'TWC', 'TJK', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN',
                  'TYC', 'UA', 'UNP', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'URBN', 'VFC',
                  'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA',
                  'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB', 'WEC', 
                  'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION',
                  'ZTS']
dow_j = ['XOM','WMT','VZ','V','UTX','UNH','TRV','PG','PFE','NKE','MSFT','MRK','MMM','MCD','KO','JPM','JNJ','INTC','IBM','HD','GS','GE',
         'DIS','DD','CVX','CSCO','CAT','BA','AXP','AAPL']

                  
# get sentiment data

start_date = "2013-01-01"
end_date = "2017-07-31"

data = pd.DataFrame()

stock_basket = dow_j

for stock in stock_basket:
    try:
        stock_data = quandl.get('NS1/' + stock + '_US', authtoken="8-rjnPgRNxxdyLBQHxaW",
                                start_date= start_date, end_date=end_date, returns="pandas")
        stock_data['ticker'] = stock
        stock_data = stock_data.reset_index()
        data = pd.concat([data, stock_data], axis = 0)
    except:
        pass
data['Date'] = pd.to_datetime(data['Date'])
 
#data.to_csv('/Users/selenahe/Documents/Sentiment project/sp_100_sentiment_2013_2017.csv', index=False) 

# get earnings per share
earnings = pd.DataFrame()

for stock in stock_basket:
    try:
        stock_data = quandl.get('SF1/' + stock + '_EPS_MRQ', authtoken="8-rjnPgRNxxdyLBQHxaW",
                                start_date= start_date, end_date=end_date, returns="pandas")
        stock_data['ticker'] = stock
        stock_data = stock_data.reset_index()
        earnings = pd.concat([earnings, stock_data], axis = 0)
    except:
        pass

earnings.columns = ['Date', 'EPS', 'ticker']

earnings['Date'] = pd.to_datetime(earnings['Date'])

data = pd.merge(data,earnings, on=['Date', 'ticker'] , how= 'left')

data['EPS'] = data['EPS'].fillna(method='ffill')
data = data[ pd.notnull(data['EPS'])]
data = data[ pd.notnull(data['PE'])]


## get price data

price = test.price
price['Date'] = pd.to_datetime(price['Date'])

data = pd.merge(data,price[['Date','ticker','Adj_Close']], on=['Date', 'ticker'] , how= 'left')
data = data[ pd.notnull(data['Adj_Close'])]

data['PE']=(data['Adj_Close']/data['EPS'])

''' 
strategy 1:
  use sentiment: buy or short all stocks
'''
senti_mean = data.groupby(['Date'])['Sentiment'].mean().reset_index(name='daily_mean')
senti_sd = data.groupby(['Date'])['Sentiment'].std().reset_index(name='daily_sd')
senti_stats = pd.merge(senti_mean, senti_sd, on='Date')

senti_data = pd.merge(data, senti_stats, on='Date')

# use sentiment
senti_data['z_score'] =  (senti_data['Sentiment'] - senti_data['daily_mean'])/senti_data['daily_sd']
senti_data['z_sign'] =  np.where(senti_data['z_score']>0, 1, -1)
senti_data['abs_z'] =  senti_data['z_score'].abs()

daily_z = senti_data.groupby(['Date'])['abs_z'].sum().reset_index(name='daily_z')
senti_data = pd.merge(senti_data, daily_z, on='Date')

senti_data['weights'] = senti_data['z_score']/senti_data['daily_z'] 


df_alpha = pd.pivot_table(senti_data, columns='ticker', values='weights', index='Date', fill_value=0)

# backesting
test=BacktestEngine()
test.backtest_engine(df_alpha)
price=test.price

''' 
strategy 2:
  use sentiment: buy top 20% and short lowest 20%
'''
senti_mean = data.groupby(['Date'])['Sentiment'].mean().reset_index(name='daily_mean')
senti_sd = data.groupby(['Date'])['Sentiment'].std().reset_index(name='daily_sd')
senti_stats = pd.merge(senti_mean, senti_sd, on='Date')

senti_data = pd.merge(data, senti_stats, on='Date')

# use sentiment
senti_data['z_score'] =  (senti_data['Sentiment'] - senti_data['daily_mean'])/senti_data['daily_sd']
senti_data['z_sign'] =  np.where(senti_data['z_score']>0, 1, -1)
senti_data['abs_z'] =  senti_data['z_score'].abs()

# set buy or short 
buy_q = senti_data['Sentiment'] .quantile(.8)
short_q = senti_data['Sentiment'] .quantile(.2)

# subset
senti_data = senti_data[(senti_data['Sentiment'] < short_q) | (senti_data['Sentiment'] > buy_q) ].copy()


daily_z = senti_data.groupby(['Date'])['abs_z'].sum().reset_index(name='daily_z')
senti_data = pd.merge(senti_data, daily_z, on='Date')

senti_data['weights'] = senti_data['z_score']/senti_data['daily_z'] 


df_alpha = pd.pivot_table(senti_data, columns='ticker', values='weights', index='Date', fill_value=0)

# backesting
test=BacktestEngine(price)
test.backtest_engine(df_alpha)

''' 
strategy 3:
  use sentiment and news buzz: buy or short buzz >7
'''
senti_data = pd.merge(data, senti_stats, on='Date')

# use sentiment
senti_data['z_score'] =  (senti_data['Sentiment'] - senti_data['daily_mean'])/senti_data['daily_sd']
senti_data['z_sign'] =  np.where(senti_data['z_score']>0, 1, -1)
senti_data['abs_z'] =  senti_data['z_score'].abs()

# use news Buzz
senti_data = senti_data[ senti_data['News Buzz']>7 ]

# get weights
daily_z = senti_data.groupby(['Date'])['abs_z'].sum().reset_index(name='daily_z')
senti_data = pd.merge(senti_data, daily_z, on='Date')

senti_data['weights'] = senti_data['z_score']/senti_data['daily_z'] 

df_alpha = pd.pivot_table(senti_data, columns='ticker', values='weights', index='Date', fill_value=0)

# backesting
test=BacktestEngine(price)
test.backtest_engine(df_alpha)

''' 
strategy 4:
  use sentiment and news buzz: buy or short all stocks
'''

data['Senti_Buzz'] = data['Sentiment'] * data['News Buzz']
senti_mean = data.groupby(['Date'])['Senti_Buzz'].mean().reset_index(name='daily_mean')
senti_sd = data.groupby(['Date'])['Senti_Buzz'].std().reset_index(name='daily_sd')
senti_stats = pd.merge(senti_mean, senti_sd, on='Date')

senti_data = pd.merge(data, senti_stats, on='Date')

# use sentiment
senti_data['z_score'] =  (senti_data['Senti_Buzz'] - senti_data['daily_mean'])/senti_data['daily_sd']
senti_data['z_sign'] =  np.where(senti_data['z_score']>0, 1, -1)
senti_data['abs_z'] =  senti_data['z_score'].abs()

daily_z = senti_data.groupby(['Date'])['abs_z'].sum().reset_index(name='daily_z')
senti_data = pd.merge(senti_data, daily_z, on='Date')

senti_data['weights'] = senti_data['z_score']/senti_data['daily_z'] 

df_alpha = pd.pivot_table(senti_data, columns='ticker', values='weights', index='Date', fill_value=0)

# backesting
test=BacktestEngine(price)
test.backtest_engine(df_alpha)

''' 
strategy 5:
  use sentiment and news buzz: buy top 20% and short lowest 20%
'''

data['Senti_Buzz'] = data['Sentiment'] * data['News Buzz']
senti_mean = data.groupby(['Date'])['Senti_Buzz'].mean().reset_index(name='daily_mean')
senti_sd = data.groupby(['Date'])['Senti_Buzz'].std().reset_index(name='daily_sd')
senti_stats = pd.merge(senti_mean, senti_sd, on='Date')

senti_data = pd.merge(data, senti_stats, on='Date')

# use sentiment
senti_data['z_score'] =  (senti_data['Senti_Buzz'] - senti_data['daily_mean'])/senti_data['daily_sd']
senti_data['z_sign'] =  np.where(senti_data['z_score']>0, 1, -1)
senti_data['abs_z'] =  senti_data['z_score'].abs()

# set buy or short 
buy_q = senti_data['Senti_Buzz'] .quantile(.8)
short_q = senti_data['Senti_Buzz'] .quantile(.2)

# subset
senti_data = senti_data[(senti_data['Senti_Buzz'] < short_q) | (senti_data['Senti_Buzz'] > buy_q) ].copy()

daily_z = senti_data.groupby(['Date'])['abs_z'].sum().reset_index(name='daily_z')
senti_data = pd.merge(senti_data, daily_z, on='Date')

senti_data['weights'] = senti_data['z_score']/senti_data['daily_z'] 

df_alpha = pd.pivot_table(senti_data, columns='ticker', values='weights', index='Date', fill_value=0)

# backesting
test=BacktestEngine(price)
test.backtest_engine(df_alpha)

''' 
strategy 6:
  use sentiment and news buzz and EPS: buy top 20% and short lowest 20%
'''


data['Senti_Buzz'] = data['Sentiment'] * data['News Buzz'] * data['EPS']
senti_mean = data.groupby(['Date'])['Senti_Buzz'].mean().reset_index(name='daily_mean')
senti_sd = data.groupby(['Date'])['Senti_Buzz'].std().reset_index(name='daily_sd')
senti_stats = pd.merge(senti_mean, senti_sd, on='Date')

senti_data = pd.merge(data, senti_stats, on='Date')

# use sentiment
senti_data['z_score'] =  (senti_data['Senti_Buzz'] - senti_data['daily_mean'])/senti_data['daily_sd']
senti_data['z_sign'] =  np.where(senti_data['z_score']>0, 1, -1)
senti_data['abs_z'] =  senti_data['z_score'].abs()

# set buy or short 
buy_q = senti_data['Senti_Buzz'] .quantile(.8) 
short_q = senti_data['Senti_Buzz'] .quantile(.2)

# subset
senti_data = senti_data[(senti_data['Senti_Buzz'] < short_q) | (senti_data['Senti_Buzz'] > buy_q) ].copy()

daily_z = senti_data.groupby(['Date'])['abs_z'].sum().reset_index(name='daily_z')
senti_data = pd.merge(senti_data, daily_z, on='Date')

senti_data['weights'] = senti_data['z_score']/senti_data['daily_z'] 

df_alpha = pd.pivot_table(senti_data, columns='ticker', values='weights', index='Date', fill_value=0)

# backesting
test=BacktestEngine(price)
test.backtest_engine(df_alpha)

''' 
strategy 7:
  use sentiment and news buzz and EPS: buy top 20% and short lowest 20%
'''


data['Senti_Buzz'] = data['Sentiment'] * data['News Buzz'] * data['PE']

senti_mean = data.groupby(['Date'])['Senti_Buzz'].mean().reset_index(name='daily_mean')
senti_sd = data.groupby(['Date'])['Senti_Buzz'].std().reset_index(name='daily_sd')
senti_stats = pd.merge(senti_mean, senti_sd, on='Date')

senti_data = pd.merge(data, senti_stats, on='Date')
senti_data= senti_data[ pd.notnull(senti_data['Senti_Buzz'])]

# use sentiment
senti_data['z_score'] =  (senti_data['Senti_Buzz'] - senti_data['daily_mean'])/senti_data['daily_sd']
senti_data['z_sign'] =  np.where(senti_data['z_score']>0, 1, -1)
senti_data['abs_z'] =  senti_data['z_score'].abs()

# set buy or short 
buy_q = senti_data['Senti_Buzz'] .quantile(.8)
short_q = senti_data['Senti_Buzz'] .quantile(.2)

# subset
senti_data = senti_data[(senti_data['Senti_Buzz'] < short_q) | (senti_data['Senti_Buzz'] > buy_q) ].copy()

daily_z = senti_data.groupby(['Date'])['abs_z'].sum().reset_index(name='daily_z')
senti_data = pd.merge(senti_data, daily_z, on='Date')

senti_data['weights'] = senti_data['z_score']/senti_data['daily_z'] 

df_alpha = pd.pivot_table(senti_data, columns='ticker', values='weights', index='Date', fill_value=0)

# backesting
test=BacktestEngine(price)
test.backtest_engine(df_alpha)