#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 13:16:07 2017

@author: DeanMeyer
"""

# load packages
#import datetime as dt

import quandl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

##########################################################
def two_dec_places(x, pos):
    """
    Adds 1/100th decimal to plot ticks.

    """

    return '%.2f' % x


def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """

    return '%.0f%%' % x
###########################################################

class BacktestEngine():

    
    def __init__(self, price = pd.DataFrame()):
        print('Start Backtest Engine ...')
        
        if price.empty:
            self.loadData()
            print('Finish Data Loading.')
        else:
            self.price = price
            self.ret = pd.pivot_table(self.price, columns='ticker', values='Adj_Close', index='Date', fill_value=0).pct_change()
            print('Preload Data.')

        
    
    def loadData(self):
        """
        load data (price and returns) 
        """        
        
        sp_100 = ['AAPL','ABBV','ABT','ACN','AGN','AIG','ALL','AMGN','AMZN','AXP','BA',
        'BAC','BIIB','BK','BLK','BMY','BRK.B','C','CAT','CELG','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DHR','DIS','DOW','DUK','EMR','EXC','F','FB','FDX','FOX','FOXA','GD','GE','GILD','GM','GOOG','GOOGL','GS','HAL','HD','HON','IBM','INTC','JNJ','JPM','KHC','KMI','KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON','MRK','MS','MSFT','NEE','NKE','ORCL','OXY','PCLN','PEP','PFE' ,'PG','PM','PYPL','QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP','UPS','USB','UTX','V','VZ','WBA','WFC','WMT','XOM']
        
        start_date = "2013-01-01"
        end_date = "2017-07-31"
        
        
        # get price data
        p_data = pd.DataFrame()
        
        for stock in sp_100:
            try:
                stock_data = quandl.get('EOD/' + stock, authtoken="8-rjnPgRNxxdyLBQHxaW",
                                        start_date= start_date, end_date=end_date, returns="pandas")
                
                stock_data['ticker'] = stock
                stock_data = stock_data.reset_index()
                p_data = pd.concat([p_data, stock_data], axis = 0)
            except:
                pass
            
        self.price = p_data
        
        # calculate returns
        self.ret = pd.pivot_table(self.price, columns='ticker', values='Adj_Close', index='Date', fill_value=0).pct_change()

        
    def cum_returns(self, returns, starting_value=0):
        """
        Compute cumulative returns from simple returns.
        Parameters
        ----------
        returns : pd.Series, np.ndarray, or pd.DataFrame
            Returns of the strategy as a percentage, noncumulative.
            Date as index
        starting_value : float, optional
           The starting returns.
        Returns
        -------
        pd.Series, np.ndarray, or pd.DataFrame
            Series of cumulative returns.

        """
    
        if len(returns) < 1:
            return type(returns)([])
    
        if np.any(np.isnan(returns)):
            returns = returns.copy()
            returns[np.isnan(returns)] = 0.
    
        df_cum = (returns + 1).cumprod(axis=0)
    
        if starting_value == 0:
            return df_cum - 1
        else:
            return df_cum * starting_value
            
    
    
    def aggregate_returns(self, returns, convert_to):
        """
        Aggregates returns by week, month, or year.
        Parameters
        ----------
        returns : pd.Series
           Daily returns of the strategy, noncumulative.
        convert_to : str
            Can be 'weekly', 'monthly', or 'yearly'.
        Returns
        -------
        pd.Series
            Aggregated returns.
        """
    
        def cumulate_returns(x):
            return self.cum_returns(x).iloc[-1]
    
        if convert_to == 'weekly':
            grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
        elif convert_to == 'monthly':
            grouping = [lambda x: x.year, lambda x: x.month]
        elif convert_to == 'yearly':
            grouping = [lambda x: x.year]
        else:
            raise ValueError(
                'convert_to must be {}, {} or {}'.format('weekly', 'monthly', 'yearly')
            )
    
        return returns.groupby(grouping).apply(cumulate_returns)
    
    
    def max_drawdown(self, returns):
        """
        Determines the maximum drawdown of a strategy.
        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Daily returns of the strategy, noncumulative.
        Returns
        -------
        float
            Maximum drawdown.

        """
    
        if len(returns) < 1:
            return np.nan
    
        if type(returns) == pd.Series:
            returns = returns.values
    
        cumulative = np.insert(self.cum_returns(returns, starting_value=100), 0, 100)
        max_return = np.fmax.accumulate(cumulative)
    
        return np.nanmin((cumulative - max_return) / max_return)
  
        
    def value_at_risk(self, returns, cutoff=0.05):
        """
        Value at risk (VaR) of a returns stream.
        Parameters
        ----------
        returns : pandas.Series or 1-D numpy.array
            Non-cumulative daily returns.
        cutoff : float, optional
            Decimal representing the percentage cutoff for the bottom percentile of
            returns. Defaults to 0.05.
        Returns
        -------
        VaR : float
            The VaR value.
        """
        return np.percentile(returns, 100 * cutoff)

    def get_max_drawdown_underwater(self, underwater):
        """
        Determines peak, valley, and recovery dates given an 'underwater'
        DataFrame.
        An underwater DataFrame is a DataFrame that has precomputed
        rolling drawdown.
        Parameters
        ----------
        underwater : pd.Series
           Underwater returns (rolling drawdown) of a strategy.
        Returns
        -------
        peak : datetime
            The maximum drawdown's peak.
        valley : datetime
            The maximum drawdown's valley.
        recovery : datetime
            The maximum drawdown's recovery.
        """
    
        valley = np.argmin(underwater)  # end of the period
        # Find first 0
        peak = underwater[:valley][underwater[:valley] == 0].index[-1]
        # Find last 0
        try:
            recovery = underwater[valley:][underwater[valley:] == 0].index[0]
        except IndexError:
            recovery = np.nan  # drawdown not recovered
        return peak, valley, recovery
            
    def get_max_drawdown(self, returns):
        """
        Determines the maximum drawdown of a strategy.
        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
        Returns
        -------
        float
            Maximum drawdown.

        """
    
        returns = returns.copy()
        df_cum = self.cum_returns(returns, 1.0)
        running_max = np.maximum.accumulate(df_cum)
        underwater = df_cum / running_max - 1
        return self.get_max_drawdown_underwater(underwater)
        
        
    def get_top_drawdowns(self, returns, top=3):
        """
        Finds top drawdowns, sorted by drawdown amount.
        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
        top : int, optional
            The amount of top drawdowns to find (default 3).
        Returns
        -------
        drawdowns : list
            List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
        """
    
        returns = returns.copy()
        df_cum = self.cum_returns(returns, 1.0)
        running_max = np.maximum.accumulate(df_cum)
        underwater = df_cum / running_max - 1
    
        drawdowns = []
        for t in range(top):
            peak, valley, recovery = self.get_max_drawdown_underwater(underwater)
            # Slice out draw-down period
            if not pd.isnull(recovery):
                underwater.drop(underwater[peak: recovery].index[1:-1],
                                inplace=True)
            else:
                # drawdown has not ended yet
                underwater = underwater.loc[:peak]
    
            drawdowns.append((peak, valley, recovery))
            if (len(returns) == 0) or (len(underwater) == 0):
                break
    
        return drawdowns        
        
    def gen_drawdown_table(self, returns, top=3):
        """
        Places top drawdowns in a table.
        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
        top : int, optional
            The amount of top drawdowns to find (default 3).
        Returns
        -------
        df_drawdowns : pd.DataFrame
            Information about top drawdowns.
        """
    
        df_cum = self.cum_returns(returns, 1.0)
        drawdown_periods = self.get_top_drawdowns(returns, top=top)
        df_drawdowns = pd.DataFrame(index=list(range(top)),
                                    columns=['Net drawdown in %',
                                             'Peak date',
                                             'Valley date',
                                             'Recovery date',
                                             'Duration'])
    
        for i, (peak, valley, recovery) in enumerate(drawdown_periods):
            if pd.isnull(recovery):
                df_drawdowns.loc[i, 'Duration'] = np.nan
            else:
                df_drawdowns.loc[i, 'Duration'] = len(pd.date_range(peak,
                                                                    recovery,
                                                                    freq='B'))
            df_drawdowns.loc[i, 'Peak date'] = (peak.to_pydatetime()
                                                .strftime('%Y-%m-%d'))
            df_drawdowns.loc[i, 'Valley date'] = (valley.to_pydatetime()
                                                  .strftime('%Y-%m-%d'))
            if isinstance(recovery, float):
                df_drawdowns.loc[i, 'Recovery date'] = recovery
            else:
                df_drawdowns.loc[i, 'Recovery date'] = (recovery.to_pydatetime()
                                                        .strftime('%Y-%m-%d'))
            df_drawdowns.loc[i, 'Net drawdown in %'] = (
                (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]) * 100
    
        df_drawdowns['Peak date'] = pd.to_datetime(df_drawdowns['Peak date'])
        df_drawdowns['Valley date'] = pd.to_datetime(df_drawdowns['Valley date'])
        df_drawdowns['Recovery date'] = pd.to_datetime(
            df_drawdowns['Recovery date'])
    
        return df_drawdowns
        
    def plot_drawdown_periods(self, returns, top=3, **kwargs):
        """
        Plots cumulative returns highlighting top drawdown periods.
        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
        top : int, optional
            Amount of top drawdowns periods to plot (default 3).

        **kwargs, optional
            Passed to plotting function.

        """
        #ax = plt.gca()
    
        #y_axis_formatter = FuncFormatter(two_dec_places)
    
        df_cum_rets = self.cum_returns(returns, starting_value=1.0)
        df_drawdowns = self.gen_drawdown_table(returns, top=top)
    
        plt.figure()
        ax = df_cum_rets.plot(**kwargs)
    
        lim = ax.get_ylim()
        colors = sns.cubehelix_palette(len(df_drawdowns))[::-1]
        for i, (peak, recovery) in df_drawdowns[
                ['Peak date', 'Recovery date']].iterrows():
            if pd.isnull(recovery):
                recovery = returns.index[-1]
            ax.fill_between((peak, recovery),
                            lim[0],
                            lim[1],
                            alpha=.4,
                            color=colors[i])
        #ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        ax.set_ylim(lim)
        ax.set_title('Top %i drawdown periods' % top)
        ax.set_ylabel('Cumulative returns')
        ax.legend(['Portfolio'], loc='upper left')
        ax.set_xlabel('')
        print(df_drawdowns)
    
    
    def plot_drawdown_underwater(self, returns, **kwargs):
        '''
        Plots how far underwaterr returns are over time, or plots current
        drawdown vs. date.
        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.
 
        **kwargs, optional
            Passed to plotting function.
 
        '''
    
        #ax = plt.gca()
    
        #y_axis_formatter = FuncFormatter(percentage)
        plt.figure()
        df_cum_rets = self.cum_returns(returns, starting_value=1.0)
        running_max = np.maximum.accumulate(df_cum_rets)
        underwater = -100 * ((running_max - df_cum_rets) / running_max)
        ax = (underwater).plot(kind='area', color='coral', alpha=0.7, **kwargs)
        #ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        ax.set_ylabel('Drawdown')
        ax.set_title('Underwater plot')
        ax.set_xlabel('')

    
        
    def plot_monthly_returns_heatmap(self, returns, **kwargs):
        '''
        Plots a heatmap of returns by month.
        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.

        **kwargs, optional
            Passed to seaborn plotting function.
        '''
    

        #ax = plt.gca()
    
        monthly_ret_table = self.aggregate_returns(returns, 'monthly')
        monthly_ret_table = monthly_ret_table.unstack().round(3)
        
        plt.figure()
        ax = sns.heatmap(
            monthly_ret_table.fillna(0) *
            100.0,
            annot=True,
            annot_kws={
                "size": 9},
            alpha=1.0,
            center=0.0,
            cbar=False,
            cmap=matplotlib.cm.RdYlGn,
             **kwargs)
        ax.set_ylabel('Year')
        ax.set_xlabel('Month')
        ax.set_title("Monthly returns (%)")
        return ax           


    def plot_annual_returns(self, returns, **kwargs):
        '''
        Plots a bar graph of returns by year.
        
        Parameters
        ----------
        returns : pd.Series
            Daily returns of the strategy, noncumulative.

        **kwargs: optional
            Passed to plotting function.
        '''

    
        ann_ret_df = pd.DataFrame(
            self.aggregate_returns(
                returns,
                'yearly'))
            
        plt.figure()
        ax =(100 * ann_ret_df.sort_index(ascending=True)
         ).plot(kind='bar', alpha=0.70, color = 'steelblue', **kwargs)

        ax.axhline(100 * ann_ret_df.values.mean(), color='red',
        linestyle='--', lw=4, alpha=0.4)

        plt.xticks(rotation=0)
        ax.set_ylabel('Returns (%)')
        ax.set_xlabel('Year')
        ax.set_title("Annual returns")
        ax.legend(['mean'])
        ax.axhline(0.0, color='black', linestyle='-', lw=3)

    def get_benchmark_ret(self):
        ben = quandl.get('BCIW/_SPXT', authtoken="8-rjnPgRNxxdyLBQHxaW",
                         start_date= self.start_date, end_date=self.end_date, returns="pandas")
        ben = ben.reset_index()
        ben['S&P 500'] = 'S&P 500'
        ben = pd.pivot_table(ben, columns='S&P 500', values='Close', index='Date', fill_value=0).pct_change()
        return ben['S&P 500']
    
    def plot_cum_returns(self, returns):
        
        cum_ret = self.cum_returns(returns, starting_value=1).to_frame(name= 'Portfolio')
        cum_ben = self.cum_returns(self.get_benchmark_ret(), starting_value=1).to_frame(name='S&P 500')
        
        cum_ret2 = cum_ret.merge(cum_ben, how='inner',left_index=True, right_index=True)
        plt.figure()
        cum_ret2.plot(title = 'Backtest Equity Curve')
        
    def backtest_engine(self, df_alpha): 
        """
        Main backtest function
        Returns strategy equity curve time-series and statistics 
        """
        
        df_alpha = df_alpha.shift(-1)
        df_alpha = df_alpha.dropna(how='all')
        
        
        print('Total Tickers in alpha: ' + str(df_alpha.shape[1]))
                
        # backtest period
        self.start_date = pd.to_datetime(df_alpha.index)[0]
        self.end_date = pd.to_datetime(df_alpha.index)[-1]
        print('Backtest period: ' + self.start_date.strftime("%Y-%m-%d") + ' to ' + self.end_date.strftime("%Y-%m-%d"))
        
        # dataframe to hold key statistics for backtesting
        stats = pd.DataFrame(index = ['Annual Return (%)','Annual Volatility (%)',
                                      'Information Ratio','Sortino Ratio',
                                      'Max Drawdown(%)', 'Value at Risk (%)'],
                             columns = ['Statistics'])  
        
        ret2 = self.ret[pd.to_datetime(self.ret.index) >= self.start_date ]
        ret2 = ret2[pd.to_datetime(ret2.index) <= self.end_date ]
        
        
        self.strategy_ret = (ret2 * df_alpha).sum(axis = 1) # daily return performance
            
        stats.loc['Annual Return (%)','Statistics'] = round(self.strategy_ret.mean() * 252 * 100 ,2)
        stats.loc['Annual Volatility (%)','Statistics'] = round(self.strategy_ret.std() * np.sqrt(252) * 100 ,2)
                 
        ir = self.strategy_ret.mean() / self.strategy_ret.std() * np.sqrt(252) # information ratio
        sortino = self.strategy_ret.mean() / self.strategy_ret[self.strategy_ret<0].std() * np.sqrt(252) # sortino ratio
        stats.loc['Information Ratio','Statistics'] = round(ir,2)
        stats.loc['Sortino Ratio','Statistics'] = round(sortino,2)
        '''        
        xs = ret3.cumsum().values # 
        xs[np.isnan(xs)] = 0
        i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
        j = np.argmax(xs[:i]) # start of period
        maxdd = xs[i] - xs[j] # max drawdown

        '''
        
        maxdd = self.gen_drawdown_table(self.strategy_ret, top=1)['Net drawdown in %'][0]
        stats.loc['Max Drawdown(%)','Statistics'] = round(maxdd,2) 
        
        VaR = self.value_at_risk(self.strategy_ret)
        stats.loc['Value at Risk (%)','Statistics'] = round(VaR * 252 * 100,2) 
        
        print(stats)
        
        self.plot_cum_returns(self.strategy_ret)
        self.plot_drawdown_periods(self.strategy_ret)
        self.plot_drawdown_underwater(self.strategy_ret)
        self.plot_monthly_returns_heatmap(self.strategy_ret)
        self.plot_annual_returns(self.strategy_ret)
        
################################################################

# run this for the first time
#test=BacktestEngine()
#test.backtest_engine(df_alpha)

# run this after first time
#price = test.price # run this first
#test=BacktestEngine(price)
#test.backtest_engine(df_alpha)

# run it one by one
#test.plot_cum_returns(test.strategy_ret)
#test.plot_drawdown_periods(test.strategy_ret)
#test.plot_drawdown_underwater(test.strategy_ret)
#test.plot_monthly_returns_heatmap(test.strategy_ret)
#test.plot_annual_returns(test.strategy_ret)
