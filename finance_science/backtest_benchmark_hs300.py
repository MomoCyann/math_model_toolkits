from __future__ import (absolute_import, division, print_function, unicode_literals)
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt
from fontTools.varLib.mutator import percents
import glob
import os
import pyfolio as pf

def data_reshape(data):
    #改变为时间格式
    data['tradeDate'] = pd.to_datetime(data['tradeDate'])
    df = pd.DataFrame(data=None, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'openinterest',
                                          'sentimentFactor', 'ticker'])
    df['date'] = data['tradeDate']
    df['open'] = data['openIndex']
    df['high'] = data['highestIndex']
    df['low'] = data['lowestIndex']
    df['close'] = data['closeIndex']
    df['volume'] = data['turnoverVol']
    df['openinterest'] = 0
    df['ticker'] = data['secShortName']

    df.set_index('date', inplace=True)
    # df返回一个符合格式的dataframe
    return df

# 策略配置
class NewStrategy(bt.Strategy):
    params = (
        # 参数这里不用管，可以在主函数进行设置
    )

    def __init__(self):
        print('init completed')

    def prenext(self):
        self.next()
    def next(self):
        self.buy()

    def stop(self):
        # benchmark_data = []
        # benchmark_data.append(self.stats.benchmark.benchmark[0])
        # self.mystats = pd.DataFrame(benchmark_data, columns=['benchmark'])
        # self.mystats.to_csv('benchmark.csv')
        return

def show_result_empyrical(returns):
    import empyrical

    print('累计收益：', empyrical.cum_returns_final(returns))
    print('最大回撤：', empyrical.max_drawdown(returns))
    print('夏普比', empyrical.sharpe_ratio(returns))
    # alpha, beta = empyrical.alpha_beta(returns, benchmark_returns)
    # print('Alpha', alpha)
    print('卡玛比', empyrical.calmar_ratio(returns))
    print('omega', empyrical.omega_ratio(returns))

if __name__ == '__main__':
    '''
            1.设置路径trade_path、feature_path
            2.设置情感阈值 cerebro.addstrategy(TestStrategy, senti_pos_threshold=1, senti_neg_threshold=0.3)
            3.设置每次买入百分比 cerebro.addsizer(PercentSizerPlus, percents=10)
            4.设置开始时间 start_date = datetime(2022, 1, 3)  # 回测开始时间
            5.设置起始资金 cerebro.broker.setcash(100000.0)
        '''
    # 读取
    data = pd.read_csv('data/HS300_55/HS300_TradeDate.csv')

    # 参数设置
    cash_total = 52129313
    plot = 1

    # 初始化
    cerebro = bt.Cerebro()
    # 加一个策略
    cerebro.addstrategy(NewStrategy)

    # 回测时间
    start_date = datetime(2020, 12, 31)
    end_date = datetime(2021, 12, 31)

    # 导入数据
    # 匹配dataframe格式为回测框架要求格式
    df = data_reshape(data)
    # 把tiker的数据导出来做名字。
    ticker = str(df.iloc[0, 7])
    df = df.iloc[:, :7]
    data = bt.feeds.PandasData(dataname=df, fromdate=start_date, todate=end_date)  # 加载数据
    cerebro.adddata(data, name=ticker)  # 将数据传入回测系统
    print(ticker)
    print('Add Data Completed')
    print('---------------------')

    # 百分比投资？
    cerebro.addsizer(bt.sizers.FixedSize, stake=10000)
    # 设置本金
    cerebro.broker.setcash(cash_total)
    print('本金: %.2f' % cerebro.broker.getvalue())
    print('Loading……')

    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    results = cerebro.run()
    strats = results[0]
    pyfoliozer = strats.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    show_result_empyrical(returns)
    returns.to_csv('hs300_returns.csv')
    print('最终持有: %.2f' % (cerebro.broker.getvalue()))

    # 累计收益
    # 最大回撤
    # 夏普比率：表示承受每单位总风险能够获得的超额报酬，该比率越高，证明 风险回报越大，该投资组合效果越好。

    # Alpha衡量了股票或组合相对于市场的超额收益，可以获得的与市场波动部分无关的回报。
    # 若alpha=0，则说明投资组合表现与大盘基本一致；
    # 若alpha<0，则说明投资组 合相比大盘收益要差，投资组合相对于风险难以获得收益；
    # 若alpha>0，则说明股票或 组合表现优于大盘，投资组合可以从中获取一定的超额收益。alpha=1%相当于高于同期 市场收益1%。

    # 卡玛比率表示投资组合收益率与最大回撤的比率，也可称为单位回撤收益率，
    # 可以衡量投资组合的收益风险比，一般该数值越大，投资组合表现越好

    if plot == 1:
        # matplotlib.use('QT5Agg')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
        cerebro.plot(
            #  Format string for the display of ticks on the x axis
            fmt_x_ticks='%Y-%m-%d',

            # Format string for the display of data points values
            fmt_x_data='%Y-%m-%d',
            iplot=False
        )