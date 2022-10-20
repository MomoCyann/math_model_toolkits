from __future__ import (absolute_import, division, print_function, unicode_literals)
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import matplotlib
from fontTools.varLib.mutator import percents
import glob
import os

def data_reshape(trade_path, feature_path):
    '''
    此方法是为了让外部数据源满足回测框架的数据格式要求
    :param trade_path:
    :param feature_path:
    :return:
    '''
    #数据读取
    data = pd.read_csv(trade_path)
    feature_data = pd.read_csv(feature_path)
    #改变为时间格式
    data['tradeDate'] = pd.to_datetime(data['tradeDate'])
    feature_data['selectedDate'] = pd.to_datetime(feature_data['selectedDate'])
    #合并准备
    feature_data = feature_data.loc[:, ['selectedDate', 'sentimentFactor']]
    feature_data.columns = ['tradeDate', 'sentimentFactor']
    data = pd.merge(data, feature_data, how='left', on=['tradeDate'])
    df = pd.DataFrame(data=None, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'openinterest',
                                          'sentimentFactor'])
    df['date'] = data['tradeDate']
    df['open'] = data['openPrice']
    df['high'] = data['highestPrice']
    df['low'] = data['lowestPrice']
    df['close'] = data['closePrice']
    df['volume'] = data['turnoverVol']
    df['openinterest'] = 0
    df['sentimentFactor'] = data['sentimentFactor']
    df.set_index('date', inplace=True)
    # df返回一个符合格式的dataframe
    return df

# 继承官方的类进行修改，能在df加入自定义列，我们这里加入的是sentimentFactor
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('sentimentFactor',)  # 要添加的列名
    # 设置 line 在数据源上新增的位置
    params = (
        ('sentimentFactor', -1),  # turnover对应传入数据的列名，这个-1会自动匹配backtrader的数据类与原有pandas文件的列名
        # 如果是个大于等于0的数，比如8，那么backtrader会将原始数据下标8(第9列，下标从0开始)的列认为是turnover这一列
    )

# 继承官方的类进行修改，能买入10%每次，卖出全部。size 是股数
class PercentSizerPlus(bt.sizers.PercentSizer):
    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)
        if not position:
            size = cash_total / data.close[0] * (self.params.percents / 100)
            if size < 100:
                size = 100
        else:
            if isbuy:
            # 如果是买
                size = cash_total / data.close[0] * (self.params.percents / 100)
                if size < 100:
                    size = 100
            else:
            # 如果是卖，卖出全部=position.size
                size = position.size

        if self.p.retint:
            size = int(size)

        return size

def main():
    '''
        1.设置路径trade_path、feature_path
        2.设置情感阈值 cerebro.addstrategy(TestStrategy, senti_pos_threshold=1, senti_neg_threshold=0.3)
        3.设置每次买入百分比 cerebro.addsizer(PercentSizerPlus, percents=10)
        4.设置开始时间 start_date = datetime(2022, 1, 3)  # 回测开始时间
        5.设置起始资金 cerebro.broker.setcash(100000.0)
    '''
    # 读取决策表
    decision = pd.read_csv('data/HS300_50/decision/decision_50_50.csv')
    company_profit = []
    benchmark_profit = []
    all_company_name = decision.loc[:, 'company_name']

    # 参数设置
    cash_total = 1000000.0
    stop_days = 5
    plot = 0
    percents = 10
    # 1是混合 0是均线策略
    strategy = 1
    benchmark_result = 0

    # 初始化
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BaseStrategy)
    # 百分比投资？
    cerebro.addsizer(PercentSizerPlus, percents=percents)
    # 获取数据
    start_date = datetime(2021, 1, 1)  # 回测开始时间
    end_date = datetime(2021, 12, 31)  # 回测结束时间

    # 循环导入数据
    for index in decision.index:
        # 读取公司名字方便匹配文件名
        company_name = decision.loc[index, 'company_name']
        if (company_name == '隆基股份') or (company_name == '苏宁易购') or (company_name == '青岛海尔') or (company_name == '东方财富'):
            company_profit.append(-1)
            benchmark_profit.append(0)
            continue

        trade_path = 'data/HS300/tradeData/'  # + index + '_' + company_name + '_trade_20-22.csv'
        file = os.listdir(trade_path)
        # 模糊匹配文件名→找到tradedata路径
        for f in file:
            if company_name in f:
                trade_path = trade_path + f
        # featuredata
        feature_path = 'data/HS300_50/feature/50_50/' + company_name + '_feature_50.csv'

        # 匹配dataframe格式为回测框架要求格式
        df = data_reshape(trade_path, feature_path)
        # 读取decision的各指标 导入策略
        upper_A, decision_A, upper_B, decision_B = decision.loc[index,
                                                                ['upper_A', 'decision_A', 'upper_B', 'decision_B']]
        df.insert(7, 'upper_A', upper_A)
        df.insert(8, 'decision_A', decision_A)
        df.insert(9, 'upper_B', upper_B)
        df.insert(10, 'decision_B', decision_B)
        print('data reshape finished')
        print('---------------------')
        data = PandasDataPlus(dataname=df, fromdate=start_date, todate=end_date)  # 加载数据
        cerebro.adddata(data)  # 将数据传入回测系统

    # 加入benchmark基准对比 继承notimeframe类表示整个数据持续时间
    cerebro.addobserver(bt.observers.Benchmark, timeframe=bt.TimeFrame.NoTimeFrame)

    cerebro.broker.setcash(cash_total)  # 加到100000元资金
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('最终持有: %.2f' % (cerebro.broker.getvalue()))
    benchmarks = pd.read_csv('benchmark.csv')
    benchmarking = benchmarks.loc[0, 'benchmark']
    print('基准收益率: %.2f' % (benchmarking))
    benchmark_profit.append(benchmarking)
    # 计入收益
    company_profit.append(cerebro.broker.getvalue() / cash_total)  #
    print('策略收益率: %.2f' % ((cerebro.broker.getvalue() / cash_total) - 1))
    print(company_name)


if __name__ == '__main__':
    main()