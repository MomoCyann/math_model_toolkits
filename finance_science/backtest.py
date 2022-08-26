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

# 继承官方的类进行修改，能买入10%每次，卖出全部。
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

# 策略配置
class NewStrategy(bt.Strategy):
    params = (
        # 参数这里不用管，可以在主函数进行设置
        ('senti_pos_threshold', 0),
        ('senti_neg_threshold', 0),
        ('upper_A', 0),
        ('decision_A', 0),
        ('upper_B', 0),
        ('decision_B', 0),
        ('maperiod', 15),
        ('stop_days', 30),
        ('own_days', 0),
        ('sell_flag', False)
    )

    def log(self, txt, dt=None):
        ''' 记录策略信息'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.datasenti = self.datas[0].sentimentFactor
        self.mystats = pd.DataFrame(data=None, columns=['benchmark'])

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    # 这个类是用来打印买卖信息的
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):  # 交易执行后，在这里处理
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))  # 记录下盈利数据。

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        if self.params.own_days >= self.params.stop_days:
            self.params.sell_flag = True
        # 此时 A右边 B左边 对应同一种策略，中间用均线辅助
        if self.params.decision_A == self.params.decision_B:
            # AB的行为为买入，则中间是卖出
            if self.params.decision_A == 1:
                if (self.datasenti[0] >= self.params.upper_A) or (self.datasenti[0] <= self.params.upper_B):
                    # 买入
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])
                    self.order = self.buy()
                if self.params.sell_flag:
                    if self.position:
                        if self.dataclose[0] <= self.sma[0]:
                            # 小于均线卖卖卖！
                            self.log('SELL CREATE, %.2f' % self.dataclose[0])
                            self.order = self.sell()
                            self.params.own_days = 0
                            self.params.sell_flag = False
            elif self.params.decision_A == -1:
                if self.dataclose[0] >= self.sma[0]:
                    # 大于均线
                    # 买入
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])
                    self.order = self.buy()
                if self.params.sell_flag:
                    if self.position:
                        # A右边 B左边 卖出
                        if (self.datasenti[0] >= self.params.upper_A) or (self.datasenti[0] <= self.params.upper_B):
                            self.log('SELL CREATE, %.2f' % self.dataclose[0])
                            self.order = self.sell()
                            self.params.own_days = 0
                            self.params.sell_flag = False
            else:
                print('纯均线不买不卖')
                # #此时decision_A和decision_B都是0，采用均线决策
                # if self.dataclose[0] >= self.sma[0]:
                #     # 大于均线
                #     # 买入
                #     self.log('BUY CREATE, %.2f' % self.dataclose[0])
                #     self.order = self.buy()
                # if self.params.sell_flag:
                #     if self.position:
                #         if self.dataclose[0] <= self.sma[0]:
                #             # 小于均线卖卖卖！
                #             self.log('SELL CREATE, %.2f' % self.dataclose[0])
                #             self.order = self.sell()
                #             self.params.own_days = 0
                #             self.params.sell_flag = False
        # decision_A不等于decision_B
        else:
            if self.params.decision_A == 1:
                # 大于upper_A买入
                if self.datasenti[0] >= self.params.upper_A:
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])
                    self.order = self.buy()
                if self.params.sell_flag:
                    if self.position:
                        if self.params.decision_B == -1:
                            # 小于upper_B卖出
                            if self.datasenti[0] <= self.params.upper_B:
                                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                                self.order = self.sell()
                                self.params.own_days = 0
                                self.params.sell_flag = False
                        else:
                            # 均线
                            if self.dataclose[0] <= self.sma[0]:
                                # 小于均线卖卖卖！
                                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                                self.order = self.sell()
                                self.params.own_days = 0
                                self.params.sell_flag = False

            elif self.params.decision_A == -1:
                if self.params.decision_B == 1:
                    if self.datasenti[0] <= self.params.upper_B:
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])
                        self.order = self.buy()
                else:
                    if self.dataclose[0] >= self.sma[0]:
                        # 大于均线买入
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])
                        self.order = self.buy()
                if self.params.sell_flag:
                    if self.position:
                        if self.datasenti[0] >= self.params.upper_A:
                            self.log('SELL CREATE, %.2f' % self.dataclose[0])
                            self.order = self.sell()
                            self.params.own_days = 0
                            self.params.sell_flag = False

            else:
                if self.params.decision_B == 1:
                    if self.datasenti[0] <= self.params.upper_B:
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])
                        self.order = self.buy()
                    if self.params.sell_flag:
                        if self.position:
                            if self.dataclose[0] <= self.sma[0]:
                                # 小于均线卖卖卖！
                                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                                self.order = self.sell()
                                self.params.own_days = 0
                                self.params.sell_flag = False
                else:
                    if self.dataclose[0] >= self.sma[0]:
                        # 大于均线买入
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])
                        self.order = self.buy()
                    if self.params.sell_flag:
                        if self.position:
                            # 小于upper_B卖出
                            if self.datasenti[0] <= self.params.upper_B:
                                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                                self.order = self.sell()
                                self.params.own_days = 0
                                self.params.sell_flag = False
        if self.position:
            self.params.own_days += 1

    def stop(self):
        benchmark_data = []
        benchmark_data.append(self.stats.benchmark.benchmark[0])
        self.mystats = pd.DataFrame(benchmark_data, columns=['benchmark'])
        self.mystats.to_csv('benchmark.csv')


# 策略配置
class BaseStrategy(bt.Strategy):
    params = (
        # 参数这里不用管，可以在主函数进行设置
        ('maperiod', 15),
        ('senti_pos_threshold', 1.4),
        ('senti_neg_threshold', 0.4)
    )

    def log(self, txt, dt=None):
        ''' 记录策略信息'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.datasenti = self.datas[0].sentimentFactor

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

    # 这个类是用来打印买卖信息的
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):  # 交易执行后，在这里处理
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))  # 记录下盈利数据。

    def next(self):
        # Simply log the closing price of the series from the reference
        # self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # 舆情因子大于阈值就买
        # print(self.datasenti[0])
        if self.dataclose[0] >= self.sma[0]:
            # 大于均线买入
            self.log('BUY CREATE, %.2f' % self.dataclose[0])
            self.order = self.buy()

        if self.position:

            if self.dataclose[0] < self.sma[0]:
                # 大于均线卖卖卖！
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


if __name__ == '__main__':
    '''
    1.设置路径trade_path、feature_path
    2.设置情感阈值 cerebro.addstrategy(TestStrategy, senti_pos_threshold=1, senti_neg_threshold=0.3)
    3.设置每次买入百分比 cerebro.addsizer(PercentSizerPlus, percents=10)
    4.设置开始时间 start_date = datetime(2022, 1, 3)  # 回测开始时间
    5.设置起始资金 cerebro.broker.setcash(100000.0)  
    '''
    # 读取决策表
    decision = pd.read_csv('data/HS300_50/decision/decision_50_50_18-20(1).csv')
    # 储存每个公司的收益
    company_profit = []
    benchmark_profit = []
    all_company_name = decision.loc[:, 'company_name']
    # 循环回测每个公司的收益
    cash_total = 100000.0
    stop_days = 5
    plot=0
    percents = 10
    # 1是混合 0是均线策略
    strategy=1
    benchmark_result = 0
    for index in decision.index:
        # 读取公司名字方便匹配文件名
        company_name = decision.loc[index, 'company_name']
        if (company_name == '隆基股份') or (company_name == '苏宁易购') or (company_name == '青岛海尔') or (company_name == '东方财富'):
            company_profit.append(-1)
            benchmark_profit.append(0)
            continue
        trade_path = 'data/HS300/tradeData/'   #  + index + '_' + company_name + '_trade_20-22.csv'
        file = os.listdir(trade_path)
        #模糊匹配文件名→找到tradedata路径
        for f in file:
            if company_name in f:
                trade_path = trade_path + f
        # featuredata
        feature_path = 'data/HS300_50/feature/50_50/' + company_name + '_feature_50.csv'
        # feature_path = 'data/HS300_50/feature/50_50/' + company_name + '_feature_50.csv'
        # 匹配dataframe格式为回测框架要求格式
        df = data_reshape(trade_path, feature_path)
        print('data reshape finished')
        print('---------------------')
        # 回测开始
        cerebro = bt.Cerebro()

        # 读取decision的各指标 导入策略
        upper_A, decision_A, upper_B, decision_B = decision.loc[index,
                                    ['upper_A','decision_A','upper_B','decision_B']]
        # 加一个策略
        if strategy == 1:
            cerebro.addstrategy(NewStrategy, upper_A=upper_A, decision_A=decision_A, upper_B=upper_B, decision_B=decision_B,
                                stop_days=stop_days)
        else:
            cerebro.addstrategy(BaseStrategy)
        # # 多策略
        # strats = cerebro.optstrategy(
        #     TestStrategy,
        #     maperiod=range(10, 21))

        # (国内1手是100股，最小的交易单位)
        # cerebro.addsizer(bt.sizers.FixedSize, stake=300)
        # 百分比投资？
        cerebro.addsizer(PercentSizerPlus, percents=percents)
        # # 梭哈
        # cerebro.addsizer(bt.sizers.PercentSizer, percents=90)

        # 获取数据
        start_date = datetime(2021, 1, 1)  # 回测开始时间
        end_date = datetime(2021, 12, 31)  # 回测结束时间
        data = PandasDataPlus(dataname=df, fromdate=start_date, todate=end_date)  # 加载数据
        cerebro.adddata(data)  # 将数据传入回测系统

        # 加入benchmark基准对比 继承notimeframe类表示整个数据持续时间
        cerebro.addobserver(bt.observers.Benchmark, timeframe=bt.TimeFrame.NoTimeFrame)

        cerebro.broker.setcash(cash_total)  # 加到100000元资金
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        cerebro.run()
        print('最终持有: %.2f' % (cerebro.broker.getvalue()))
        benchmarks = pd.read_csv('benchmark.csv')
        benchmarking = benchmarks.loc[0,'benchmark']
        print('基准收益率: %.2f' % (benchmarking))
        benchmark_profit.append(benchmarking)
        # 计入收益
        company_profit.append(cerebro.broker.getvalue()/cash_total) #
        print('策略收益率: %.2f' % ((cerebro.broker.getvalue()/cash_total)-1))
        print(company_name)
        if plot==1:
            cerebro.plot(
                #  Format string for the display of ticks on the x axis
                fmt_x_ticks='%Y-%m-%d',

                # Format string for the display of data points values
                fmt_x_data='%Y-%m-%d'
            )

    all_company_name = np.array(decision.loc[:, 'company_name'])
    benchmark_profit_new = []
    for i in benchmark_profit:
        benchmark_profit_new.append(i+1)

    results = pd.DataFrame({'公司名称': all_company_name, '收益率': company_profit, '基准收益': benchmark_profit_new})
    if strategy==1:
        strategy_name = 'mix'
    else:
        strategy_name = 'ma'
    results.to_csv(strategy_name + '_results_' + str(stop_days) + '.csv')

    company_profit = list(filter(lambda x:x!=1,company_profit))
    company_profit = list(filter(lambda x: x != -1, company_profit))
    benchmark_profit_new = list(filter(lambda x: x != 1, benchmark_profit_new))
    print('平均收益率为：' + str(np.mean(company_profit)))
    print('最大收益率为：' + str(max(company_profit)))
    print('最小收益率为：' + str(min(company_profit)))
    print('平均基准收益率为：' + str(np.mean(benchmark_profit_new)))
    print('最大基准收益率为：' + str(max(benchmark_profit_new)))
    print('最小基准收益率为：' + str(min(benchmark_profit_new)))

    # trade_path = 'data/HS300/tradeData/100_欣旺达_trade_20-22.csv'
    # feature_path = 'data/HS300/feature/30_30/欣旺达_feature_30.csv'
    # df = data_reshape(trade_path, feature_path)
    # print('data reshape finished')
    # print('---------------------')
    #
    # cerebro = bt.Cerebro()
    #
    # # 加一个策略
    # # cerebro.addstrategy(BaseStrategy, senti_pos_threshold=1.9, senti_neg_threshold=1.6)
    # cerebro.addstrategy(MAStrategy, maperiod=15)
    # # # 多策略
    # # strats = cerebro.optstrategy(
    # #     TestStrategy,
    # #     maperiod=range(10, 21))
    #
    # # (国内1手是100股，最小的交易单位)
    # # cerebro.addsizer(bt.sizers.FixedSize, stake=300)
    # # 百分比投资？
    # cerebro.addsizer(PercentSizerPlus, percents=90)
    # # 梭哈
    # cerebro.addsizer(bt.sizers.PercentSizer, percents=90)
    #
    # # 获取数据
    # start_date = datetime(2020, 10, 19)  # 回测开始时间
    # end_date = datetime(2022, 7, 29)  # 回测结束时间
    # data = PandasDataPlus(dataname=df, fromdate=start_date, todate=end_date)  # 加载数据
    # cerebro.adddata(data)  # 将数据传入回测系统
    #
    # # 加入benchmark基准对比 继承notimeframe类表示整个数据持续时间
    # cerebro.addobserver(bt.observers.Benchmark, timeframe=bt.TimeFrame.NoTimeFrame)
    # cerebro.addobserver(bt.observers.DrawDown)
    #
    # cerebro.broker.setcash(100000.0)  # 加到100000元资金
    # print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # cerebro.run()
    # print('最终持有: %.2f' % (cerebro.broker.getvalue()))
    #
    # cerebro.plot(
    #     #  Format string for the display of ticks on the x axis
    #     fmt_x_ticks='%Y-%m-%d',
    #
    #     # Format string for the display of data points values
    #     fmt_x_data='%Y-%m-%d'
    # )

