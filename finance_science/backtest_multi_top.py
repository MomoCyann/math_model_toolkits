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
import quantstats

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
                                          'sentimentFactor', 'ticker'])
    df['date'] = data['tradeDate']
    df['open'] = data['openPrice']
    df['high'] = data['highestPrice']
    df['low'] = data['lowestPrice']
    df['close'] = data['closePrice']
    df['volume'] = data['turnoverVol']
    df['openinterest'] = 0
    df['sentimentFactor'] = data['sentimentFactor']
    df['ticker'] = data['secShortName']

    df.set_index('date', inplace=True)
    # df返回一个符合格式的dataframe
    return df

# 继承官方的类进行修改，能在df加入自定义列，我们这里加入的是sentimentFactor
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('sentimentFactor', 'upper_A', 'decision_A', 'upper_B', 'decision_B')  # 要添加的列名
    # 设置 line 在数据源上新增的位置
    params = (
        ('sentimentFactor', -1),
        ('upper_A', -1),
        ('decision_A', -1),
        ('upper_B', -1),
        ('decision_B', -1),# turnover对应传入数据的列名，这个-1会自动匹配backtrader的数据类与原有pandas文件的列名
        # 如果是个大于等于0的数，比如8，那么backtrader会将原始数据下标8(第9列，下标从0开始)的列认为是turnover这一列
    )

# 策略配置
class NewStrategy(bt.Strategy):
    params = (
        # 参数这里不用管，可以在主函数进行设置
        ('selnum', 10),
        ('s_maperiod', 10),
        ('l_maperiod', 60),
        ('reserve', 0.05),
    )

    def log(self, arg):
        print('{} {}'.format(self.datetime.date(), arg))

    def __init__(self):
        self.dataclose = dict()
        self.datasenti = dict()

        self.upper_A = dict()
        self.decision_A = dict()
        self.upper_B = dict()
        self.decision_B = dict()

        self.order = dict()
        self.buyprice = dict()
        self.buycomm = dict()
        self.sma = dict()
        self.lma = dict()

        self.own_days = dict()
        self.sell_flag = dict()
        # 循环保存每个股票的收盘价，情绪因子
        # 循环保存upper_A, decision_A, upper_B, decision_B
        # own_days和sell_flag也要保存
        for data in self.datas:
            self.dataclose[data._name] = data.close
            self.datasenti[data._name] = data.lines.sentimentFactor

            self.upper_A[data._name] = data.lines.upper_A
            self.decision_A[data._name] = data.lines.decision_A
            self.upper_B[data._name] = data.lines.upper_B
            self.decision_B[data._name] = data.lines.decision_B

            self.order[data._name] = None
            self.buyprice[data._name] = None
            self.buycomm[data._name] = None
            self.sma[data._name] = bt.indicators.SMA(data, period=self.params.s_maperiod)
            self.lma[data._name] = bt.indicators.SMA(data, period=self.params.l_maperiod)

            self.own_days[data._name] = 0
            self.sell_flag[data._name] = False
        print('init completed')
        # self.mystats = pd.DataFrame(data=None, columns=['benchmark'])
        self.perctarget = (1.0 - self.p.reserve) / self.p.selnum
        self.min_buy_num = 1000

    def next(self):
        '''
        1.先每一个data计算购买时用的标准，然后在循环最后把指标和股票名添加进字典
            舆情的权重较高，用舆情和阈值的差值 x 1000000
            均线差距用百分比衡量
            保证舆情的排序在前面。
        2.循环完之后进行排序

        3.用order命令对排序后的top10指定股票做出操作
        '''
        stocks = {}
        for data in self.datas:
            # 每一次循环初始化指标
            flag = 0
            # 此时 A右边 B左边 对应同一种策略，中间用均线辅助
            if self.decision_A[data._name] == self.decision_B[data._name]:
                # AB的行为为买入，则中间是卖出
                if self.decision_A[data._name] == 1:
                    if (self.datasenti[data._name][0] >= self.upper_A[data._name]):
                        # 买入 舆情的权重比较高
                        flag = abs(self.datasenti[data._name][0] - self.upper_A[data._name]) * 1000000
                    elif (self.datasenti[data._name][0] <= self.upper_B[data._name]):
                        # 买入
                        flag = abs(self.datasenti[data._name][0] - self.upper_B[data._name]) * 1000000
                elif self.decision_A[data._name] == -1:
                    if self.sma[data._name][-1] < self.lma[data._name][-1] and self.sma[data._name][0] > self.lma[data._name][0]:
                        # 大于均线
                        # 买入
                        flag = (self.sma[data._name][0] - self.lma[data._name][0]) / self.lma[data._name][0]
                # else:
                #     print('纯均线不买不卖')
                #     大概有6家公司只用到了均线
                #
                #     #此时decision_A和decision_B都是0，采用均线决策
                #     if self.sma[data._name][-1] < self.lma[data._name][-1] and self.sma[data._name][0] > self.lma[data._name][0]:
                #         # 大于均线
                #         # 买入
                #         self.log('BUY CREATE, Stock: %s, Price: %.2f' % (data._name, self.dataclose[data._name][0]))
                #         self.order[data._name] = self.buy(data=data)
                #     if self.sell_flag[data._name]:
                #         if self.broker.getposition(data):
                #             if self.sma[data._name][-1]>self.lma[data._name][-1] and self.sma[data._name][0]<self.lma[data._name][0]:
                #                 # 小于均线卖卖卖！
                #                 self.log('SELL CREATE, Stock: %s, Price: %.2f' % (data._name, self.dataclose[data._name][0]))
                #                 self.order[data._name] = self.sell(data=data)
                #                 self.own_days[data._name] = 0
                #                 self.sell_flag[data._name] = False

            # decision_A不等于decision_B
            else:
                if self.decision_A[data._name] == 1:
                    # 大于upper_A买入
                    if self.datasenti[data._name][0] >= self.upper_A[data._name]:
                        flag = abs(self.datasenti[data._name][0] - self.upper_A[data._name]) * 1000000
                elif self.decision_A[data._name] == -1:
                    if self.decision_B[data._name] == 1:
                        if self.datasenti[data._name][0] <= self.upper_B[data._name]:
                            flag = abs(self.datasenti[data._name][0] - self.upper_B[data._name]) * 1000000
                    else:
                        if self.sma[data._name][-1] < self.lma[data._name][-1] and self.sma[data._name][0] > self.lma[data._name][0]:
                            # 大于均线买入
                            flag = (self.sma[data._name][0] - self.lma[data._name][0]) / self.lma[data._name][0]
                else:
                    if self.decision_B[data._name] == 1:
                        if self.datasenti[data._name][0] <= self.upper_B[data._name]:
                            flag = abs(self.datasenti[data._name][0] - self.upper_B[data._name]) * 1000000
                    else:
                        if self.sma[data._name][-1] < self.lma[data._name][-1] and self.sma[data._name][0] > self.lma[data._name][0]:
                            # 大于均线买入
                            flag = (self.sma[data._name][0] - self.lma[data._name][0]) / self.lma[data._name][0]
            # flag计算完毕,把股票和flag做成字典保存在stocks
            # stocks[str(data._name)] = flag
            stocks[data] = flag
        # 对stocks进行排序
        stocks = dict(sorted(stocks.items(), key=lambda x: x[1], reverse=True))

        # 剔除掉为0的，即没有信号的。
        stocks_no_zero = {}
        for key, value in stocks.items():
            if value != 0:
                stocks_no_zero[key] = value
        # # 创建排名
        # ranks = {d: i for d, i in zip(stocks, range(1, len(stocks) + 1))}
        # ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=False)

        # 创建排名 非0
        ranks = {d: i for d, i in zip(stocks_no_zero, range(1, len(stocks_no_zero) + 1))}
        ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=False)

        if self.min_buy_num >= len(ranks):
            self.min_buy_num = len(ranks)
        if len(ranks) >= self.p.selnum:
            # 选取前 self.p.selnum 只股票作为持仓股
            rtop = dict(ranks[:self.p.selnum])
            # 剩余股票将从持仓中剔除（如果在持仓里的话）
            rbot = dict(ranks[self.p.selnum:])
        else:
            print("今天是：", self.datetime.date(), '买入信号不足持有量')
            rtop = dict(ranks)


        # 提取有仓位的股票
        posdata = [d for d, pos in self.getpositions().items() if pos]

        # 删除不在继续持有的股票，进而释放资金用于买入新的股票
        for d in (d for d in posdata if d not in rtop):
            self.log('Leave {}'.format(d._name))
            # self.log('Leave {} - Rank {:.2f}'.format(d._name, rbot[d]))
            self.order_target_percent(d, target=0.0)

        # 对下一期继续持有的股票，进行仓位调整
        for d in (d for d in posdata if d in rtop):
            self.log('Rebal {} - Rank {:.2f}'.format(d._name, rtop[d]))
            self.order_target_percent(d, target=self.perctarget)
            del rtop[d]

        # 买入当前持仓中没有的股票
        for d in rtop:
            self.log('Enter {} - Rank {:.2f}'.format(d._name, rtop[d]))
            self.order_target_percent(d, target=self.perctarget)
    def stop(self):
        print('最小购买量：', self.min_buy_num)
def show_result_empyrical(returns, benchmark_returns):
    import empyrical

    print('累计收益：', empyrical.cum_returns_final(returns))
    print('最大回撤：', empyrical.max_drawdown(returns))
    print('夏普比', empyrical.sharpe_ratio(returns))
    alpha, beta = empyrical.alpha_beta(np.array(returns), np.array(benchmark_returns))
    print('Alpha', alpha)
    print('卡玛比', empyrical.calmar_ratio(returns))
    print('omega', empyrical.omega_ratio(returns))

if __name__ == '__main__':
    '''
        1.设置路径trade_path、feature_path
        4.设置开始时间 start_date = datetime(2022, 1, 3)  # 回测开始时间
    '''
    # 读取决策表
    decision = pd.read_csv('data/HS300_55/decision/decision_30_30_55_18-20.csv')
    all_company_name = decision.loc[:, 'company_name']

    # 参数设置
    cash_total = 1000000.0
    plot = 1
    s_maperiod = 15
    l_maperiod = 60
    # 持有股票数量
    top = 6
    reserve = 0.01

    # 初始化
    cerebro = bt.Cerebro()
    # 加一个策略
    cerebro.addstrategy(NewStrategy, s_maperiod=s_maperiod, l_maperiod=l_maperiod, selnum=top, reserve=reserve)
    # 佣金
    cerebro.broker.setcommission(commission=0.003)
    # 回测时间
    start_date = datetime(2020, 10, 10)
    end_date = datetime(2021, 12, 31)

    # 循环导入数据
    for index in decision.index:
        # 读取公司名字方便匹配文件名
        company_name = decision.loc[index, 'company_name']
        # if (company_name == '隆基股份') or (company_name == '苏宁易购') or (company_name == '青岛海尔') or (company_name == '东方财富'):
        #     company_profit.append(-1)
        #     benchmark_profit.append(0)
        #     continue

        trade_path = 'data/HS300_55/tradeData_55/'  # + index + '_' + company_name + '_trade_20-22.csv'
        file = os.listdir(trade_path)
        # 模糊匹配文件名→找到tradedata路径
        for f in file:
            if company_name in f:
                trade_path = trade_path + f
        # featuredata
        feature_path = 'data/HS300_55/feature/30_30_55/' + company_name + '_feature_30.csv'

        # 匹配dataframe格式为回测框架要求格式
        df = data_reshape(trade_path, feature_path)
        # 把tiker的数据导出来做名字。
        ticker = str(df.iloc[0, 7])
        df = df.iloc[:, :7]
        # 读取decision的各指标 导入策略
        upper_A, decision_A, upper_B, decision_B = decision.loc[index,
                                                                ['upper_A', 'decision_A', 'upper_B', 'decision_B']]
        df.insert(7, 'upper_A', upper_A)
        df.insert(8, 'decision_A', decision_A)
        df.insert(9, 'upper_B', upper_B)
        df.insert(10, 'decision_B', decision_B)
        data = PandasDataPlus(dataname=df, fromdate=start_date, todate=end_date, plot=False)  # 加载数据
        cerebro.adddata(data, name=ticker)  # 将数据传入回测系统
        print(ticker)
        print('Add Data Completed')
        print('---------------------')

    # 设置本金
    cerebro.broker.setcash(cash_total)
    print('本金: %.2f' % cerebro.broker.getvalue())
    print('Loading……')

    # 风险指标
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    results = cerebro.run()
    strats = results[0]
    pyfoliozer = strats.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    # load benchmark
    benchmark = pd.read_csv('hs300_returns_shift.csv', header=0, index_col=0)
    benchmark.columns = ['0']
    benchmark_returns = benchmark.iloc[:, 0]
    returns = returns.iloc[58:]
    # 储存return序列
    returns.to_csv('Senti_returns.csv')
    show_result_empyrical(returns, benchmark_returns)

    returns.index = returns.index.tz_convert(None)
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
    benchmark_returns.index = benchmark_returns.index.tz_convert(None)

    quantstats.reports.html(returns, benchmark=benchmark_returns, output='stats.html', title='Sentiment')
    print('最终持有: %.2f' % (cerebro.broker.getvalue()))
    print('沪深300，2021年收益率: -5.2%')

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
            iplot = False
        )