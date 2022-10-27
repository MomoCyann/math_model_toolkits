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

def data_reshape(trade_path, feature_path):
    '''
    �˷�����Ϊ�����ⲿ����Դ����ز��ܵ����ݸ�ʽҪ��
    :param trade_path:
    :param feature_path:
    :return:
    '''
    #���ݶ�ȡ
    data = pd.read_csv(trade_path)
    feature_data = pd.read_csv(feature_path)
    #�ı�Ϊʱ���ʽ
    data['tradeDate'] = pd.to_datetime(data['tradeDate'])
    feature_data['selectedDate'] = pd.to_datetime(feature_data['selectedDate'])
    #�ϲ�׼��
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
    # df����һ�����ϸ�ʽ��dataframe
    return df

# �̳йٷ���������޸ģ�����df�����Զ����У���������������sentimentFactor
class PandasDataPlus(bt.feeds.PandasData):
    lines = ('sentimentFactor', 'upper_A', 'decision_A', 'upper_B', 'decision_B')  # Ҫ��ӵ�����
    # ���� line ������Դ��������λ��
    params = (
        ('sentimentFactor', -1),
        ('upper_A', -1),
        ('decision_A', -1),
        ('upper_B', -1),
        ('decision_B', -1),# turnover��Ӧ�������ݵ����������-1���Զ�ƥ��backtrader����������ԭ��pandas�ļ�������
        # ����Ǹ����ڵ���0����������8����ôbacktrader�Ὣԭʼ�����±�8(��9�У��±��0��ʼ)������Ϊ��turnover��һ��
    )

# ��������
class NewStrategy(bt.Strategy):
    params = (
        # �������ﲻ�ùܣ���������������������
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
        # ѭ������ÿ����Ʊ�����̼ۣ���������
        # ѭ������upper_A, decision_A, upper_B, decision_B
        # own_days��sell_flagҲҪ����
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
        1.��ÿһ��data���㹺��ʱ�õı�׼��Ȼ����ѭ������ָ��͹�Ʊ����ӽ��ֵ�
            �����Ȩ�ؽϸߣ����������ֵ�Ĳ�ֵ x 1000000
            ���߲���ðٷֱȺ���
            ��֤�����������ǰ�档
        2.ѭ����֮���������

        3.��order�����������top10ָ����Ʊ��������
        '''
        stocks = {}
        for data in self.datas:
            # ÿһ��ѭ����ʼ��ָ��
            flag = 0
            # ��ʱ A�ұ� B��� ��Ӧͬһ�ֲ��ԣ��м��þ��߸���
            if self.decision_A[data._name] == self.decision_B[data._name]:
                # AB����ΪΪ���룬���м�������
                if self.decision_A[data._name] == 1:
                    if (self.datasenti[data._name][0] >= self.upper_A[data._name]):
                        # ���� �����Ȩ�رȽϸ�
                        flag = abs(self.datasenti[data._name][0] - self.upper_A[data._name]) * 1000000
                    elif (self.datasenti[data._name][0] <= self.upper_B[data._name]):
                        # ����
                        flag = abs(self.datasenti[data._name][0] - self.upper_B[data._name]) * 1000000
                elif self.decision_A[data._name] == -1:
                    if self.sma[data._name][-1] < self.lma[data._name][-1] and self.sma[data._name][0] > self.lma[data._name][0]:
                        # ���ھ���
                        # ����
                        flag = (self.sma[data._name][0] - self.lma[data._name][0]) / self.lma[data._name][0]
                # else:
                #     print('�����߲�����')
                #     �����6�ҹ�˾ֻ�õ��˾���
                #
                #     #��ʱdecision_A��decision_B����0�����þ��߾���
                #     if self.sma[data._name][-1] < self.lma[data._name][-1] and self.sma[data._name][0] > self.lma[data._name][0]:
                #         # ���ھ���
                #         # ����
                #         self.log('BUY CREATE, Stock: %s, Price: %.2f' % (data._name, self.dataclose[data._name][0]))
                #         self.order[data._name] = self.buy(data=data)
                #     if self.sell_flag[data._name]:
                #         if self.broker.getposition(data):
                #             if self.sma[data._name][-1]>self.lma[data._name][-1] and self.sma[data._name][0]<self.lma[data._name][0]:
                #                 # С�ھ�����������
                #                 self.log('SELL CREATE, Stock: %s, Price: %.2f' % (data._name, self.dataclose[data._name][0]))
                #                 self.order[data._name] = self.sell(data=data)
                #                 self.own_days[data._name] = 0
                #                 self.sell_flag[data._name] = False

            # decision_A������decision_B
            else:
                if self.decision_A[data._name] == 1:
                    # ����upper_A����
                    if self.datasenti[data._name][0] >= self.upper_A[data._name]:
                        flag = abs(self.datasenti[data._name][0] - self.upper_A[data._name]) * 1000000
                elif self.decision_A[data._name] == -1:
                    if self.decision_B[data._name] == 1:
                        if self.datasenti[data._name][0] <= self.upper_B[data._name]:
                            flag = abs(self.datasenti[data._name][0] - self.upper_B[data._name]) * 1000000
                    else:
                        if self.sma[data._name][-1] < self.lma[data._name][-1] and self.sma[data._name][0] > self.lma[data._name][0]:
                            # ���ھ�������
                            flag = (self.sma[data._name][0] - self.lma[data._name][0]) / self.lma[data._name][0]
                else:
                    if self.decision_B[data._name] == 1:
                        if self.datasenti[data._name][0] <= self.upper_B[data._name]:
                            flag = abs(self.datasenti[data._name][0] - self.upper_B[data._name]) * 1000000
                    else:
                        if self.sma[data._name][-1] < self.lma[data._name][-1] and self.sma[data._name][0] > self.lma[data._name][0]:
                            # ���ھ�������
                            flag = (self.sma[data._name][0] - self.lma[data._name][0]) / self.lma[data._name][0]
            # flag�������,�ѹ�Ʊ��flag�����ֵ䱣����stocks
            # stocks[str(data._name)] = flag
            stocks[data] = flag
        # ��stocks��������
        stocks = dict(sorted(stocks.items(), key=lambda x: x[1], reverse=True))

        # �޳���Ϊ0�ģ���û���źŵġ�
        stocks_no_zero = {}
        for key, value in stocks.items():
            if value != 0:
                stocks_no_zero[key] = value
        # # ��������
        # ranks = {d: i for d, i in zip(stocks, range(1, len(stocks) + 1))}
        # ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=False)

        # �������� ��0
        ranks = {d: i for d, i in zip(stocks_no_zero, range(1, len(stocks_no_zero) + 1))}
        ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=False)

        if self.min_buy_num >= len(ranks):
            self.min_buy_num = len(ranks)
        if len(ranks) >= self.p.selnum:
            # ѡȡǰ self.p.selnum ֻ��Ʊ��Ϊ�ֲֹ�
            rtop = dict(ranks[:self.p.selnum])
            # ʣ���Ʊ���ӳֲ����޳�������ڳֲ���Ļ���
            rbot = dict(ranks[self.p.selnum:])
        else:
            print("�����ǣ�", self.datetime.date(), '�����źŲ��������')
            rtop = dict(ranks)


        # ��ȡ�в�λ�Ĺ�Ʊ
        posdata = [d for d, pos in self.getpositions().items() if pos]

        # ɾ�����ڼ������еĹ�Ʊ�������ͷ��ʽ����������µĹ�Ʊ
        for d in (d for d in posdata if d not in rtop):
            self.log('Leave {}'.format(d._name))
            # self.log('Leave {} - Rank {:.2f}'.format(d._name, rbot[d]))
            self.order_target_percent(d, target=0.0)

        # ����һ�ڼ������еĹ�Ʊ�����в�λ����
        for d in (d for d in posdata if d in rtop):
            self.log('Rebal {} - Rank {:.2f}'.format(d._name, rtop[d]))
            self.order_target_percent(d, target=self.perctarget)
            del rtop[d]

        # ���뵱ǰ�ֲ���û�еĹ�Ʊ
        for d in rtop:
            self.log('Enter {} - Rank {:.2f}'.format(d._name, rtop[d]))
            self.order_target_percent(d, target=self.perctarget)
    def stop(self):
        print('��С��������', self.min_buy_num)
def show_result_empyrical(returns, benchmark_returns):
    import empyrical

    print('�ۼ����棺', empyrical.cum_returns_final(returns))
    print('���س���', empyrical.max_drawdown(returns))
    print('���ձ�', empyrical.sharpe_ratio(returns))
    alpha, beta = empyrical.alpha_beta(np.array(returns), np.array(benchmark_returns))
    print('Alpha', alpha)
    print('�����', empyrical.calmar_ratio(returns))
    print('omega', empyrical.omega_ratio(returns))

if __name__ == '__main__':
    '''
        1.����·��trade_path��feature_path
        4.���ÿ�ʼʱ�� start_date = datetime(2022, 1, 3)  # �ز⿪ʼʱ��
    '''
    # ��ȡ���߱�
    decision = pd.read_csv('data/HS300_55/decision/decision_30_30_55_18-20.csv')
    all_company_name = decision.loc[:, 'company_name']

    # ��������
    cash_total = 1000000.0
    plot = 1
    s_maperiod = 15
    l_maperiod = 60
    # ���й�Ʊ����
    top = 8
    reserve = 0.01

    # ��ʼ��
    cerebro = bt.Cerebro()
    # ��һ������
    cerebro.addstrategy(NewStrategy, s_maperiod=s_maperiod, l_maperiod=l_maperiod, selnum=top, reserve=reserve)
    # Ӷ��
    cerebro.broker.setcommission(commission=0.003)
    # �ز�ʱ��
    start_date = datetime(2020, 10, 10)
    end_date = datetime(2021, 12, 31)

    # ѭ����������
    for index in decision.index:
        # ��ȡ��˾���ַ���ƥ���ļ���
        company_name = decision.loc[index, 'company_name']
        # if (company_name == '¡���ɷ�') or (company_name == '�����׹�') or (company_name == '�ൺ����') or (company_name == '�����Ƹ�'):
        #     company_profit.append(-1)
        #     benchmark_profit.append(0)
        #     continue

        trade_path = 'data/HS300_55/tradeData_55/'  # + index + '_' + company_name + '_trade_20-22.csv'
        file = os.listdir(trade_path)
        # ģ��ƥ���ļ������ҵ�tradedata·��
        for f in file:
            if company_name in f:
                trade_path = trade_path + f
        # featuredata
        feature_path = 'data/HS300_55/feature/30_30_55/' + company_name + '_feature_30.csv'

        # ƥ��dataframe��ʽΪ�ز���Ҫ���ʽ
        df = data_reshape(trade_path, feature_path)
        # ��tiker�����ݵ����������֡�
        ticker = str(df.iloc[0, 7])
        df = df.iloc[:, :7]
        # ��ȡdecision�ĸ�ָ�� �������
        upper_A, decision_A, upper_B, decision_B = decision.loc[index,
                                                                ['upper_A', 'decision_A', 'upper_B', 'decision_B']]
        df.insert(7, 'upper_A', upper_A)
        df.insert(8, 'decision_A', decision_A)
        df.insert(9, 'upper_B', upper_B)
        df.insert(10, 'decision_B', decision_B)
        data = PandasDataPlus(dataname=df, fromdate=start_date, todate=end_date, plot=False)  # ��������
        cerebro.adddata(data, name=ticker)  # �����ݴ���ز�ϵͳ
        print(ticker)
        print('Add Data Completed')
        print('---------------------')

    # ���ñ���
    cerebro.broker.setcash(cash_total)
    print('����: %.2f' % cerebro.broker.getvalue())
    print('Loading����')

    # ����ָ��
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
    show_result_empyrical(returns, benchmark_returns)

    print('���ճ���: %.2f' % (cerebro.broker.getvalue()))
    print('����300��2021��������: -5.2%')

    # �ۼ�����
    # ���س�
    # ���ձ��ʣ���ʾ����ÿ��λ�ܷ����ܹ���õĳ���꣬�ñ���Խ�ߣ�֤�� ���ջر�Խ�󣬸�Ͷ�����Ч��Խ�á�

    # Alpha�����˹�Ʊ�����������г��ĳ������棬���Ի�õ����г����������޹صĻر���
    # ��alpha=0����˵��Ͷ����ϱ�������̻���һ�£�
    # ��alpha<0����˵��Ͷ���� ����ȴ�������Ҫ�Ͷ���������ڷ������Ի�����棻
    # ��alpha>0����˵����Ʊ�� ��ϱ������ڴ��̣�Ͷ����Ͽ��Դ��л�ȡһ���ĳ������档alpha=1%�൱�ڸ���ͬ�� �г�����1%��

    # ������ʱ�ʾͶ����������������س��ı��ʣ�Ҳ�ɳ�Ϊ��λ�س������ʣ�
    # ���Ժ���Ͷ����ϵ�������ձȣ�һ�����ֵԽ��Ͷ����ϱ���Խ��

    if plot == 1:
        # matplotlib.use('QT5Agg')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
        plt.rcParams['axes.unicode_minus'] = False  # ��������Ҫ�ֶ�����
        cerebro.plot(
            #  Format string for the display of ticks on the x axis
            fmt_x_ticks='%Y-%m-%d',

            # Format string for the display of data points values
            fmt_x_data='%Y-%m-%d',
            iplot = False
        )