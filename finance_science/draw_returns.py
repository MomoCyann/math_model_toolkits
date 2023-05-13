import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
import matplotlib.dates as mdates
def date_convert(data):
    data = pd.DataFrame(data)
    data['time_date'] = pd.to_datetime(data.index, utc=True)
    data.set_index('time_date', inplace=True)
    data.index = data.index.date
    return data

if __name__ == '__main__':
    # 读取数据
    # 基准
    benchmark = pd.read_csv('/returns_data/hs300_returns_shift.csv', header=0, index_col=0)
    benchmark.columns = ['0']
    benchmark_returns = benchmark.iloc[1:, 0]
    benchmark_returns_cumulative = (benchmark_returns + 1).cumprod()
    benchmark_returns_cumulative = date_convert(benchmark_returns_cumulative)

    # 双均线
    ma = pd.read_csv('/returns_data/MA_returns.csv', header=0, index_col=0)
    ma.columns = ['0']
    ma_returns = ma.iloc[1:, 0]
    ma_returns_cumulative = (ma_returns + 1).cumprod()
    ma_returns_cumulative = date_convert(ma_returns_cumulative)
    # 舆情
    senti = pd.read_csv('/returns_data/Senti_returns.csv', header=0, index_col=0)
    senti.columns = ['0']
    senti_returns = senti.iloc[1:, 0]
    senti_returns_cumulative = (senti_returns + 1).cumprod()
    senti_returns_cumulative = date_convert(senti_returns_cumulative)
    # plot
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.style.use('seaborn')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # 绘制累计收益曲线
    plt.plot(benchmark_returns_cumulative, color='#FF6666', label='沪深300')
    plt.plot(ma_returns_cumulative, color='#F1C40F', label='双均线策略')
    plt.plot(senti_returns_cumulative, color='#99CCFF', label='舆情策略')

    # 虚线对齐
    benchmark_result = benchmark_returns_cumulative.iloc[-1, 0]
    ma_result = ma_returns_cumulative.iloc[-1, 0]
    senti_result = senti_returns_cumulative.iloc[-1, 0]
    plt.axhline(benchmark_result, color='#FF6666', linestyle='--', clip_on=False, xmax=0.955)
    plt.axhline(ma_result, color='#F1C40F', linestyle='--', clip_on=False, xmax=0.955)
    plt.axhline(senti_result, color='#99CCFF', linestyle='--', clip_on=False, xmax=0.955)

    plt.annotate(str(benchmark_result),xy = (2,benchmark_result), xytext = (2,benchmark_result))

    plt.grid(True)
    plt.legend(fontsize=16)
    plt.xlabel('日期', fontsize=16)
    plt.ylabel('累计收益率', fontsize=16)
    plt.tick_params(labelsize=16)
    # x轴显示月份
    months = mdates.MonthLocator()
    plt.gca().xaxis.set_major_locator(months)

    plt.show()

    print('yes')