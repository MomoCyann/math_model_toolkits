import math
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
from matplotlib import ticker
import io
from PIL import Image
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''公式'''
# 自然降温 ut=0
'''t = t*math.exp(-1/C*p*V*R) + (Tair[i]) * (1-math.exp(-1/C*p*V*R))'''
# 加热 ut=1
'''t = t*math.exp(-1/C*p*V*R) + (n*P*ut*R + Tair[i]) * (1-math.exp(-1/C*p*V*R))'''


'''条件'''
C=4200
p=1000
V=0.06
# R=1.22866894197952
F=1.08
K=0.879
R=1/(F*K)
n=0.98
P=1500
Tsumm=45
Twint=60

'''室外温度1440个'''
data = pd.read_csv('t1440.csv', header=None)
# 通过插值，每分钟插60个
Tair_sum = data.iloc[:, 0]
Tair_win = data.iloc[:, 1]
data.reset_index(inplace=True)
data['time']=pd.TimedeltaIndex(data['index'], unit='m')
data['minutes']=0
for index, rows in data.iterrows():
    time = str(rows['time'])
    time = time[7:]
    time = time[:-3]
    data['minutes'][index]=time
# plt插值
date = data['minutes']
x_name = '时间'
y_name = '温度 / °C'
title = '冬夏季一日代表性室内温度变化'
# 每类数据依次绘图散点
# dpi?
plt.figure(figsize=(16, 10))

# plt.scatter(date, tair, s=10, color='deepskyblue')
plt.plot(date, Tair_win, color='deepskyblue', label='冬季')
plt.plot(date, Tair_sum, color='salmon', label='夏季')
# 坐标轴
plt.xlabel(x_name, fontsize=16)
plt.ylabel(y_name, fontsize=16)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(60))
plt.gcf().autofmt_xdate()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title(title, fontsize=22)
plt.legend(fontsize=16, loc='upper right')

plt.savefig(title + '.png')
# Save the image in memory in PNG format
png1 = io.BytesIO()
plt.savefig(png1, format="png", dpi=100, pad_inches=.1, bbox_inches='tight')

# Load this image into PIL
png2 = Image.open(png1)

# Save as TIFF
png2.save(title + ".tiff")
png1.close()

def heat(t, tair, ut):
    t = t * math.exp(-60 / (C * p * V * R)) + (n * P * ut * R + tair) * (1 - math.exp(-60 / (C * p * V * R)))
    return t

def cooldown(t, tair):
    t = t * math.exp(-60 / (C * p * V * R)) + tair * (1 - math.exp(-60 / (C * p * V * R)))
    return t

def summer_open():
    # 储存温度
    t_moment = []
    # 初始温度
    # t=45
    t=20
    ut=0
    heat_sw=False

    tloop=0
    for i in range(0,1440):
        # 先检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t>=Tsumm:
                # 到达规定温度 停止加热
                ut=0
                heat_sw=False
                # 自然降温
                t = cooldown(t, Tair_sum[i])
            else:
                # 继续加热
                # t = t * math.exp(-60 / (C * p * V * R)) + (n * P * ut * R + Tair_win[i]) * (1 - math.exp(-60 / (C * p * V * R)))
                t = heat(t, Tair_sum[i], ut)
                tloop+=1
        else:
            # 若没有在加热，则检查温度是否降温到5度一下，ut变为1
            if (t<Tsumm-5):
                # 此时代表重新加热
                ut=1
                heat_sw=True
                t = heat(t, Tair_sum[i], ut)
                tloop+=1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_sum[i])

        t_moment.append(t)
    print("夏季一直开启电源加热时间为"+str(tloop)+"分钟")
    W = (P * tloop) / (1000 * 3600)
    print("夏季一直开启电源加热用电量为" + str(W) + "千瓦时")
    # plt
    date = data['minutes']
    tair = Tair_sum
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = '夏季一直开启电源模式下水温变化'
    # 每类数据依次绘图散点
    # dpi?
    plt.figure(figsize=(16, 10))

    # plt.scatter(date, tair, s=10, color='deepskyblue')
    plt.plot(date, tair, color='deepskyblue', label='室外温度')
    plt.plot(date, t_water, color='salmon', label='水温')
    # 坐标轴
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(60))
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=22)
    plt.legend(fontsize=16, loc='upper right')

    plt.savefig(title + '.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=100, pad_inches=.1, bbox_inches='tight')

    # Load this image into PIL
    png2 = Image.open(png1)

    # Save as TIFF
    png2.save(title + ".tiff")
    png1.close()

def winter_open():
    # 储存温度
    t_moment = []
    # 初始温度
    # t=60
    t=20
    ut=0
    heat_sw=False

    tloop=0
    for i in range(0,1440):
        # 先检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t>=Twint:
                # 到达规定温度 停止加热
                ut=0
                heat_sw=False
                # 自然降温
                t = cooldown(t, Tair_win[i])
            else:
                # 继续加热
                t = heat(t, Tair_win[i], ut)
                tloop+=1
        else:
            # 若没有在加热，则检查温度是否降温到5度一下，ut变为1
            if (t<Twint-5):
                # 此时代表重新加热
                ut=1
                heat_sw=True
                t = heat(t, Tair_win[i], ut)
                tloop+=1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_win[i])
        t_moment.append(t)
    print("冬季一直开启电源加热时间为"+str(tloop)+"分钟")
    W = (P * tloop) / (1000 * 3600)
    print("冬季一直开启电源加热用电量为" + str(W) + "千瓦时")
    # plt
    date = data['minutes']
    tair = Tair_win
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = '冬季一直开启电源模式下水温变化'
    # 每类数据依次绘图散点
    # dpi?
    plt.figure(figsize=(16, 10))

    # plt.scatter(date, tair, s=10, color='deepskyblue')
    plt.plot(date, tair, color='deepskyblue', label='室外温度')
    plt.plot(date, t_water, color='salmon', label='水温')
    # 坐标轴
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(60))
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=22)
    plt.legend(fontsize=16, loc='upper right')

    plt.savefig(title + '.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=100, pad_inches=.1, bbox_inches='tight')

    # Load this image into PIL
    png2 = Image.open(png1)

    # Save as TIFF
    png2.save(title + ".tiff")
    png1.close()

def summer_onetime():
    T0=20
    Tair=29
    t = C*p*V*R*math.log((T0-Tair-n*P*R)/(Tsumm-Tair-n*P*R))
    print("夏季一次性加热时间为"+str(t)+"秒")
    W = (P*t) / (1000*3600)
    print("夏季一次性加热用电量为"+str(W)+"千瓦时")

def winter_onetime():
    # 起始温度还是20吗
    T0 = 20
    Tair = 7
    t = C * p * V * R * math.log((T0 - Tair - n * P * R) / (Twint - Tair - n * P * R))
    print("冬季一次性加热时间为" + str(t) + "秒")
    W = (P * t) / (1000 * 3600)
    print("冬季一次性加热用电量为" + str(W) + "千瓦时")

    # 初始温度
    # t=60
    t = 20
    ut = 1
    heat_sw = False

    tloop = 0
    # 20:00开始
    for i in range(1200, 1440):
        # 加热中要检查是否到达规定温度
        if t >= Twint:
            # 到达规定温度 停止加热
            break
        else:
            # 继续加热
            t = heat(t, Tair_win[i], ut)
            tloop += 1
    print("冬季一次性加热时间为" + str(tloop) + "分钟")
    W = (P * tloop*60) / (1000 * 3600)
    print("冬季一次性加热用电量为" + str(W) + "千瓦时")

if __name__ == '__main__':
    summer_open()
    winter_open()
    summer_onetime()
    winter_onetime()