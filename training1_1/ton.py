import math

import numpy as np
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
Tsumm=50
Twint=60
B = 0.00008

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
        # # 20:00 洗澡，20:00+72分钟停下结算已经的用电量
        # if i == 1272:
        #     W = (P * tloop * 60) / (1000 * 3600)
        #     print("夏季在一次性加热完毕后，一直开启模式的用电量为"+ str(W) + "千瓦时")
    print("夏季一直开启电源加热时间为"+str(tloop)+"分钟")
    W = (P * tloop*60) / (1000 * 3600)
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
    plt.legend(fontsize=16, loc='center right')

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
    t=5
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
        # if i == 1317:
        #     W = (P * tloop * 60) / (1000 * 3600)
        #     print("冬季在一次性加热完毕后，一直开启模式的用电量为"+ str(W) + "千瓦时")
    print("冬季一直开启电源加热时间为"+str(tloop)+"分钟")
    W = (P * tloop*60) / (1000 * 3600)
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
    plt.legend(fontsize=16, loc='center right')

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
    T0 = 5
    Tair = 7
    t = C * p * V * R * math.log((T0 - Tair - n * P * R) / (Twint - Tair - n * P * R))
    print("冬季一次性加热时间为" + str(t) + "秒")
    W = (P * t) / (1000 * 3600)
    print("冬季一次性加热用电量为" + str(W) + "千瓦时")

    # 初始温度
    # t=60
    t = 5
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

def summer_suitable():
    # 储存温度
    t_moment = []
    # 初始温度
    # t=45
    t=Tsumm
    ut=0
    heat_sw=False
    shower=False
    tloop=0
    Tset = 37
    Tcw=20
    # 热水流速 Tset出水口，Tcw冷水温度
    shower_time = 1200


    for i in range(0,1440):
        # 假设在20:00开始洗澡
        if i == shower_time:
            shower=True
        if i == shower_time + 15:
            shower=False
            print("洗完澡温度为"+str(t))
            # 洗澡过程结束重新整理下heat_sw
            if (t<Tsumm-5):
                heat_sw=True
            else:
                heat_sw=False
        if shower:
            Bt = B * ((Tset - Tcw) / (t - Tcw))
            R1 = R / (C * p * Bt * R + 1)
            if (t<Tsumm-5):
                ut=1
            else:
                ut=0
            t = t * math.exp(-60 / (C * p * V * R1)) + (
                        n * P * ut * R1 + (Tair_sum[i] * R1 / R) + (C * p * Bt * R1 * Tcw)) * (
                            1 - math.exp(-60 / (C * p * V * R1)))


        # 检查是否在加热
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
                print("在i="+str(i)+"时温度小于设定-5")
                ut=1
                heat_sw=True
                t = heat(t, Tair_sum[i], ut)
                tloop+=1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_sum[i])

        t_moment.append(t)
    print("夏季洗澡一直开启电源加热时间为"+str(tloop)+"分钟")
    W = (P * tloop*60) / (1000 * 3600)
    print("夏季洗澡一直开启电源加热用电量为" + str(W) + "千瓦时")

    # plt
    date = data['minutes']
    tair = Tair_sum
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = '夏季洗澡'
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
    plt.legend(fontsize=16, loc='center right')

    plt.savefig(title + '.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=100, pad_inches=.1, bbox_inches='tight')

    # Load this image into PIL
    png2 = Image.open(png1)

    # Save as TIFF
    png2.save(title + ".tiff")
    png1.close()

def tenet():
    # 储存温度
    t_moment = []
    # 初始温度
    # t=45
    t = 37
    ut = 1
    Tset = 37
    Tcw = 20
    Tair = min(Tair_sum)
    # 热水流速 Tset出水口，Tcw冷水温度

    for i in range(0, 15):
        Bt = B * ((Tset - Tcw) / (t - Tcw))
        R1 = R / (C * p * Bt * R + 1)
        # t = t * math.exp(-60 / (C * p * V * R1)) + (n * P * ut * R1 + (Tair * R1 / R) + (C * p * Bt * R1 * Tcw)) * (1 - math.exp(-60 / (C * p * V * R1)))
        t = (t - (n * P * ut * R1 + (Tair * R1 / R) + (C * p * Bt * R1 * Tcw)) * (1 - math.exp(-60 / (C * p * V * R1)))) / (math.exp(-60 / (C * p * V * R1)))
        t_moment.append(t)
    print("温度为"+str(t))

    # plt
    date = np.linspace(0,15,15)
    tair = Tair
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = 'Tenet'
    # 每类数据依次绘图散点
    # dpi?
    plt.figure(figsize=(16, 10))

    # plt.scatter(date, tair, s=10, color='deepskyblue')
    # plt.plot(date, tair, color='deepskyblue', label='室外温度')
    plt.plot(date, t_water, color='salmon', label='水温')
    # 坐标轴
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(60))
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=22)
    plt.legend(fontsize=16, loc='center right')

    plt.savefig(title + '.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=100, pad_inches=.1, bbox_inches='tight')

    # Load this image into PIL
    png2 = Image.open(png1)

    # Save as TIFF
    png2.save(title + ".tiff")
    png1.close()

def summer_breaking():
    print("夏季搜索开始")
    print("设置温度为"+str(Tsumm))
    print("夏季室温为最低恒定24°")
    # 储存温度
    t_moment = []
    # 初始温度
    # t=45
    t = Tsumm
    ut = 0
    heat_sw = False
    tloop = 0
    # 第一步是为了找出温度最低点作为shower_time
    shower_time = 9999
    i_lowest = []

    # 假设Tair全为24°
    for j in range(0, len(Tair_sum)):
        Tair_sum[j] = min(Tair_sum)

    for i in range(0, 1440):
        # 检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t >= Tsumm:
                # 到达规定温度 停止加热
                ut = 0
                heat_sw = False
                # 自然降温
                t = cooldown(t, Tair_sum[i])
            else:
                # 继续加热
                # t = t * math.exp(-60 / (C * p * V * R)) + (n * P * ut * R + Tair_win[i]) * (1 - math.exp(-60 / (C * p * V * R)))
                t = heat(t, Tair_sum[i], ut)
                tloop += 1
        else:
            # 若没有在加热，则检查温度是否降温到5度一下，ut变为1
            if (t < Tsumm - 5):
                # 此时代表重新加热
                print("在i=" + str(i) + "时温度小于设定-5")
                i_lowest.append(i)
                ut = 1
                heat_sw = True
                t = heat(t, Tair_sum[i], ut)
                tloop += 1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_sum[i])

    shower_time = i_lowest[0]
    t = Tsumm
    ut = 0
    heat_sw = False
    shower = False
    tloop = 0
    Tset = 37
    Tcw = 20
    # 热水流速 Tset出水口，Tcw冷水温度
    # 第二次 返回最低温度
    for i in range(0, 1440):
        if i == shower_time:
            shower = True
            print("在i=" + str(i) + "时开始洗澡")
        if i == shower_time + 15:
            shower = False
            print("洗完澡温度为" + str(t))
            t_target = t
            # 洗澡过程结束重新整理下heat_sw
            if (t < Tsumm - 5):
                heat_sw = True
            else:
                heat_sw = False
        if shower:
            Bt = B * ((Tset - Tcw) / (t - Tcw))
            R1 = R / (C * p * Bt * R + 1)
            if (t < Tsumm - 5):
                ut = 1
            else:
                ut = 0
            t = t * math.exp(-60 / (C * p * V * R1)) + (
                    n * P * ut * R1 + (Tair_sum[i] * R1 / R) + (C * p * Bt * R1 * Tcw)) * (
                        1 - math.exp(-60 / (C * p * V * R1)))

        # 检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t >= Tsumm:
                # 到达规定温度 停止加热
                ut = 0
                heat_sw = False
                # 自然降温
                t = cooldown(t, Tair_sum[i])
            else:
                # 继续加热
                # t = t * math.exp(-60 / (C * p * V * R)) + (n * P * ut * R + Tair_win[i]) * (1 - math.exp(-60 / (C * p * V * R)))
                t = heat(t, Tair_sum[i], ut)
                tloop += 1
        else:
            # 若没有在加热，则检查温度是否降温到5度一下，ut变为1
            if (t < Tsumm - 5):
                # 此时代表重新加热
                ut = 1
                heat_sw = True
                t = heat(t, Tair_sum[i], ut)
                tloop += 1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_sum[i])

        t_moment.append(t)

    # plt
    date = data['minutes']
    tair = Tair_sum
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = '暴力搜索'
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
    plt.legend(fontsize=16, loc='center right')

    plt.savefig(title + '.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=100, pad_inches=.1, bbox_inches='tight')

    # Load this image into PIL
    png2 = Image.open(png1)

    # Save as TIFF
    png2.save(title + ".tiff")
    png1.close()

    return t_target

def summer_breaking():
    print("夏季搜索开始")
    print("设置温度为"+str(Tsumm))
    # print("夏季室温为最低恒定24°")
    print("夏季室温为自然变化")
    # 储存温度
    t_moment = []
    # 初始温度
    # t=45
    t = Tsumm
    ut = 0
    heat_sw = False
    tloop = 0
    # 第一步是为了找出温度最低点作为shower_time
    shower_time = 9999
    i_lowest = []

    # # 假设Tair全为24°
    # for j in range(0, len(Tair_sum)):
    #     Tair_sum[j] = min(Tair_sum)

    for i in range(0, 1440):
        # 检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t >= Tsumm:
                # 到达规定温度 停止加热
                ut = 0
                heat_sw = False
                # 自然降温
                t = cooldown(t, Tair_sum[i])
            else:
                # 继续加热
                # t = t * math.exp(-60 / (C * p * V * R)) + (n * P * ut * R + Tair_win[i]) * (1 - math.exp(-60 / (C * p * V * R)))
                t = heat(t, Tair_sum[i], ut)
                tloop += 1
        else:
            # 若没有在加热，则检查温度是否降温到5度一下，ut变为1
            if (t < Tsumm - 5):
                # 此时代表重新加热
                print("在i=" + str(i) + "时温度小于设定-5")
                i_lowest.append(i)
                ut = 1
                heat_sw = True
                t = heat(t, Tair_sum[i], ut)
                tloop += 1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_sum[i])

    shower_time = i_lowest[0]
    t = Tsumm
    ut = 0
    heat_sw = False
    shower = False
    tloop = 0
    Tset = 37
    Tcw = 20
    # 热水流速 Tset出水口，Tcw冷水温度
    # 第二次 返回最低温度
    for i in range(0, 1440):
        if i == shower_time:
            shower = True
            print("在i=" + str(i) + "时开始洗澡")
        if i == shower_time + 15:
            shower = False
            print("洗完澡温度为" + str(t))
            t_target = t
            # 洗澡过程结束重新整理下heat_sw
            if (t < Tsumm - 5):
                heat_sw = True
            else:
                heat_sw = False
        if shower:
            Bt = B * ((Tset - Tcw) / (t - Tcw))
            R1 = R / (C * p * Bt * R + 1)
            if (t < Tsumm - 5):
                ut = 1
            else:
                ut = 0
            t = t * math.exp(-60 / (C * p * V * R1)) + (
                    n * P * ut * R1 + (Tair_sum[i] * R1 / R) + (C * p * Bt * R1 * Tcw)) * (
                        1 - math.exp(-60 / (C * p * V * R1)))

        # 检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t >= Tsumm:
                # 到达规定温度 停止加热
                ut = 0
                heat_sw = False
                # 自然降温
                t = cooldown(t, Tair_sum[i])
            else:
                # 继续加热
                # t = t * math.exp(-60 / (C * p * V * R)) + (n * P * ut * R + Tair_win[i]) * (1 - math.exp(-60 / (C * p * V * R)))
                t = heat(t, Tair_sum[i], ut)
                tloop += 1
        else:
            # 若没有在加热，则检查温度是否降温到5度一下，ut变为1
            if (t < Tsumm - 5):
                # 此时代表重新加热
                ut = 1
                heat_sw = True
                t = heat(t, Tair_sum[i], ut)
                tloop += 1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_sum[i])

        t_moment.append(t)

    # plt
    date = data['minutes']
    tair = Tair_sum
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = '暴力搜索'
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
    plt.legend(fontsize=16, loc='center right')

    plt.savefig(title + '.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=100, pad_inches=.1, bbox_inches='tight')

    # Load this image into PIL
    png2 = Image.open(png1)

    # Save as TIFF
    png2.save(title + ".tiff")
    png1.close()

    return t_target

def winter_breaking():
    print("冬季搜索开始")
    print("设置温度为"+str(Twint))
    # print("夏季室温为最低恒定24°")
    print("冬季室温为自然变化")
    # 储存温度
    t_moment = []
    # 初始温度
    # t=45
    t = Twint
    ut = 0
    heat_sw = False
    tloop = 0
    # 第一步是为了找出温度最低点作为shower_time
    shower_time = 9999
    i_lowest = []

    # # 假设Tair全为24°
    # for j in range(0, len(Tair_sum)):
    #     Tair_sum[j] = min(Tair_sum)

    for i in range(0, 1440):
        # 检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t >= Twint:
                # 到达规定温度 停止加热
                ut = 0
                heat_sw = False
                # 自然降温
                t = cooldown(t, Tair_win[i])
            else:
                # 继续加热
                # t = t * math.exp(-60 / (C * p * V * R)) + (n * P * ut * R + Tair_win[i]) * (1 - math.exp(-60 / (C * p * V * R)))
                t = heat(t, Tair_win[i], ut)
                tloop += 1
        else:
            # 若没有在加热，则检查温度是否降温到5度一下，ut变为1
            if (t < Twint - 5):
                # 此时代表重新加热
                print("在i=" + str(i) + "时温度小于设定-5")
                i_lowest.append(i)
                ut = 1
                heat_sw = True
                t = heat(t, Tair_win[i], ut)
                tloop += 1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_win[i])

    shower_time = i_lowest[0]
    t = Twint
    ut = 0
    heat_sw = False
    shower = False
    tloop = 0
    Tset = 42
    Tcw = 20
    # 热水流速 Tset出水口，Tcw冷水温度
    # 第二次 返回最低温度
    for i in range(0, 1440):
        if i == shower_time:
            shower = True
            print("在i=" + str(i) + "时开始洗澡")
        if i == shower_time + 15:
            shower = False
            print("洗完澡温度为" + str(t))
            t_target = t
            # 洗澡过程结束重新整理下heat_sw
            if (t < Twint - 5):
                heat_sw = True
            else:
                heat_sw = False
        if shower:
            Bt = B * ((Tset - Tcw) / (t - Tcw))
            R1 = R / (C * p * Bt * R + 1)
            if (t < Twint - 5):
                ut = 1
            else:
                ut = 0
            t = t * math.exp(-60 / (C * p * V * R1)) + (
                    n * P * ut * R1 + (Tair_win[i] * R1 / R) + (C * p * Bt * R1 * Tcw)) * (
                        1 - math.exp(-60 / (C * p * V * R1)))

        # 检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t >= Twint:
                # 到达规定温度 停止加热
                ut = 0
                heat_sw = False
                # 自然降温
                t = cooldown(t, Tair_win[i])
            else:
                # 继续加热
                # t = t * math.exp(-60 / (C * p * V * R)) + (n * P * ut * R + Tair_win[i]) * (1 - math.exp(-60 / (C * p * V * R)))
                t = heat(t, Tair_win[i], ut)
                tloop += 1
        else:
            # 若没有在加热，则检查温度是否降温到5度一下，ut变为1
            if (t < Twint - 5):
                # 此时代表重新加热
                ut = 1
                heat_sw = True
                t = heat(t, Tair_win[i], ut)
                tloop += 1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_win[i])

        t_moment.append(t)

    # plt
    date = data['minutes']
    tair = Tair_win
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = '暴力搜索冬'
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
    plt.legend(fontsize=16, loc='center right')

    plt.savefig(title + '.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=100, pad_inches=.1, bbox_inches='tight')

    # Load this image into PIL
    png2 = Image.open(png1)

    # Save as TIFF
    png2.save(title + ".tiff")
    png1.close()

    return t_target

if __name__ == '__main__':
    print("第二问")
    # Tsumm = 45
    # Twint = 60
    # summer_open()
    # winter_open()
    # summer_onetime()
    # winter_onetime()
    print("第三问")
    # Tsumm = 51.5
    # t_searching = 0
    # while t_searching<37:
    #     Tsumm+=0.01
    #     t_searching = summer_breaking()
    #     print(t_searching)
    # print("最优夏天设置温度为"+str(Tsumm))
    Twint = 62.5
    t_searching = 0
    while t_searching < 42:
        Twint+=0.01
        t_searching = winter_breaking()
        print(t_searching)
    print("最优冬天设置温度为" + str(Twint))
    '''夏天51.56/51.52'''

    # Tsumm = 51.56
    # summer_suitable()
    # tenet()

