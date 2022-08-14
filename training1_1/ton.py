import math
import copy
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
    t=20
    ut=0
    heat_sw=False
    Tcw = 20
    tloop=0
    for i in range(0,1440):
        Bt = 0
        R1 = R / ((C * p * Bt * R) + 1)
        # 先检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t>=Twint:
                # 到达规定温度 停止加热
                ut=0
                heat_sw=False
                # 自然降温
                # t = cooldown(t, Tair_win[i])
                t = t * math.exp(-60 / (C * p * V * R1)) + (
                        n * P * ut * R1 + ((Tair_win[i] * R1) / R) + (C * p * Bt * R1 * Tcw)) * (
                            1 - math.exp(-60 / (C * p * V * R1)))
            else:
                # 继续加热
                # t = heat(t, Tair_win[i], ut)
                t = t * math.exp(-60 / (C * p * V * R1)) + (
                        n * P * ut * R1 + ((Tair_win[i] * R1) / R) + (C * p * Bt * R1 * Tcw)) * (
                            1 - math.exp(-60 / (C * p * V * R1)))
                tloop+=1
        else:
            # 若没有在加热，则检查温度是否降温到5度一下，ut变为1
            if (t<Twint-5):
                # 此时代表重新加热
                ut=1
                heat_sw=True
                # t = heat(t, Tair_win[i], ut)
                t = t * math.exp(-60 / (C * p * V * R1)) + (
                        n * P * ut * R1 + ((Tair_win[i] * R1) / R) + (C * p * Bt * R1 * Tcw)) * (
                            1 - math.exp(-60 / (C * p * V * R1)))
                tloop+=1
            else:
                # 此时代表继续降温
                # t = cooldown(t, Tair_win[i])
                t = t * math.exp(-60 / (C * p * V * R1)) + (
                        n * P * ut * R1 + ((Tair_win[i] * R1) / R) + (C * p * Bt * R1 * Tcw)) * (
                            1 - math.exp(-60 / (C * p * V * R1)))
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
    print("最佳温度夏季洗澡一直开启电源加热时间为" + str(tloop) + "分钟")
    W = (P * tloop * 60) / (1000 * 3600)
    print("最佳温度夏季洗澡一直开启电源加热用电量为" + str(W) + "千瓦时")
    # plt
    date = data['minutes']
    tair = Tair_sum
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = '夏季最优设定温度下EWH水温变化'
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
            R1 = R / ((C * p * Bt * R) + 1)
            if (t < Twint - 5):
                ut = 1
            else:
                ut = 0
            t = t * math.exp(-60 / (C * p * V * R1)) + (
                    n * P * ut * R1 + ((Tair_win[i] * R1) / R) + (C * p * Bt * R1 * Tcw)) * (
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
    print("最佳温度冬季洗澡一直开启电源加热时间为" + str(tloop) + "分钟")
    W = (P * tloop * 60) / (1000 * 3600)
    print("最佳温度冬季洗澡一直开启电源加热用电量为" + str(W) + "千瓦时")
    # plt
    date = data['minutes']
    tair = Tair_win
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = '冬季最优设定温度下EWH水温变化'
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

def winter_double_heat(tset, tone):
    print("设置温度为" + str(tset))
    print("洗钱加热温度为"+ str(tone))
    # 储存温度
    t_moment = []
    # 初始温度
    # t=45
    t = 20
    ut = 0
    heat_sw = False
    tloop = 0
    # 第一步是为了找出温度最低点作为shower_time
    shower_time = 9999
    i_lowest = []

    # 扩大温度
    Tair_win2 = list(Tair_win)*2

    for i in range(0, 1440):
        # 检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t >= tset:
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
            if (t < tset - 5):
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

    first_time = i_lowest[2]
    first_heat = False
    second_heat = False
    first_heat_complete = False
    shower = False
    t = 20
    ut = 0
    heat_sw = False
    first_shower_complete = False
    tloop = 0
    Tset = 42
    Tcw = 20
    # 第一次加热后温度
    T_firstheat = tone
    # 等待时间1
    t_firstwait = 0
    t_secondwait = 0
    # 初始化洗澡时间
    first_shower_time = 9999
    second_shower_time =9999
    # 热水流速 Tset出水口，Tcw冷水温度
    # 第二次 返回最低温度
    for i in range(0, 1440):
        #第一次加热
        if i == first_time:
            first_heat = True
            first_heat_complete = True
            print("在i=" + str(i) + "时开始第一次加热")
        if t >= T_firstheat and first_heat_complete and first_shower_complete==False:
            first_heat = False
            shower = True
            first_shower_time = i
            print("在i=" + str(i) + "时开始第一次洗澡")
            first_shower_complete = True
        if first_heat:
            ut = 1
            t = heat(t, Tair_win[i], ut)
            tloop += 1
            t_firstwait += 1
            t_moment.append(t)
            continue
        if i == first_shower_time + 15:
            shower = False
            print("第一次洗完澡温度为" + str(t))
            t_finish_1shower = t
            # 第一次洗完澡马上第二次加热
            second_heat = True
            print("在i=" + str(i) + "时开始第二次加热")
        if i == second_shower_time + 15:
            shower = False
            print("第二次洗完澡温度为" + str(t))
            # 洗澡过程结束重新整理下heat_sw
            if (t < tset - 5):
                heat_sw = True
            else:
                heat_sw = False
        if second_heat:
            if t >= 62.57:
                second_heat = False
                shower = True
                second_shower_time = i
                print("在i=" + str(i) + "时开始第二次洗澡")
            else:
                ut = 1
                t = heat(t, Tair_win[i], ut)
                tloop += 1
                t_secondwait += 1
                t_moment.append(t)
                continue
        if shower:
            Bt = B * ((Tset - Tcw) / (t - Tcw))
            R1 = R / ((C * p * Bt * R) + 1)
            ut=1
            t = t * math.exp(-60 / (C * p * V * R1)) + (
                    n * P * ut * R1 + ((Tair_win2[i] * R1) / R) + (C * p * Bt * R1 * Tcw)) * (
                        1 - math.exp(-60 / (C * p * V * R1)))
            # t = heat(t, Tair_win[i], ut)
            tloop += 1
            t_moment.append(t)
            continue
        # 检查是否在加热
        if heat_sw:
            # 加热中要检查是否到达规定温度
            if t >= tset:
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
            if (t < tset - 5):
                # 此时代表重新加热
                ut = 1
                heat_sw = True
                t = heat(t, Tair_win[i], ut)
                tloop += 1
            else:
                # 此时代表继续降温
                t = cooldown(t, Tair_win[i])

        t_moment.append(t)
    print("最佳温度冬季洗澡一直开启电源加热时间为" + str(tloop) + "分钟")
    W = (P * tloop * 60) / (1000 * 3600)
    print("最佳温度冬季洗澡一直开启电源加热用电量为" + str(W) + "千瓦时")
    wait = t_firstwait + t_secondwait
    print("等待时间为"+str(wait)+"分钟")
    # plt
    date = data['minutes']

    date_2 = "day2 " + date
    date2 = list(date)+list(date_2)
    tair = Tair_win
    t_water = t_moment
    x_name = '时间'
    y_name = '温度 / °C'
    title = '冬季二人洗澡最优设定温度下EWH水温变化'
    # 每类数据依次绘图散点
    # dpi?
    plt.figure(figsize=(16, 10))

    # plt.scatter(date, tair, s=10, color='deepskyblue')
    plt.plot(date, tair, color='deepskyblue', label='室内温度')
    plt.plot(date, t_water, color='salmon', label='水箱温度')
    # 坐标轴
    plt.xlabel(x_name, fontsize=16)
    plt.ylabel(y_name, fontsize=16)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(120))
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

    fitness = 0.7*W + 0.3*(wait/60)
    return fitness

# 粒子群
class PSO(object):
    def __init__(self, w, population_size, max_steps, x_bound, v_bound):
        self.w = w
        self.c1 = 2  # 惯性权重
        self.c2 = 2  # 惯性权重
        self.population_size = population_size                                              # 粒子群大小
        self.x_bound = x_bound                                                              # 粒子位置范围
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1], (population_size, 2))  # 初始化粒子位置
        self.v_bound = v_bound                                                              # 粒子速度范围
        self.v = np.random.uniform(self.v_bound[0], self.v_bound[1], (population_size, 2))# 初始化粒子速度
        self.max_steps = max_steps
        # 第一列要重新赋值
        for k in range(0, len(self.x)):
            self.x[k][0] = np.random.uniform(50, 75)
        for k in range(0, len(self.x)):
            if self.x[k][0] > self.x[k][1]:
                self.x[k][0], self.x[k][1] = self.x[k][1], self.x[k][0]
        # 最大迭代次数
        self.calculate_fitness(self.x)
        # 计算每个粒子适应度
        self.p = copy.deepcopy(self.x)                                                      # 每个粒子的历史最佳位置-x
        self.global_best_x = copy.deepcopy(self.x[np.argmin(fitness)])                      # 种群的粒子最佳位置-x
        self.individual_best_fitness = copy.deepcopy(fitness)                               # 个体的最优适应度-y
        self.global_best_fitness = np.min(fitness)
        # 全局最佳适应度-y

    def calculate_fitness(self, x):
        for i in range(self.population_size):
            fitness[i] = winter_double_heat(x[i, 0], x[i, 1])
        return fitness

    def evolve(self):
        fig = plt.figure()

        for j in range(self.max_steps):
            r1 = np.random.uniform(low=-0.1, high=0.1, size=(self.population_size, 2))
            r2 = np.random.uniform(low=-0.1, high=0.1, size=(self.population_size, 2))
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.global_best_x - self.x)
            self.x = self.v + self.x
            # 限定
            self.x = np.where(self.x<=75,self.x,np.random.uniform(63,75))
            self.x = np.where(self.x >= 63, self.x, np.random.uniform(63, 75))
            for k in range(0, len(self.x)):
                if self.x[k][0] > self.x[k][1]:
                    self.x[k][0], self.x[k][1] = self.x[k][1], self.x[k][0]
            plt.clf()  # Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用
            plt.scatter(self.p[:, 0], self.p[:, 1], s=self.population_size, color='r')
            plt.xlim(43, 75)
            plt.ylim(63, 75)
            plt.pause(0.01)  # python每次画完图像后暂停0.01秒
            self.calculate_fitness(self.x)
            # 新一代粒子和前代粒子比较，如果新一代粒子的适应度比前代更好地,更新每个粒子的历史最佳位置，更新适应度
            for k in range(self.population_size):
                if fitness[k] < self.individual_best_fitness[k]:
                    self.individual_best_fitness[k] = fitness[k]
                    self.p[k] = self.x[k]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.global_best_x = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            D.append(self.global_best_fitness)
        print(self.global_best_x)
        print(self.global_best_fitness)

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
    # Twint = 62.55
    # t_searching = 0
    # while t_searching < 42:
    #     Twint+=0.01
    #     t_searching = winter_breaking()
    #     print(t_searching)
    # print("最优冬天设置温度为" + str(Twint))
    # '''夏天51.56/冬天62.57'''
    print("第四问")
    result = winter_double_heat(50.18957975,63.37830807)
    print(result)


    # print("粒子群")
    # fitness = np.zeros((200, 1))
    # A = D = []
    # pso = PSO(0.9, 200, 20, [63, 75], [-1, 1])
    # pso.evolve()
    # print("y = %s" % C)
    # x = range(len(D))
    # plt.plot(x, D, color='deepskyblue')
    # plt.show()


