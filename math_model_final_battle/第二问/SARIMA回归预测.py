import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pylab
import seaborn as sns
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import io
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
def lubo(df):
    # 这里是假设A=1，H=1的情况

    # intial parameters
    # 总样本数，迭代次数
    n_iter = df.shape[0]
    sz = (n_iter,)  # size of array 若元组仅一个元素需携带括号，否则会被识别为运算符

    # 观测数据
    z = np.array(df.loc[:, '降水量(mm)'])

    Q = 1e-5  # process variance

    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor

    R = 0.00005  # estimate of measurement variance, change to see effect

    # intial guesses
    # 首个最有估计值取对应位置的原数据
    xhat[0] = df.loc[0, '降水量(mm)']
    P[0] = 1

    # 根据KF五条公式进行循环迭代
    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1 - K[k]) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

    # 对最优估计值进行四舍五入取整操作
    xhat = np.round(xhat, 0).reshape(-1,1)
    print(xhat.shape)
    xhat_amend = np.zeros(xhat.shape)
    xhat_amend[:-700] = xhat[700:]
    xhat_amend[-700:] = xhat[-700:]

    pylab.figure()
    pylab.plot(z, 'r-', label='原始数据', alpha=0.7)  # 测量值
    # pylab.plot(xhat, 'b-', label='a posteri estimate', alpha=0.5)  # 过滤后的值
    pylab.plot(xhat_amend, 'b-', label='修正数据', alpha=0.8)  # 过滤后的值
    pylab.legend()
    pylab.xlabel('时间')
    pylab.ylabel('降水量')
    #
    # pylab.figure()
    # valid_iter = range(1,n_iter) # Pminus not valid at step 0
    # pylab.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
    # pylab.xlabel('Iteration')
    # pylab.ylabel('$(Voltage)^2$')
    # pylab.setp(pylab.gca(),'ylim',[0,.01])
    pylab.show()
    return xhat_amend

def show_data():
    # 折线重叠 取均值红线
    df = pd.read_csv('../数据集/整理数据/all_data_ym.csv')
    watermean = []
    for i in range(1,13):
        data = df.loc[df['月份']==i, '降水量(mm)']
        watermean.append(data.mean())
        print('1')
    for i in range(2012, 2022):
        data = df.loc[df['年份']==i, '降水量(mm)']
        data.reset_index(drop=True, inplace=True)
        plt.plot(data)
    plt.plot(watermean, color='salmon', linewidth=4)
    plt.show()

    xhat_amend = lubo(df)
    df.insert(0, '降雨量(mm)', xhat_amend)

    watermean = []
    for i in range(1,13):
        data = df.loc[df['月份']==i, '降雨量(mm)']
        watermean.append(data.mean())
        print('1')

    for i in range(2012, 2022):
        data = df.loc[df['年份']==i, '降雨量(mm)']
        data.reset_index(drop=True, inplace=True)
        plt.plot(data)
    plt.plot(watermean, color='salmon', linewidth=4)
    plt.show()

def show_figure():
    df = pd.read_csv('../数据集/整理数据/all_data_ym.csv')
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(df['降水量(mm)'])
    plt.ylabel('降水量(mm)')
    plt.show()
    plt.plot(df['土壤蒸发量(mm)'])
    plt.ylabel('土壤蒸发量(mm)')
    plt.show()

def get_eval_indicator(y_test, y_pre):
    '''
    :param y_test: 真实值
    :param y_pre: 预测值（模型预测出来的)
    :return: 4种评价指标
    返回回归任务的4种评价指标
    '''
    mae = mean_absolute_error(y_test, y_pre)
    mse = mean_squared_error(y_test, y_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_pre))
    r2 = r2_score(y_test, y_pre)
    return mae, mse, rmse, r2

def save_png_to_tiff(name):
    '''
    保存图表为PNG和TIFF两种格式
    :param name: 文件名
    :return: tiff-dpi：200 → 2594x1854
    '''
    plt.savefig('./fig_preview/' + name + '_arima.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=200, pad_inches=.1, bbox_inches='tight')
    # Load this image into PIL
    png2 = Image.open(png1)
    # Save as TIFF
    png2.save('./fig_preview/' + name + "_arima.tiff")
    png1.close()


def sarima_regression(column, test_size=0.8):
    # 读取数据和预设
    plt.clf()
    df = pd.read_csv('所有特征整合数据.csv')
    df_ = df.iloc[1:123]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 数据划分并且可视化
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # df_ = scaler.fit_transform(np.array(df[column]).reshape(-1, 1))
    # train_size = int(len(df) * test_size)
    train = df_[column]
    # train = train.loc[:, column]
    # test = test.loc[:, column]

    plt.ylabel(column)
    plt.xlabel("日期")

    model = pm.auto_arima(train,
                          information_criterion='aic',
                          test='adf',  # use adftest to find optimal 'd'
                          max_p=5, max_q=5,  # maximum p and q
                          d=None,  # let LSTM模型参数 determine 'd'
                          seasonal=True, m=12, D=1,
                          max_order=None)
    print(model.summary())

    # 模型导出与载入
    import joblib
    joblib.dump(model, 'auto_arima_' + column + '.pkl')
    # LSTM模型参数 = joblib.load(LSTM模型参数,'auto_arima.pkl')

    # make your forecasts
    # predict N steps into the future
    pred = model.predict(21)

    # # 保存指标
    # df_index=0
    # result = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'R2'])
    # mae, mse, rmse, r2 = get_eval_indicator(test, pred)
    # result.loc[df_index, 'MAE'] = mae
    # result.loc[df_index, 'MSE'] = mse
    # result.loc[df_index, 'RMSE'] = rmse
    # result.loc[df_index, 'R2'] = r2
    # result.to_csv('result_' + column + '_anima.csv')

    # 可视化
    # pred = list(pred)
    # pred.insert(0,train.iloc[-1])

    plt.plot(train)
    plt.plot(pred)

    plt.xticks(range(0, df.shape[0], 3), df.loc[range(0, df.shape[0], 3), 'date'], rotation=45)

    pred.to_csv('pred_' + column + '_anima.csv')
    save_png_to_tiff(column)

if __name__ == '__main__':
    # show_data()
    # show_figure()
    test_size = 0.8
    # columns = ['降水量(mm)', '土壤蒸发量(mm)', '植被指数(NDVI)']
    columns = ['icstore']
    columns = ['40cm湿度(kgm2)', '100cm湿度(kgm2)', '200cm湿度(kgm2)']
    for column in columns:
        sarima_regression(column, test_size)
