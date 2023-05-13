import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sko.DE import DE

def scale_x():
    data_12_20 = pd.read_csv("../数据集/整理数据/data_12_20.csv")
    data_12_20 = data_12_20.iloc[:,1:]
    data_c = pd.read_csv('../数据集/整理数据/14-化学性质-orig.csv')
    data_c = data_c.iloc[:,3]
    scaler = MinMaxScaler(feature_range=(0.2, 0.8))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    result_12_20 = scaler.fit_transform(np.array(data_12_20))
    result_c = scaler2.fit_transform(np.array(data_c).reshape(-1, 1))

    scaler3 = MinMaxScaler(feature_range=(0.2, 0.8))
    rongz = np.array([1.13,1.15,1.27,1.32])
    rongz = scaler3.fit_transform(rongz.reshape(-1, 1))
    print('1')


def cal_result(w=0.5):
    for x in [0, 1, 3, 5]:
        if 0 == x:
            shidu = 0.2
            rongz = 0.2
            youji = 0.23976

        elif 0 < x <= 2:
            shidu = 0.23947
            rongz = 0.26316
            youji = 0.39707

        elif 2 < x <= 4:
            shidu = 0.29119
            rongz = 0.64211
            youji = 0.64499

        elif 4 < x <= 8:
            shidu = 0.26787
            rongz = 0.8
            youji = 0.41355

        no_change_data = np.array([0.20000,0.75890,0.52432,0.20000,0.60713,0.80000])
        no_change_weight = np.array([0.1802,0.0787,0.0685,0.2036,0.1282,0.0808])
        normal_data = np.sum(no_change_data*no_change_weight)

        weights_de = np.array([0.0808])
        de_data = np.array([shidu])
        de = de_data * weights_de + normal_data

        weights_bjh = np.array([0.342092, 0.462573, 0.195334])
        bjh_data = np.array([shidu, rongz, youji])
        bjh = np.sum(bjh_data * weights_bjh)

        print(x)
        print(de * w + bjh * (1 - w))

if __name__ == '__main__':
    # scale_x()
    cal_result()