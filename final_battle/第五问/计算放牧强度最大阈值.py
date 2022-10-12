import pandas as pd
import numpy as np

def cal_max():
    df = pd.read_csv('各降雨量下所需数据.csv')
    df = df.iloc[:,1:]


    data= np.array(df)
    no_change_weight = np.array([0.1802,0.0787,0.0685,0.2036,0.0808,0.1282])
    for i in range(data.shape[0]):
        no_change_data = data[i,1:7]
        for j in [1,2,3,4]:
            if j==1:
                rongz = 0.2
                youji = 0.23976
            elif j==2:
                rongz = 0.26316
                youji = 0.39707
            elif j==3:
                rongz = 0.64211
                youji = 0.64499
            elif j==4:
                rongz = 0.8
                youji = 0.41355
            shidu = no_change_data[5]
            weights_bjh = np.array([0.342092, 0.462573, 0.195334])
            bjh_data = np.array([shidu, rongz, youji])
            bjh = np.sum(bjh_data * weights_bjh)
            if bjh>0.52:
                print(bjh)
                print(str(j)+'级放牧强度板结化大于0.52')
                print('此时降水量是第'+str(i))
                continue

            de = 0.4
            normal_data = np.sum(no_change_data*no_change_weight)
            if de-normal_data < 0:
                print("沙漠化指数大于0.4")
                print('此时降水量是第' + str(i))
            else:
                stre = ((de-normal_data) / 0.1282)*8
                print("最大放牧强度为"+ str(stre))
                print('此时降水量是第' + str(i))



if __name__ == '__main__':
    cal_max()