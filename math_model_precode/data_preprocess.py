import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import io
from PIL import Image
from minepy import MINE
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import dcor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances
import eli5
import math
from matplotlib import cm
def del_same_feature(data):
    '''
    删除方差为0的列（全部相同）
    :param data: 读取的数据集，只含特征，dataframe
    :returns: 清洗后的dataframe
    '''
    # 全类型通用，包括字符串
    data = data.loc[:, (data != data.iloc[0]).any()]
    # 仅适用于数值类型
    #data.drop(data.columns[data.std() == 0], axis=1, inplace=True)
    return data

def del_perc_same_feature(data, threshold):
    '''
    删除相同比列高于阈值的列
    :param data: 读取的数据集，只含特征，dataframe
    :param threshold: 阈值，某个值占比超过阈值的列会被删除
    :return: 清洗后的dataframe，并打印所删除的特征名
    '''
    # 待删除的列存放
    column_indexs = []
    # 按列遍历dataframe
    for column_index, row_data in data.iteritems():
        counts = row_data.value_counts(normalize=True)
        # 若占比最大的某个值超过阈值，则记入待删除列
        if counts.iloc[0] >= threshold:
            column_indexs.append(column_index)
    print("删除的特征为：" + str(column_indexs))
    data = data.drop(labels=column_indexs, axis=1)
    return data

def del_std_small_feature(data, threshold):
    '''
    删除方差小于阈值的列
    :param data: 读取的数据集，只含特征，dataframe
    :param threshold: 阈值，方差小于阈值的列会被删除
    :return: 清洗后的dataframe
    '''
    # 待删除的列存放
    column_indexs = []
    # 按列遍历dataframe
    for column_index, row_data in data.iteritems():
        counts = row_data.std()
        # 若占比最大的某个值超过阈值，则记入待删除列
        if counts <= threshold:
            column_indexs.append(column_index)
    print("删除的特征为：" + str(column_indexs))
    data = data.drop(labels=column_indexs, axis=1)
    return data

def save_png_to_tiff(name):
    '''
    保存图表为PNG和TIFF两种格式
    :param name: 文件名
    :return: tiff-dpi：200 → 2594x1854
    '''
    plt.savefig('./fig_preview/' + name + '.png')
    # Save the image in memory in PNG format
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=200, pad_inches=.1, bbox_inches='tight')
    # Load this image into PIL
    png2 = Image.open(png1)
    # Save as TIFF
    png2.save('./fig_preview/' + name + ".tiff")
    png1.close()

def draw_feature(data):
    '''
    选择整型、浮点型变量各16个画出分布图
    :param data: 特征
    :return: 特征分布图，png 1600x1000, tiff-dpi：200 → 2594x1854
    '''
    # 数据准备
    # 遍历每列的类型，分成整型和浮点型
    columns_int = []
    columns_float = []
    for column_index in data.columns:
        if str(type(data[column_index][0])) == "<class 'numpy.int64'>":
            columns_int.append(column_index)
        if str(type(data[column_index][0])) == "<class 'numpy.float64'>":
            columns_float.append(column_index)
    # 随机选择16个特征做分布图，整形做条形图，浮点型会多一个拟合曲线
    columns_int_samples = random.sample(columns_int, 16)
    columns_float_samples = random.sample(columns_float, 16)
    print("选取的整型变量为：" + str(columns_int_samples))
    print("选取的浮点变量为：" + str(columns_float_samples))
    # # 根据实际情况也可手动指定
    # columns_int_samples = []
    # columns_float_samples = []
    # 清理空值
    data = data.dropna()

    # 开始画图
    # 预设
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 整体画布大小 1600x1000 tiff-dpi：200 → 2594x1854
    plt.figure(figsize=(16, 10))
    # 绘制16个整型变量子图
    for i in range(1,17):
        plt.subplot(4,4,i)
        sns.distplot(data[columns_int_samples[i-1]], bins=15, kde=False, hist=True, hist_kws={'histtype':'stepfilled'},
                     color='deepskyblue')
    plt.subplots_adjust(hspace=0.35)
    # 可选添加标题
    title = ''
    plt.suptitle(title, fontsize=20)
    # 文件名
    name = '整型变量直方图'
    save_png_to_tiff(name)
    plt.show()

    # 绘制16个浮点型变量子图
    plt.figure(figsize=(16, 10))
    for i in range(1,17):
        plt.subplot(4,4,i)
        sns.distplot(data[columns_float_samples[i-1]], hist=True, norm_hist=False, color='deepskyblue')
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    # 可选添加标题
    title = ''
    plt.suptitle(title, fontsize=20)
    # 文件名
    name = '浮点变量直方图'
    save_png_to_tiff(name)
    plt.show()

def palette(arg:str):
    '''
    调色板
    fade: 12色 红黄绿蓝紫渐变
    simple: 蓝到红渐变 6色
    rainbow: 彩虹 12色
    :return:
    '''
    # 标准彩色调色板，12色 红黄绿蓝紫渐变
    fade = sns.hls_palette(12, l=0.7, s=0.9)

    # 简约 蓝到红渐变 6色
    simple = sns.diverging_palette(240, 10, sep=12)
    # sns.palplot(simple)
    # # cmap = sns.diverging_palette(200, 20, as_cmap=True)

    # 彩虹 12色
    rainbow = sns.color_palette('rainbow', 12)

    # 自定义
    colors = ["deepskyblue", "salmon"]
    # colors = ['#00bfff', '#fa8072']
    custom = sns.color_palette(colors)

    dict = {
        "fade": fade,
        "simple": simple,
        "rainbow": rainbow
    }
    choice = dict.get(arg)
    # 设置调色板
    sns.set_palette(choice)
    sns.palplot(choice)
    plt.show()

def sigma3_rules(data):
    '''
    根据3σ法则，删除某特征为异常值的样本，即删掉某些行，因为他的某个特征是异常值。
    :param data: 特征集
    :return: 清洗后的数据集，并打印所删除样本索引
    '''
    mean = data.mean()
    std = data.std()
    drop_indices = []
    for index, row in data.iterrows():
        tmp = (row - mean).abs() > 3 * std
        if tmp.any():
            drop_indices.append(index)
    data.drop(drop_indices, inplace=True)
    no = [i + 1 for i in drop_indices]
    print(no)
    return data

def del_perc_null_feature(data, threshold):
    '''
    删除缺失值比例大于阈值的特征
    :param data: 特征集
    :param threshold: 缺失值比例的阈值
    :return: 清洗后的特征集，并打印被
    '''
    # 根据需要可以把表格的0转换为空值
    data[data == 0] = np.nan

    nan_perc = data.isnull().sum() / len(data)
    # 筛选缺失值占比大于阈值的列
    nan_columns = list(nan_perc[nan_perc>threshold].index)
    print("删除的特征为：" + str(nan_columns))
    data = data.drop(labels=nan_columns, axis=1)
    data = data.fillna(0)
    return data

def fill_null(data):
    '''
    填充缺失值，支持将0转换成空值处理，方法包含：前后填充、均值填充、线性插值
    :param data:
    :return:
    '''
    # 根据需要可以把表格的0转换为空值
    data[data == 0] = np.nan
    # 打印有缺失值的列和缺失值数目
    nan_count = data.isnull().sum()
    print(nan_count[nan_count>0])

    # 选择一个填充方法 ↓

    # # 取后一个有效值填充
    # data = data.fillna(method='bfill')
    # # 取前一个有效值填充
    # data = data.fillna(method='ffill')
    # 填充列的平均值
    data = data.fillna(data.mean())
    # # 线性插值
    # data = data.interpolate()
    return data

def ShowHeatMap(DataFrame, title):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    plt.title(title, y=1.05, size=15)
    sns.heatmap(DataFrame.astype(float), square=True, cmap="YlGnBu",annot=False)
    plt.tick_params(labelsize=12, left=False, bottom=False)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.show()

def grey_all_features(df):
    '''
    归一化的方法不同，关系系数不同，但趋势是一样
    :param df:
    :return:
    '''
    columns = df.columns
    # 归一化
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    # rho: 分辨率 会影响结果
    rho = 0.01

    def get_grey(data, target):
        # 数列间的绝对距离
        distance = np.abs(data - target)
        dis_min = distance.min()
        dis_max = rho * distance.max()
        # 关联系数: 比较数列中每个值与参照数列的关联性
        cor_param = (dis_min + dis_max) / (distance + dis_max)
        # 关联度: 关联系数按列求平均
        grey = cor_param.mean(axis=0)
        if np.isnan(grey):
            grey = 1
        return grey
    def grey_matrix(dataframe):
        data = np.array(dataframe)
        n = len(data[0, :])
        result = np.zeros([n, n])

        for i in range(n):
            for j in range(n):
                grey = get_grey(data[:, i], data[:, j])
                result[i, j] = grey
                result[j, i] = grey
        RT = pd.DataFrame(result)
        return RT
    data_grey = grey_matrix(df)
    data_grey.columns = columns
    data_grey['column'] = columns
    data_grey.set_index('column', inplace=True)
    data_grey.to_csv('grey_all_features.csv')
    print('Done:输出所有特征之间GREY')
    # ShowHeatMap(data_grey)

def grey_top_m(df, target, m=20):
    '''

    :param df: 特征
    :param target: 被预测量
    :param m: 排名前m个数
    :return:
    '''
    columns = df.columns
    # 归一化
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    target = scaler.fit_transform(np.array(target).reshape(-1, 1))

    # rho: 分辨率 会影响结果
    rho=0.5
    # 数列间的绝对距离
    distance = np.abs(df - target)
    dis_min = distance.min()
    dis_max = rho * distance.max()
    # 关联系数: 比较数列中每个值与参照数列的关联性
    cor_param = (dis_min + dis_max) / (distance + dis_max)
    # 关联度: 关联系数按列求平均
    grey = pd.DataFrame(cor_param.mean(axis=0))
    grey['column'] = columns
    grey.columns = ['importance', 'features']
    grey = grey[['features', 'importance']]
    grey.to_csv('grey_to_y.csv')
    print('Done:输出所有特征对被预测量的灰色关联系数')

    # top m
    grey.sort_values(by=['importance'], ascending=False, inplace=True)
    top_m = grey.iloc[0:m]
    top_m.to_csv('grey_top_m.csv', index=False)
    print('Done:输出灰色关联系数的top'+str(m))

def mic_all_features(df):
    '''
    这个是返回变量间的关系。
    :param df:
    :return:
    '''
    def MIC_matrix(dataframe, mine):

        data = np.array(dataframe)
        n = len(data[0, :])
        result = np.zeros([n, n])

        for i in range(n):
            for j in range(n):
                mine.compute_score(data[:, i], data[:, j])
                result[i, j] = mine.mic()
                result[j, i] = mine.mic()
        RT = pd.DataFrame(result)
        return RT

    mine = MINE(alpha=0.6, c=15)
    data_mic = MIC_matrix(df, mine)
    data_mic.columns = df.columns
    data_mic['column'] = df.columns
    data_mic.set_index('column', inplace=True)
    data_mic.to_csv('data/mic_all_features.csv')
    ShowHeatMap(data_mic, 'MIC')

def mic_top_m(df, target, m=20):
    '''
    这个是返回变量与被预测量。
    :param df: 特征
    :param target: 被预测量
    :param m: 排名前m个数
    :return:
    '''
    def MIC_matrix(dataframe, mine):

        data = np.array(dataframe)
        # 特征数量
        n = len(data[0, :])
        result = np.zeros([1, n])

        for i in range(n):
            mine.compute_score(data[:, i], target)
            result[0, i] = mine.mic()
        RT = pd.DataFrame(result)
        return RT
    # 输出每个特征与被预测量的的MIC
    mine = MINE(alpha=0.6, c=15)
    data_mic = MIC_matrix(df, mine)
    data_mic.columns = df.columns
    data_mic.to_csv('mic_to_y.csv')
    print('Done:输出所有特征对被预测量的MIC')

    # 输出MIC最大的前m个特征
    mic = data_mic.iloc[0, :]
    mic.sort_values(ascending=False, inplace=True)
    top_m = mic.iloc[0:m]
    top_m.to_csv('mic_top_m.csv')
    print('Done:输出MIC的top'+str(m))
    # 作图
    def ShowHeatMap(DataFrame):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title('MIC', y=1.05, size=15)
        sns.heatmap(DataFrame.astype(float), square=True, cmap="YlGnBu",annot=False)
        plt.yticks(rotation=0)
        plt.show()
    # ShowHeatMap(data_mic)

def dcor_all_features(df):
    '''
    这个是返回变量间的关系。
    :param df:
    :return:
    '''
    def dcor_matrix(dataframe):

        data = np.array(dataframe)
        n = len(data[0, :])
        result = np.zeros([n, n])

        for i in range(n):
            for j in range(n):
                d_cor = dcor.distance_correlation(data[:, i], data[:, j])
                result[i, j] = d_cor
                result[j, i] = d_cor
        RT = pd.DataFrame(result)
        return RT

    data_dcor = dcor_matrix(df)
    data_dcor.columns = df.columns
    data_dcor['column'] = df.columns
    data_dcor.set_index('column', inplace=True)
    data_dcor.to_csv('dcor_all_features.csv')

    ShowHeatMap(data_dcor, 'DCOR')

def dcor_top_m(df, target, m=20):
    '''
    这个是返回变量与被预测量。
    :param df: 特征
    :param target: 被预测量
    :param m: 排名前m个数
    :return:
    '''
    def dcor_matrix(dataframe):

        data = np.array(dataframe)
        # 特征数量
        n = len(data[0, :])
        result = np.zeros([1, n])

        for i in range(n):
            d_cor = dcor.distance_correlation(data[:, i], target)
            result[0, i] = d_cor
        RT = pd.DataFrame(result)
        return RT
    # 输出每个特征与被预测量的
    data_dcor = dcor_matrix(df)
    data_dcor.columns = df.columns
    data_dcor.to_csv('dcor_to_y.csv')
    print('Done:输出所有特征对被预测量的dcor')

    # 输出MIC最大的前m个特征
    ddcor = data_dcor.iloc[0, :]
    ddcor.sort_values(ascending=False, inplace=True)
    top_m = ddcor.iloc[0:m]
    top_m.to_csv('dcor_top_m.csv')
    print('Done:输出dcor的top'+str(m))
    # 作图
    def ShowHeatMap(DataFrame):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        colormap = plt.cm.RdBu
        plt.figure(figsize=(14, 12))
        plt.title('MIC', y=1.05, size=15)
        sns.heatmap(DataFrame.astype(float), square=True, cmap="YlGnBu",annot=False)
        plt.yticks(rotation=0)
        plt.show()
    # ShowHeatMap(data_mic)

def rf_features(x, y, m=20):
    '''

    :param x:
    :param y:
    :return:
    '''
    columns = x.columns

    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=1000)
    y_pred = model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

    # 重要性
    imp = pd.DataFrame({'features': columns, 'importance': model.feature_importances_})
    imp.to_csv('rf_all_features.csv')
    print('Done:输出所有特征对被预测量的随机森林重要性')
    imp.sort_values(by=['importance'], ascending=False, inplace=True)
    top_m = imp.iloc[0:m]
    top_m.to_csv('rf_top_m.csv', index=False)
    print('Done:输出随机森林重要性的top'+str(m))

    # Permutation Importance
    # model是已经训练好的模型
    perm_imp_model = PermutationImportance(model, random_state=42).fit(x_test, y_test)
    perm_imp = pd.DataFrame({'features': columns, 'importance': perm_imp_model.feature_importances_})
    perm_imp.to_csv('perm_imp_all_features.csv')
    print('Done:输出所有特征对被预测量的排列重要性')
    perm_imp.sort_values(by=['importance'], ascending=False, inplace=True)
    top_m = perm_imp.iloc[0:m]
    top_m.to_csv('perm_imp_top_m.csv', index=False)
    print('Done:输出排列重要性的top'+str(m))
    # 显示结果
    eli5.show_weights(perm_imp_model, feature_names=columns.tolist())

def feature_preprocess():
    '''
    21D题729特征处理后变为228个
    :return:
    '''
    # setting
    feature_file = './dataset/Molecular_Descriptor.xlsx'
    data = pd.read_excel(feature_file)
    # 删掉第一列，分子结构，只保留特征
    data = data.iloc[:, 1:]
    print(data.info)
    # [1974 rows x 729 columns]>
    # testing
    data = del_same_feature(data)
    print(data.info)
    # [1974 rows x 504 columns]>
    data = del_perc_same_feature(data, 0.9)
    print(data.info)
    # [1974 rows x 362 columns] >
    data = del_std_small_feature(data, 0.05)
    # [1974 rows x 341 columns] >
    data = del_perc_null_feature(data, 0.5)
    print(data.info)
    # [1974 rows x 228 columns]>
    input()
    data.to_csv('dataset/features.csv')

def feature_selection():
    # 特征选择
    features = './dataset/features.csv'
    y_file = './dataset/ER_activity.xlsx'
    data = pd.read_csv('./dataset/features.csv', index_col=0)

    y = pd.read_excel(y_file)
    y = y.iloc[:, 2]

    # 最大信息系数top
    mic_top_m(data, y, 40)
    # 灰色关联top
    grey_top_m(data, y, 40)
    # dcor top
    dcor_top_m(data, y, 40)
    # rf and permutation_importance
    rf_features(data, y, 40)

def feature_selection_graph():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    grey = pd.read_csv('grey_to_y.csv', header=None)
    mic = pd.read_csv('mic_to_y.csv', index_col=0).T
    rf = pd.read_csv('rf_all_features.csv', index_col=0)
    perm = pd.read_csv('perm_imp_all_features.csv', index_col=0)

    grey_imp = np.array(grey.iloc[:, 1])
    mic_imp = np.array(mic.iloc[:, 0])
    rf_imp = np.array(rf.iloc[:, 1])
    perm_imp = np.array(perm.iloc[:, 1])

    columns = mic.index.values

    # 渐变
    def get_colors(y):
        # 归一化
        norm = plt.Normalize(0, y.max())
        norm_values = norm(y)
        map_vir = cm.get_cmap(name='rainbow')
        return map_vir(norm_values)

    # 整体画布大小 1600x1000 tiff-dpi：200 → 2594x1854
    plt.figure(figsize=(16, 10))
    imp_dict = {'grey_imp':grey_imp, 'mic_imp':mic_imp, 'rf_imp':rf_imp, 'perm_imp':perm_imp}
    for name in imp_dict.keys():
        imp = imp_dict[name]
        plt.xlabel("特征变量", fontsize = 16)  # 设置x坐标标签
        plt.ylabel("特征重要度", fontsize = 16)  # 设置y坐标标签
        plt.bar(columns, imp, color=get_colors(imp))
        plt.xticks([])
        plt.tick_params(labelsize=16)
        if name in ['rf_imp', 'perm_imp']:
            plt.yscale('log')
        plt.show()

def check(features):
    # 相关性检验
    # 距离相关系数
    dcor_all_features(features)
    # 皮尔逊
    plt.figure(figsize=(12, 12))
    sns.heatmap(features.corr(), square=True, annot=False)
    plt.tick_params(labelsize=12, left=False, bottom=False)
    plt.xticks(rotation=45)
    plt.show()

def feature_integration(m=20):
    grey = pd.read_csv('grey_to_y.csv', index_col=0)
    mic = pd.read_csv('mic_to_y.csv', index_col=0).T
    rf = pd.read_csv('rf_all_features.csv', index_col=0)
    perm = pd.read_csv('perm_imp_all_features.csv', index_col=0)

    stand_scaler = StandardScaler()

    grey_imp = stand_scaler.fit_transform(np.array(grey.iloc[:,1]).reshape(-1, 1))
    mic_imp = stand_scaler.fit_transform(np.array(mic.iloc[:,0]).reshape(-1, 1))
    rf_imp = stand_scaler.fit_transform(np.array(rf.iloc[:,1]).reshape(-1, 1))
    perm_imp = stand_scaler.fit_transform(np.array(perm.iloc[:,1]).reshape(-1, 1))

    columns = mic.index.values

    # result = ((grey_imp + mic_imp) / 3) * (1 + np.exp(-rf_imp)) * (1 + np.exp(-perm_imp))
    result = grey_imp * 0.2 + mic_imp * 0.2 + rf_imp * 0.3 + perm_imp * 0.3

    minmax_scaler = MinMaxScaler()
    result = minmax_scaler.fit_transform(result).ravel()
    final_result = pd.DataFrame({'features': columns, 'importance': result})
    final_result.sort_values('importance', ascending=False, inplace=True)
    final_result.reset_index(drop=True, inplace=True)

    # 相关性检验 然后去除 补上
    data = pd.read_csv('dataset/features.csv')

    # 筛选
    while True:
        top_feature = list(final_result.iloc[:m, 0])
        top_features = data.loc[:, top_feature]

        # 皮尔逊
        plt.figure(figsize=(12, 12))
        corr = top_features.corr()
        sns.heatmap(corr, square=True, annot=False)
        plt.tick_params(labelsize=12, left=False, bottom=False)
        plt.xticks(rotation=45)
        plt.show()
        print("删除前的皮尔逊相关性热力图")
        corr_matrix = corr.abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find features with correlation greater than 0.8
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        print("基于皮尔逊待删除：")
        print(to_drop)

        # 距离相关
        print("删除前的距离系数相关性热力图")
        dcor_all_features(top_features)
        dcor_matrix = pd.read_csv('dcor_all_features.csv')
        dcor_matrix.set_index('column', inplace=True)
        # Select upper triangle of correlation matrix
        upper = dcor_matrix.where(np.triu(np.ones(dcor_matrix.shape), k=1).astype(bool))
        # Find features with correlation greater than 0.8
        to_drop_dcor = [column for column in upper.columns if any(upper[column] > 0.8)]
        print("基于距离系数待删除：")
        print(to_drop_dcor)

        if len(to_drop + to_drop_dcor) == 0:
            break
            print("清洗结束")
        # Drop features
        final_result = final_result[
            ~final_result['features'].str.contains('|'.join(str(column) for column in (to_drop + to_drop_dcor)))]
        final_result.reset_index(drop=True, inplace=True)
        print("再次清洗")
    final_result.to_csv('final_feature.csv')

def feature_relation_graph(m=20):
    data = pd.read_csv('dataset/features.csv')
    # 其他方法的特征
    grey = pd.read_csv('grey_top_m.csv')
    mic = pd.read_csv('mic_top_m.csv')
    rf = pd.read_csv('rf_top_m.csv')
    perm = pd.read_csv('perm_imp_top_m.csv')

    grey_features = list(grey.iloc[:m, 0])
    mic_features = list(mic.iloc[:m, 0])
    rf_features = list(rf.iloc[:m, 0])
    perm_features = list(perm.iloc[:m, 0])
    method_dict = {'grey_features':grey_features, 'mic_features':mic_features,
                   'rf_features':rf_features, 'perm_features':perm_features}
    for method in method_dict.keys():
        method_features = method_dict[method]
        good_features = data.loc[:, method_features]
        print('输出'+ method + '的DCOR非线性相关系数和皮尔逊线性相关系数的图')
        check(good_features)

def box_plot(df):
    '''
    :param df: 各模型评价指标
    :return:   4个评价指标的箱线图
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    sns.set_theme(style="whitegrid")

    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    # 对所有指标进行遍历画图
    for column in df.columns[:1]:
        ax = sns.boxplot(x="model", y=column, data=df, hue='model', dodge=False,
                    showmeans=True,
                    meanprops={"marker": "d",
                               "markerfacecolor": "white",
                               "markeredgecolor": "black",},
                    palette=sns.diverging_palette(240, 10, sep=12))
        model_labels = ['KNN', '多层感知机', '随机森林回归', '支持向量机回归', 'XGBoost']

        n = 0
        for i in model_labels:
            ax.legend_.texts[n].set_text(i)
            n += 1

        plt.show()

if __name__ == '__main__':
    # # 特征预处理
    # feature_preprocess()

    # # 特征选择
    # feature_selection()

    # # 特征选择可视化
    # feature_selection_graph()

    # # 特征集成
    # feature_integration()

    # 四种方法得到的重要性的相关性可视化
    feature_relation_graph()

    print('complete')