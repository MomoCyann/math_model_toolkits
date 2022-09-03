import pandas_method as pd
import glob
import os
import shutil
import datetime
import time

class FeatureAndLabel:
    def __init__(self, df_trade, df_news, trade_interval, predict_interval,feature_save_path):
        # 初始化股票数据及舆情数据表
        self.df_trade = df_trade
        self.df_news = df_news
        # 初始化交易日区间
        self.trade_interval = trade_interval
        self.predict_interval=predict_interval
        self.feature_save_path = feature_save_path

        # 初始化特征表，通过 df.loc[index] = ['',...,''] 新增行
        # 第一行股票信息
        # 2-3 为舆情信息
        # 4为股票因子，分别为开盘价、收盘价、成交数量、涨跌幅（今日收盘价/昨日收盘价-1）
        self.df_feature = df_trade.iloc[0:2,1:]
        self.columns=self.df_feature.columns[5:]

    def get_feature_windows(self):
        df_trade = self.df_trade
        df_news = self.df_news

        ticker = df_trade.loc[0, 'secID'].split('.')[0]
        companyName = df_trade.loc[0, 'secShortName']
        if companyName[0]=='*':
            companyName=companyName[1:]
        print(companyName)

        featureIndex = 0

        for index in self.df_trade.index[self.trade_interval-1:-self.predict_interval]:
            # 获取交易区间数据, start 含当天  end 不含当天
            # select为选择的当天，例如 1-5 为数据 6-10为预测， 测select为5
            tradeStartDate = df_trade.loc[index - self.trade_interval+1, 'tradeDate']
            tradeEndDate = df_trade.loc[index+1, 'tradeDate']
            selectedDate = df_trade.loc[index, 'tradeDate']
            # print(tradeStartDate,tradeEndDate)

            self.df_feature.loc[featureIndex,'tradeStartDate']=tradeStartDate
            self.df_feature.loc[featureIndex,'selectedDate']=selectedDate

            # 获取舆情统计数据
            # 获取该交易区间内的舆情子表
            df_temp = df_news.loc[df_news['newsPublishTime'] > tradeStartDate].loc[
                df_news['newsPublishTime'] < tradeEndDate]

            # 新闻数量统计，各极性统计
            newsCount = df_temp['sentiment'].count()
            sentimentPositive = df_temp.loc[df_temp['sentiment'] == 1].count()['sentiment']
            sentimentNeural = df_temp.loc[df_temp['sentiment'] == 0].count()['sentiment']
            sentimentNegative = df_temp.loc[df_temp['sentiment'] == -1].count()['sentiment']

            stmPstPct=sentimentPositive/newsCount
            stmNrPct=sentimentNeural/newsCount
            stmNgtPct=sentimentNegative/newsCount

            # 新闻舆情分数统计
            maxScore = df_temp['sentimentScore'].max()
            minScore = df_temp['sentimentScore'].min()
            avgScore = df_temp['sentimentScore'].mean()
            varScore = df_temp['sentimentScore'].var()
            sumScore = df_temp['sentimentScore'].sum()

            self.df_feature.loc[featureIndex, 'newsCount']=newsCount
            self.df_feature.loc[featureIndex, 'sentimentPositive']=sentimentPositive
            self.df_feature.loc[featureIndex, 'sentimentNeural']=sentimentNeural
            self.df_feature.loc[featureIndex, 'sentimentNegative']=sentimentNegative
            self.df_feature.loc[featureIndex, 'stmPstPct']=stmPstPct
            self.df_feature.loc[featureIndex, 'stmNrPct']=stmNrPct
            self.df_feature.loc[featureIndex, 'stmNgtPct']=stmNgtPct
            self.df_feature.loc[featureIndex, 'maxScore']=maxScore
            self.df_feature.loc[featureIndex, 'minScore']=minScore
            self.df_feature.loc[featureIndex, 'avgScore']=avgScore
            self.df_feature.loc[featureIndex, 'varScore']=varScore
            self.df_feature.loc[featureIndex, 'sumScore']=sumScore

            # 获取tradeAndFactorFeature
            df_trade_temp=df_trade.loc[index-self.trade_interval-1:index,:]
            df_trade_temp.reset_index(drop=True,inplace=True)

            # 股价因子，mean函数，批量处理
            for column in self.columns:
                self.df_feature.loc[featureIndex, column] = df_trade_temp[column].mean()

            # 股价因子, 非mean函数处理
            self.df_feature.loc[featureIndex, 'openPrice'] = df_trade_temp.loc[0, 'openPrice']
            self.df_feature.loc[featureIndex, 'closePrice'] = df_trade_temp.loc[self.trade_interval-1, 'closePrice']
            self.df_feature.loc[featureIndex, 'dealAmount'] = df_trade_temp['dealAmount'].sum()
            self.df_feature.loc[featureIndex, 'chgPct'] = (df_trade_temp.loc[self.trade_interval-1, 'closePrice'] / df_trade_temp.loc[0, 'closePrice']) - 1

            # 标签是未来 predict_interval 天的收益率
            futureEnd = df_trade.loc[index+1, 'tradeDate']
            futureStart = df_trade.loc[index + self.predict_interval, 'tradeDate']

            # 以当天股价作为标签
            # label_pct = df_trade.loc[index, 'chgPct']
            label_pct = (df_trade.loc[index + self.predict_interval, 'closePrice'] / df_trade.loc[index+1, 'preClosePrice']) - 1

            label_binary = 1
            if label_pct <= 0:
                label_binary = 0
            self.df_feature.loc[featureIndex, 'label_pct']=label_pct
            self.df_feature.loc[featureIndex, 'label_binary']=label_binary

            featureIndex += 1

        ticker = str(int(self.df_feature.loc[0,'ticker']))
        while len(ticker) < 6:
            ticker = '0' + ticker

        self.df_feature['secID']=self.df_feature.loc[0,'secID']
        self.df_feature['ticker']=ticker
        self.df_feature['secShortName']=self.df_feature.loc[0,'secShortName']
        self.df_feature['exchangeCD']=self.df_feature.loc[0,'exchangeCD']
        self.df_feature.drop(columns='tradeDate',axis=1,inplace=True)

        self.df_feature.reset_index(drop=True, inplace=True)

        self.df_feature = self.df_feature.loc[:, ~self.df_feature.columns.str.contains('Unnamed')]
        self.df_feature.to_csv(f"{self.feature_save_path}{companyName}_feature_{self.trade_interval}.csv")

    # 合并当前目录下所有csv文件
    @staticmethod
    def concat_all_company(file_path, trade_interval):
        files = glob.glob(f"{file_path}*.csv")
        df = pd.read_csv(files[0])
        for f in files[1:]:
            df_temp = pd.read_csv(f)
            df = pd.concat([df, df_temp], axis=0)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = df.loc[:, ~df.columns.str.contains('Unnamed: 0')]
        df.to_csv(f"{file_path}/all_feature_{trade_interval}.csv")


if __name__ == '__main__':
    # 交易日范围
    tradeDate_interval = 1
    predict_interval=1
    del_feature = True
    # 特征存储路径
    feature_save_path = f"data/HS300/feature/{tradeDate_interval}_{predict_interval}/"

    # 若文件路径存在，则清空内部文件，反之则新建
    if del_feature:
        if os.path.exists(feature_save_path):
            shutil.rmtree(feature_save_path)
            os.makedirs(feature_save_path)
        else:
            os.makedirs(feature_save_path)

    # 获取trade和news的信息
    trade_f = glob.glob('data/HS300/tradeData/*.csv')
    news_f = glob.glob('data/HS300/newsData_clear/*.csv')
    print(len(trade_f), len(news_f))

    for index in range(30):
        tf = trade_f[index]
        nf = news_f[index]
        print(tf, nf, index)

        df_t = pd.read_csv(tf)
        df_f = pd.read_csv(nf)

        model = FeatureAndLabel(df_t, df_f, tradeDate_interval, predict_interval,feature_save_path)
        model.get_feature_windows()

    FeatureAndLabel.concat_all_company(feature_save_path, tradeDate_interval)
