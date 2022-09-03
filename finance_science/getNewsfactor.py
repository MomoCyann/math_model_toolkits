import numpy as np
import pandas_method as pd
import datetime

def getNewsfactor(tradeData, newsData):
    # 新闻间隔
    interval = 10
    print(tradeData['tradeDate'].dtype)
    tradeData['tradeDate'] = pd.to_datetime(tradeData['tradeDate'])
    print(tradeData['tradeDate'].dtype)
    newsData['newsPublishTime'] = pd.to_datetime(newsData['newsPublishTime'])

    for index in tradeData.index:
        end_date = tradeData.loc[index, 'tradeDate']
        start_date = end_date - datetime.timedelta(interval)
        news = newsData.loc[newsData['newsPublishTime'] > start_date].loc[
            newsData['newsPublishTime'] < end_date]
        # 新闻数量统计，各极性统计
        newsCount = news['sentiment'].count()
        sentimentPositive = news.loc[news['sentiment'] == 1].count()['sentiment']
        sentimentNeural = news.loc[news['sentiment'] == 0].count()['sentiment']
        sentimentNegative = news.loc[news['sentiment'] == -1].count()['sentiment']
        print('get')

if __name__ == '__main__':
    name = '0_平安银行_trade_20-22.csv'
    tradeData = pd.read_csv('data/HS300/tradeData/0_平安银行_trade_20-22.csv')
    newsData = pd.read_csv('data/HS300/newsData_clear/0_平安银行_news_20-22.csv')
    getNewsfactor(tradeData, newsData)