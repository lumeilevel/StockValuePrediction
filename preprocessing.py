#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/4/24 14:12
# @File     : preprocessing.py
# @Project  : StockValuePrediction

import codecs
import re
import random
import warnings
from bs4 import BeautifulSoup

raw = 'data/news.txt'
train = 'data/train.txt'
test = 'data/test.txt'


def processContent(content):
    news = eval(content)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        news['title'] = BeautifulSoup(news['title'], 'html.parser').get_text().replace('\n', '')
        news['content'] = BeautifulSoup(news['content'], 'html.parser').get_text().replace('\n', '')
    news['content'] = re.sub(r'\s+', '', news['content'])
    stopWords = "：，。；、（）【】“”‘’《》？！①②③④⑤⑥⑦⑧⑨⑩(),"
    for i in stopWords:
        news['content'] = news['content'].replace(i, '')
    return news


def bertData(validation_split=0.1):
    f = codecs.open(raw, "r", "utf-8")
    news = [processContent(i) for i in f.readlines()]
    f.close()
    # shuffle news first
    random.shuffle(news)
    # convert it to a dictionary with `id` as key
    news = {str(d['id']): (d['title'], d['content']) for d in news}
    # trainID = dict.fromkeys(news, 0)
    # testID = dict.fromkeys(news, 0)
    # # count the number of times each news appears in train and test
    # with codecs.open(train, 'r', 'utf-8') as f:
    #     for line in f.readlines():
    #         trainID[processContent(line)] += 1
    return news


def countID(news, dataDir):
    name = dataDir[5:-4]
    # count the number of times each news appears in train or test
    numID = dict.fromkeys(news, 0)
    with codecs.open(dataDir, 'r', 'utf-8') as f:
        if name == 'train':
            lines = [i.split()[1].split(',') for i in f.readlines()]
        else:
            lines = [i.strip().split(',') for i in f.readlines()]
    print('Maximum number of news for each group in {}: {}'.format(name, max(len(i) for i in lines)))
    print('Minimum number of news for each group in {}: {}'.format(name, min(len(i) for i in lines)))
    for line in lines:
        for i in line:
            numID[i] += 1
    print('Maximum number of times a news appears in {}: {}'.format(name, max(numID.values())))
    print('Minimum number of times a news appears in {}: {}'.format(name, min(numID.values())))
    return numID


def eda(news):
    # print the number of news
    print("Total number of news: {}".format(len(news)))
    trainID = countID(news, train)
    testID = countID(news, test)
    return trainID, testID


news = bertData()
numTrainID, numTestID = eda(news)

