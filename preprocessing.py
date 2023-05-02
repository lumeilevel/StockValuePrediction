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
from collections import defaultdict

raw = 'data/news.txt'
train = 'data/train.txt'
test = 'data/test.txt'


def processContent(content):
    news = eval(content)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        news['title'] = BeautifulSoup(news['title'].lower(), 'html.parser').get_text().replace('\n', '')
        news['content'] = BeautifulSoup(news['content'].lower(), 'html.parser').get_text().replace('\n', '')
    news['content'] = re.sub(r'\s+', '', news['content'])
    stopWords = "：，。；;、（）【】“”‘’《》?？！!~`·<>#￥$@*^_|/——①②③④⑤⑥⑦⑧⑨⑩(),&的个并且么之也些于" \
                "以诸等们乎而和即及叫吧呀呗呵哪尔拿宁你我他她它将就尽已得彼怎打把被替故某着给若虽让赶起然那随除出儿"
    for i in stopWords:
        news['content'] = news['content'].replace(i, '')
    return news


def bertData():
    f = codecs.open(raw, "r", "utf-8")
    news = [processContent(i) for i in f.readlines()]
    f.close()
    # convert it to a dictionary with `id` as key
    news = {str(d['id']): (d['title'], d['content']) for d in news}
    return news


def getRawData(dataDir):
    name = dataDir[5:-4]
    with codecs.open(dataDir, 'r', 'utf-8') as f:
        if name == 'train':
            lines = [(i.split()[1].split(','), (eval(i.split()[0]) + 1) // 2) for i in f.readlines()]
        else:
            lines = [i.strip().split(',') for i in f.readlines()]
    return lines


def eda(news):
    # print the number of news
    print("Total number of news: {}".format(len(news)))
    rawTrain = getRawData(train)
    rawTest = getRawData(test)
    print('Maximum number of news for each group in training set: {}'.format(max(len(i[0]) for i in rawTrain)))
    print('Minimum number of news for each group in training set: {}'.format(min(len(i[0]) for i in rawTrain)))
    print('Maximum number of news for each group in test set: {}'.format(max(len(i) for i in rawTest)))
    print('Minimum number of news for each group in test set: {}'.format(min(len(i) for i in rawTest)))
    numTrainID = dict.fromkeys(news, 0)
    numTestID = dict.fromkeys(news, 0)
    for line in rawTrain:
        for i in line[0]:
            numTrainID[i] += 1
    for line in rawTest:
        for i in line:
            numTestID[i] += 1
    print('Maximum number of times a news appears in training set: {}'.format(max(numTrainID.values())))
    print('Minimum number of times a news appears in training set: {}'.format(min(numTrainID.values())))
    print('Maximum number of times a news appears in test set: {}'.format(max(numTestID.values())))
    print('Minimum number of times a news appears in test set: {}'.format(min(numTestID.values())))
    return rawTrain, rawTest, numTrainID, numTestID


def getValidData(raw_train, model, validation_split=0.1):
    split = int(len(raw_train) * (1 - validation_split))
    if model == 'tfcm':
        raw_train = [(k, v) for k, v in raw_train.items()]
    else:
        raise ValueError("Invalid model name!")
    random.shuffle(raw_train)
    return raw_train[:split], raw_train[split:]


def id2vec(raw_train, score, validation_split=0.1):
    trainVec = [[score[ID] for ID in line[0]] for line in raw_train]
    random.shuffle(trainVec)
    split = int(len(trainVec) * (1 - validation_split))
    return trainVec[:split], trainVec[split:]


def getScore(raw_train):
    # get the score of each news
    score_dict = [defaultdict(int) for _ in range(2)]
    for i in raw_train:
        for j in i[0]:
            score_dict[i[1]][j] += 1
            score_dict[1 - i[1]][j] += 0
    return {k: score_dict[1][k] / (v + score_dict[1][k]) for k, v in score_dict[0].items()}


def getDistribution(raw_train, raw_valid):
    # get the distribution of the split training set and validation set
    trainD = (sum(v[1] == 0 for v in raw_train), sum(v[1] == 1 for v in raw_train), len(raw_train))
    validD = (sum(v[1] == 0 for v in raw_valid), sum(v[1] == 1 for v in raw_valid), len(raw_valid))
    totalD = (trainD[0] + validD[0], trainD[1] + validD[1], trainD[2] + validD[2])
    return trainD, validD, totalD


def preprocess(model):
    raw_ds = {'news': bertData()}
    raw_ds['train'], raw_ds['test'], numTrainID, numTestID = eda(raw_ds['news'])
    if model == 'bert':
        raw_ds['train'], raw_ds['valid'] = getValidData(raw_ds['train'], 'bert')
    elif model == 'tfcm':
        raw_ds['score'] = getScore(raw_ds['train'])
        raw_ds['test_reg'] = set([ID for line in raw_ds['test'] for ID in line if ID not in raw_ds['score']])
        raw_ds['train_reg'], raw_ds['valid_reg'] = getValidData(raw_ds['score'], 'tfcm')
        raw_ds['train'], raw_ds['valid'] = id2vec(raw_ds['train'], raw_ds['score'])
    else:
        raise ValueError('Invalid model name!')
    # raw_ds['distribution'] = getDistribution(raw_ds['train'], raw_ds['valid'])
    return raw_ds
