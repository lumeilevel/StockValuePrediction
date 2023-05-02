#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/2 17:05
# @File     : prediction.py
# @Project  : StockValuePrediction

import os


def predict(model, data, name='prediction.txt', path='../data/prediction/'):
    prediction = model.predict(data)
    with open(os.path.join(path, name), 'w') as f:
        for i in prediction:
            f.write(str(i) + '\n')
