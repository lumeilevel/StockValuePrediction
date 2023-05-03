#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/2 17:05
# @File     : prediction.py
# @Project  : StockValuePrediction

import os

import numpy as np


def predict(model, data, name='prediction.txt', path='./data/prediction/'):
    prediction = model.predict(data)
    lines = np.vectorize(lambda x: '+1\n' if x > 0.5 else '-1\n')(prediction).flatten().tolist()
    with open(os.path.join(path, name), 'w') as f:
        f.writelines(lines)
    return prediction
