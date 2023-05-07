#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/7 21:33
# @File     : w2v.py
# @Project  : StockValuePrediction

from tensorflow import keras


def build_model(shape, optimizer, loss, metrics):
    model = keras.Sequential([
        keras.Input(shape=shape),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='softplus', kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(model.summary())
    return model
