#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/4/27 17:24
# @File     : tfcm.py
# @Project  : StockValuePrediction

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras


def build_classifier_model(
        input_shape: int,  # shape of input data
        config: list  # list of dicts, each dict contains the config of a layer
):
    model = tf.keras.Sequential([keras.layers.InputLayer(input_shape=(input_shape,))])
    for layer in config:
        if layer['regularizer'][0] == 'l1':
            regularizer = keras.regularizers.l1(layer['regularizer'][1])
        elif layer['regularizer'][0] == 'l2':
            regularizer = keras.regularizers.l2(layer['regularizer'][1])
        elif layer['regularizer'][0] == 'l1_l2':
            regularizer = keras.regularizers.l1_l2(layer['regularizer'][1], layer['regularizer'][2])
        else:
            regularizer = None
        model.add(keras.layers.Dense(layer['units'], activation=layer['activation'], kernel_regularizer=regularizer))
        if layer['batch_norm']:
            model.add(keras.layers.BatchNormalization())
        if layer['dropout'] > 0:
            model.add(keras.layers.Dropout(layer['dropout']))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    return model


def build_regression_model(
        title_units,
        content_units,
        # final_units,
        fine_tune=True,
        tfhub_handle_preprocess='https://hub.tensorflow.google.cn/tensorflow/bert_zh_preprocess/3',
        tfhub_handle_encoder='https://hub.tensorflow.google.cn/tensorflow/bert_zh_L-12_H-768_A-12/4'
):
    title = tf.keras.layers.Input(shape=(), dtype=tf.string, name='title_text')
    content = tf.keras.layers.Input(shape=(), dtype=tf.string, name='content_text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs_title = preprocessing_layer(title)
    encoder_inputs_content = preprocessing_layer(content)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=not fine_tune, name='BERT_encoder')
    outputs_title = encoder(encoder_inputs_title)
    outputs_content = encoder(encoder_inputs_content)
    net_title = outputs_title['pooled_output']
    net_content = outputs_content['pooled_output']
    net_title = tf.keras.layers.Dense(title_units, activation='elu', name='title')(net_title)
    net_content = tf.keras.layers.Dense(content_units, activation='elu', name='content')(net_content)
    net = tf.keras.layers.concatenate([net_title, net_content])
    # net = tf.keras.layers.Dense(final_units, activation='softplus', name='final')(net)
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='regression')(net)
    return tf.keras.Model((title, content), net)
