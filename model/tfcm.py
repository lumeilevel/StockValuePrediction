#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/4/27 17:24
# @File     : tfcm.py
# @Project  : StockValuePrediction

import tensorflow as tf
import tensorflow_hub as hub


tfhub_handle_preprocess = 'https://hub.tensorflow.google.cn/tensorflow/bert_zh_preprocess/3'
tfhub_handle_encoder = 'https://hub.tensorflow.google.cn/tensorflow/bert_zh_L-12_H-768_A-12/4'


def build_regression_model(
        title_units,
        content_units,
        fine_tune=True,
        tfhub_handle_preprocess=tfhub_handle_preprocess,
        tfhub_handle_encoder=tfhub_handle_encoder
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
    net_title = tf.keras.layers.Dense(title_units, activation='relu', name='title')(net_title)
    net_content = tf.keras.layers.Dense(content_units, activation='relu', name='content')(net_content)
    net = tf.keras.layers.concatenate([net_title, net_content])
    net = tf.keras.layers.Dense(1, activation='sigmoid', name='regression')(net)
    return tf.keras.Model((title, content), net)
