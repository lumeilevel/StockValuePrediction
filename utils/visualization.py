#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/2 15:59
# @File     : visualization.py
# @Project  : StockValuePrediction

import os
from keras.utils import plot_model
from matplotlib import pyplot as plt


def plot_structure(model, name, path='model/structure/', dpi=128):
    os.environ["PATH"] += os.pathsep + 'D:/Graphviz/2.38/bin/'
    plot_model(model, to_file=os.path.join(path, name), dpi=dpi)


def plot_history(history_dict, figsize=(12, 6)):
    print(history_dict.keys())

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=figsize)
    # fig.tight_layout()

    plt.subplot(2, 1, 1)
    # r is for "solid red line"
    plt.plot(epochs, loss, 'ro-', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'bs--', label='Validation loss')
    plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'ro-', label='Training acc')
    plt.plot(epochs, val_acc, 'bs--', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
