#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Stanl
# @Time     : 2023/5/7 21:19
# @File     : prediction.py.py
# @Project  : StockValuePrediction

import argparse
import os

import numpy as np
import tensorflow as tf
from official.nlp import optimization
from tensorflow import keras


def tfcm(raw_ds, callbacks):
    train_data = np.vectorize(id2text := lambda x: raw_ds['news'][x])(np.array(raw_ds['train_reg'])[:, 0])
    train_label = np.array(raw_ds['train_reg'])[:, 1].astype(np.float32)
    valid_data = np.vectorize(id2text)(np.array(raw_ds['valid_reg'])[:, 0])
    valid_label = np.array(raw_ds['valid_reg'])[:, 1].astype(np.float32)
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_data, valid_label))
    AUTOTUNE = tf.data.AUTOTUNE
    BUFFER_SIZE = 512
    BATCH_SIZE = 32
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    valid_ds = valid_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    num_train = len(train_ds)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = tf.metrics.MeanAbsoluteError()
    epochs = 1
    num_train_steps = num_train * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    init_lr = 1e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    title_units = 16
    content_units = 64
    fine_tune = True
    from model import tfcm

    if tf.__version__.startswith('2.10'):
        os.environ['TFHUB_CACHE_DIR'] = 'model/hub_cache'
    regressioner = tfcm.build_regression_model(title_units, content_units, fine_tune=fine_tune)
    from utils.visualization import plot_structure, plot_history

    plot_structure(regressioner, 'bert_regressioner.png')
    regressioner.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = regressioner.fit(train_ds, validation_data=valid_ds, epochs=epochs, callbacks=[callbacks])
    test_data = np.vectorize(id2text)(np.array(list(raw_ds['test_reg'])))
    test_label = regressioner.predict(test_data)
    scores = dict(raw_ds['score'], **dict(zip(raw_ds['test_reg'], test_label.flatten())))
    test_data = [[scores[ID] for ID in line] for line in raw_ds['test']]
    train_data, train_label = raw_ds['train_cls']
    valid_data, valid_label = raw_ds['valid_cls']
    maxlen = max(max(len(i) for i in test_data), max(len(i) for i in train_data), max(len(i) for i in valid_data))
    pad_value = 0.49
    test_ds = tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=maxlen, padding='post', value=pad_value,
                                                            dtype='float32')
    train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=maxlen, padding='post',
                                                               value=pad_value, dtype='float32')
    valid_data = tf.keras.preprocessing.sequence.pad_sequences(valid_data, maxlen=maxlen, padding='post',
                                                               value=pad_value, dtype='float32')
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_label)).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE).cache().prefetch(AUTOTUNE)
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_data, valid_label)).batch(BATCH_SIZE).cache().prefetch(
        AUTOTUNE)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [tf.metrics.BinaryAccuracy(),
               tf.metrics.Precision(),
               tf.metrics.Recall(),
               tf.metrics.AUC()]
    epochs = 5
    init_lr = 1e-4
    optimizer = tf.keras.optimizers.Nadam(learning_rate=init_lr)
    config = [
        {'units': 64, 'activation': 'elu', 'regularizer': ('l2', 1e-2), 'batch_norm': True, 'dropout': 0},
        {'units': 16, 'activation': 'softplus', 'regularizer': ('l1', 1e-3), 'batch_norm': False, 'dropout': 0}
    ]
    classifier = tfcm.build_classifier_model(maxlen, config)
    classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = classifier.fit(train_ds, validation_data=valid_ds, epochs=epochs, callbacks=[callbacks])
    plot_history(history.history)
    return classifier, test_ds


def w2v(raw_ds, callbacks):
    dim = raw_ds['model'].vector_size
    mat = raw_ds['mat']
    raw_train = np.array(raw_ds['train'], dtype=list)
    raw_valid = np.array(raw_ds['valid'], dtype=list)

    train_label = raw_train[:, 1]
    valid_label = raw_valid[:, 1]
    titleK, contentK = 1, 4
    maxlen = 64
    train_pad = keras.preprocessing.sequence.pad_sequences(raw_train[:, 0], maxlen=maxlen, padding='post').astype('<U8')
    valid_pad = keras.preprocessing.sequence.pad_sequences(raw_valid[:, 0], maxlen=maxlen, padding='post').astype('<U8')
    test_pad = keras.preprocessing.sequence.pad_sequences(raw_ds['test'], maxlen=maxlen, padding='post').astype('<U8')

    def reshapeData(pad, mat, dim, maxlen, K):
        data = np.empty((pad.shape[0], dim, maxlen, K))
        for i in range(pad.shape[0]):
            for j in range(maxlen):
                data[i, :, j] = mat[pad[i, j]].T
        return data.reshape((pad.shape[0], dim, -1))

    train_data = reshapeData(train_pad, mat, dim, maxlen, titleK + contentK)
    valid_data = reshapeData(valid_pad, mat, dim, maxlen, titleK + contentK)
    test_data = reshapeData(test_pad, mat, dim, maxlen, titleK + contentK)

    train_data = tf.data.Dataset.from_tensor_slices(train_data.astype(np.float32))
    valid_data = tf.data.Dataset.from_tensor_slices(valid_data.astype(np.float32))
    test_data = tf.data.Dataset.from_tensor_slices(test_data.astype(np.float32))

    train_label = tf.data.Dataset.from_tensor_slices(train_label.astype(np.int32))
    valid_label = tf.data.Dataset.from_tensor_slices(valid_label.astype(np.int32))

    train_ds = tf.data.Dataset.zip((train_data, train_label))
    valid_ds = tf.data.Dataset.zip((valid_data, valid_label))

    AUTOTUNE = tf.data.AUTOTUNE
    BUFFER_SIZE = 512
    BATCH_SIZE = 32

    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    valid_ds = valid_ds.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
    test_ds = test_data.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

    from model import w2v

    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [keras.metrics.BinaryAccuracy(),
               keras.metrics.Precision(name='precision'),
               keras.metrics.Recall(name='recall'),
               keras.metrics.AUC(name='auc')]
    optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-4)
    epochs = 5
    maxlenK = maxlen * (titleK + contentK)

    model = w2v.build_model((dim, maxlenK), optimizer, loss, metrics)
    history = model.fit(train_ds, validation_data=valid_ds, epochs=epochs, callbacks=callbacks)

    from utils.visualization import plot_history, plot_structure

    plot_structure(model, 'word2vec.png')
    plot_history(history.history)
    return model, test_ds


def bert(raw_ds, callbacks):
    AUTOTUNE = tf.data.AUTOTUNE
    BUFFER_SIZE = 256
    BATCH_SIZE = 32

    def gen(name='train'):
        for newsID, label in raw_ds[name]:
            titles = ' '.join([raw_ds['news'][i][0] for i in newsID])
            contents = ' '.join([raw_ds['news'][i][1] for i in newsID])
            yield (titles, contents), label

    def gen_test():
        for newsID in raw_ds['test']:
            titles = ' '.join([raw_ds['news'][i][0] for i in newsID])
            contents = ' '.join([raw_ds['news'][i][1] for i in newsID])
            yield titles, contents

    train_ds = tf.data.Dataset.from_generator(
        gen, output_signature=(
            tf.TensorSpec(shape=(2,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    valid_ds = tf.data.Dataset.from_generator(
        lambda: gen('valid'), output_signature=(
            tf.TensorSpec(shape=(2,), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_ds = tf.data.Dataset.from_generator(
        gen_test, output_signature=(
            tf.TensorSpec(shape=(2,), dtype=tf.string)
        )
    ).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    num_train = train_ds.reduce(0, lambda x, _: x + 1).numpy()

    from model import bert

    tfhub_handle_encoder, tfhub_handle_preprocess = bert.load_handle()

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = tf.metrics.BinaryAccuracy()
    epochs = 20
    num_train_steps = num_train * epochs
    num_warmup_steps = int(0.1 * num_train_steps)
    init_lr = 1e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    title_units = 32
    content_units = 128

    bert_model = bert.build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder, title_units, content_units,
                                             fine_tune=False)
    from utils.visualization import plot_structure, plot_history

    plot_structure(bert_model, 'bert_classifier.png')

    bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(f'Training model with {tfhub_handle_encoder}')
    history = bert_model.fit(train_ds, validation_data=valid_ds, epochs=epochs, callbacks=[callbacks])
    plot_history(history.history)
    return bert_model, test_ds


def main(model):
    print('The model we are using: ', model)
    if model in ('tfcm', 'w2v', 'bert'):
        from utils.preprocessing import preprocess
        raw_ds = preprocess(model)
        callbacks = tf.keras.callbacks.TensorBoard(log_dir='logs')
        if model == 'tfcm':
            classifier, test_ds = tfcm(raw_ds, callbacks)
        elif model == 'w2v':
            classifier, test_ds = w2v(raw_ds, callbacks)
        else:
            classifier, test_ds = bert(raw_ds, callbacks)
        from utils.prediction import predict

        predict(classifier, test_ds, model + '_prediction.txt')
    else:
        print('Invalid model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Value Prediction')
    parser.add_argument('--model', '-m', type=str, default='tfcm', help='model to use')
    args = parser.parse_args()
    tf.get_logger().setLevel('ERROR')
    print('Configure of GPU:')
    print('The version of TensorFlow used in this project {}'.format(tf.__version__))
    print(tf.config.list_physical_devices('GPU'))
    main(args.model)
