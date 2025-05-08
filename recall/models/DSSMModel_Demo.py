# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ------------------------------------------------------------------------------
# -- author：张明阳
# -- create：2025年5月7日17:47:25
# -- function：DSSM-练习版本V1
# -- document:
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from DSSMModel import DSSMModel

import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat


def process_dense_feats(data, feats):
    data[feats] = data[feats].fillna(0)
    scaler = StandardScaler()
    data[feats] = scaler.fit_transform(data[feats])
    return data


def process_sparse_feats(data, feats):
    label_encoders = {}
    for feat in feats:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat].astype(str))
        label_encoders[feat] = le
    return data, label_encoders



def load_and_process_data():
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like", "music_id", "device", "time", "duration_time", "actors", "genres"]
    data = pd.read_csv("D:\\software\\pycharm_repository\\StarMaker\\MultiRecSys\\data_files\\train_2.csv", sep='\t', names=column_names)

    sparse_feats_user = ["uid", "user_city", "device"]
    sparse_feats_item = ["item_id", "author_id", "item_city", "channel", "music_id"]
    dense_feats = ["time", "duration_time"]
    target = 'finish'

    data = process_dense_feats(data, dense_feats)
    data, _ = process_sparse_feats(data, sparse_feats_user + sparse_feats_item)

    user_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique()+1, embedding_dim=8) for feat in sparse_feats_user] + \
                           [DenseFeat(feat, 1) for feat in dense_feats]
    item_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique()+1, embedding_dim=8) for feat in sparse_feats_item]

    feature_names = sparse_feats_user + sparse_feats_item + dense_feats
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train_model_input = {name: train[name].values for name in feature_names}
    test_model_input  = {name: test[name].values for name in feature_names}

    return train_model_input, test_model_input, train[target].values, test[target].values, user_feature_columns, item_feature_columns






if __name__ == "__main__":
    train_input, test_input, y_train, y_test, user_feature_columns, item_feature_columns = load_and_process_data()
    print(user_feature_columns)
    print(item_feature_columns)

    model = DSSMModel(user_feature_columns, item_feature_columns)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    history = model.fit(train_input, y_train, batch_size=256, epochs=10, validation_split=0.2, verbose=2)

    preds = model.predict(test_input, batch_size=256)
    print("Test LogLoss:", round(log_loss(y_test, preds), 4))
    print("Test AUC:", round(roc_auc_score(y_test, preds), 4))

    # 可视化
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.legend()
    plt.title('Training Performance')
    plt.xlabel('Epoch')
    plt.show()
