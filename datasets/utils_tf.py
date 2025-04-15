# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media



# 基础包
import os
import time
import argparse

# 数据处理和可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# sklearn 工具包
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# TensorFlow 和 Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# 设置 matplotlib 后端（根据系统环境选择）
import matplotlib
matplotlib.use('TkAgg')  # 你可以改成 'Agg' 或 'QtAgg'，看你本地 GUI 支持情况

# Pandas 显示选项（非必须，但开发调试时很有用）
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

from config.data_config import *



# 定义稀疏特征的函数，返回特征名和特征可能的取值数量
def sparse_feat(feat, feat_num):
    """
    构建稀疏特征的字典。

    参数:
    feat (str): 特征名
    feat_num (int): 特征的唯一取值数量

    返回:
    dict: 包含特征名和特征值数量的字典
    """
    return {'feat': feat, 'feat_num': feat_num}


# 定义数值特征的函数，返回特征名
def dense_feat(feat):
    """
    构建数值特征的字典。

    参数:
    feat (str): 特征名

    返回:
    dict: 包含特征名的字典
    """
    return {'feat': feat}


# 可以用下面的函数替代上面的这个
def process_dense_feats(data, feats):
    """
    使用 StandardScaler 对数值特征进行标准化处理。

    参数:
    data (DataFrame): 包含特征的数据框
    feats (list): 数值特征列名列表

    返回:
    DataFrame: 处理后的数据
    scaler: 训练好的标准化器（可保存用于线上预测）
    """
    data[feats] = data[feats].fillna(0)

    scaler = StandardScaler()
    data[feats] = scaler.fit_transform(data[feats])

    return data


# 处理稀疏特征，填充缺失值并编码
def process_sparse_feats(data, feats):
    """
    对稀疏特征进行处理，填充缺失值并使用 Label Encoding 编码。

    参数:
    data (DataFrame): 包含特征的数据框
    feats (list): 稀疏特征的列名列表

    返回:
    DataFrame: 处理后的数据框
    """
    # data[feats] = data[feats].fillna('-1')  # 填充缺失值为字符串 '-1'
    # 只读取指定的列
    # columns_to_read = ['label','I1','I2', 'C1', 'C2','C3']  # 替换为实际的列名
    # data = pd.read_csv(file, usecols=columns_to_read)

    # 对每个稀疏特征进行 Label Encoding 编码
    for f in tqdm(feats, desc='process_sparse_feats'):
        label_encoder = LabelEncoder()  # 创建 LabelEncoder 实例
        data[f] = label_encoder.fit_transform(data[f])  # 编码特征

    return data



# 创建数据集，处理特征并返回处理后的数据和特征信息
def create_dataset(file_path='./data/criteo_sampled_data.csv', embed_dim=5):
    """
    创建并处理数据集，包括数值和稀疏特征的处理。

    参数:
    file (str): 数据文件路径，默认为 './data/criteo_sampled_data.csv'
    embed_dim (int): 嵌入维度（未在此函数中使用）

    返回:
    data (DataFrame): 处理后的数据框
    feat_columns (list): 特征字典列表，包含数值和稀疏特征信息
    dense_feats (list): 数值特征列名列表
    sparse_feats (list): 稀疏特征列名列表
    """
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel",
                    "finish", "like", "music_id", "device", "time", "duration_time"]

    # data = pd.read_csv(file_path, names=column_names)
    data = pd.read_csv(file_path, sep='\t', names=column_names)  # 大数据集

    # 区分数值特征和稀疏特征
    sparse_feats  = ["uid", "user_city", "item_id", "author_id", "item_city", "channel","music_id", "device"]
    dense_feats   = ["time", "duration_time"]


    # 对数值特征和稀疏特征进行处理
    data = process_dense_feats(data,  dense_feats)
    data = process_sparse_feats(data, sparse_feats)

    # y = data[["finish", "like"]].values
    # X = data.drop(columns=["finish", "like"])
    # print(X.shape)  # (40, 234)


    # 构建特征字典列表，用于模型输入
    feat_columns = [
        [dense_feat(feat) for feat in dense_feats],                             # 数值特征的字典列表
        [sparse_feat(feat, len(data[feat].unique())) for feat in sparse_feats]  # 稀疏特征的字典列表
    ]

    train_data, test_data  = train_test_split(data,       test_size=test_size, random_state=42)
    train_data, valid_data = train_test_split(train_data, test_size=test_size, random_state=42)

    # 将pandas 的 DataFrame 数据转换成 TensorFlow 的 tf.data.Dataset，以供模型训练使用
    def df_to_dataset(df):
        sparse_tensor = tf.convert_to_tensor(df[sparse_feats].values, dtype=tf.int32)
        dense_tensor = tf.convert_to_tensor(df[dense_feats].values, dtype=tf.float32)
        labels = tf.convert_to_tensor(df[["finish", "like"]].values, dtype=tf.float32)
        # 将输入 (sparse_tensor, dense_tensor) 和标签 labels 打包成一个 tf.data.Dataset
        labels_finish = labels[:, 0]
        labels_like   = labels[:, 1]

        return tf.data.Dataset.from_tensor_slices(((sparse_tensor, dense_tensor), {'finish': labels_finish,'like': labels_like}))
        # return tf.data.Dataset.from_tensor_slices(((sparse_tensor, dense_tensor), (labels_finish, labels_like)))

    train_ds = df_to_dataset(train_data)
    valid_ds = df_to_dataset(valid_data)
    test_ds  = df_to_dataset(test_data)

    train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return data, train_ds, valid_ds,test_ds, feat_columns