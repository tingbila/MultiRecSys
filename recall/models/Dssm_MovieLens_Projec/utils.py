# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm
from tensorflow import keras
import argparse

import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个


# === 数值特征处理 ===
def process_dense_feats(data, feats):
    """
    使用 StandardScaler 对数值特征进行标准化处理。
    """
    data[feats] = data[feats].fillna(0)
    scaler = StandardScaler()
    data[feats] = scaler.fit_transform(data[feats])
    return data


# === 稀疏特征处理 ===
def process_sparse_feats(data, feats):
    """
    对稀疏特征进行填充缺失并编码为整数索引。
    """
    for f in tqdm(feats, desc='Processing Sparse Features'):
        label_encoder = LabelEncoder()
        data[f] = label_encoder.fit_transform(data[f].astype(str))
    return data


# === 变长序列特征处理 ===
def process_sequence_feats(data, sequence_features):
    """
    对变长特征使用自定义 Tokenizer 分词，并进行 padding 处理。
    """
    # 用于存储每个变长特征处理后的 padding 序列
    pad_sequences_dict = {}
    # 每个变长特征对应一个独立的 Tokenizer，用于后续文本转索引
    tokenizers = {}
    # 用于记录每个变长特征的 padding 长度（即序列被填充后的最大长度）
    pad_len_dict = {}

    for feature in sequence_features:
        # 将 ',' 分隔转为空格，适配 Tokenizer 格式， Tokenizer 默认是按 空格（whitespace）分割输入文本的
        texts = data[feature].fillna("").apply(lambda x: x.replace(',', ' ')).tolist()
        tokenizer = Tokenizer(oov_token='OOV')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, padding='post') # 把所有序列填充成等长（按最长的补 0）
        pad_sequences_dict[feature] = padded
        tokenizers[feature] = tokenizer
        pad_len_dict[feature] = padded.shape[1]


    return pad_sequences_dict, tokenizers, pad_len_dict



def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)