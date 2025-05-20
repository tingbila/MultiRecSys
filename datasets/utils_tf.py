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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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
    使用 LabelEncoder 的核心目的之一就是：将类别型特征转换为从 0 开始的整数索引，以便构建 embedding 矩阵时可以作为下标使用。
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
    label_encoders_dict = {} # 后面可能会用到
    for f in tqdm(feats, desc='process_sparse_feats'):
        label_encoder = LabelEncoder()  # 创建 LabelEncoder 实例
        data[f] = label_encoder.fit_transform(data[f])  # 编码特征
        label_encoders_dict[f] = label_encoder

    # for feat, encoder in label_encoders.items():
    #     # print(f"Feature: {feat}")
    #     mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
    #     # print("原始值 -> 编码值 映射前5项:", list(mapping.items())[:5])
    """
    Feature: item_id
    原始值 -> 编码值 映射前5项: [(224, 0), (426, 1), (1565, 2), (4273, 3), (4297, 4)]
    """
    return data,label_encoders_dict


def process_sequence_feats(data, sequence_feats):
    """
    这段代码 process_sequence_feats(data, sequence_feats) 是用于处理变长序列特征（variable-length sequence features）的，常用于推荐系统或文本类模型中，
    将某些字段中以分隔符（如英文逗号,或竖线|）分隔的多个元素转化为 稀疏整数序列（token ids），以便后续送入嵌入层或模型进行训练。
    把 DataFrame 中的每个变长序列特征字段（如 genres, actors）从字符串表示（如 "action,comedy"）变成整数 ID 的列表（如 [2, 5]），并进行 padding（补零）
    在 Tokenizer 的 fit_on_texts() 操作之后，相当于为每个出现过的单词分配了一个唯一的整数索引（index），这个过程就像构建了一个词表（vocabulary），
    每个 word → id，通过tokenizer.word_index可以进行查看。
    {'OOV': 1, '成龙': 2, '张学友': 3, '巩俐': 4, '舒淇': 5, '小李子': 6, '汤姆·哈迪': 7, '马特·达蒙': 8, '刘德华': 9, '周星驰': 10, '李连杰': 11, '艾米丽·布朗特': 12}
    # OOV 词保留为 1（用于未见过的词）
    """
    tokenizers = {}          # 每个变长特征对应一个独立的 Tokenizer，用于后续文本转索引
    # 遍历所有变长序列特征
    for feature in sequence_feats:
        # 将 '|' 分隔转为空格，适配 Tokenizer 格式， Tokenizer 默认是按 空格（whitespace）分割输入文本的
        texts = data[feature].fillna('').apply(lambda x: x.replace(',', ' ')).tolist()   # ['action comedy', 'drama', '', 'thriller horror']
        """
        设置 oov_token='OOV' 的作用是：
        在词典中添加一个专门的标记，比如 'OOV'，用于代表所有未登录词（词表中没有的词）。
        当你调用 tokenizer.texts_to_sequences() 处理新文本时，遇到没见过的词就会自动用 OOV 的索引来代替。
        """
        tokenizer = Tokenizer(oov_token='OOV')
        tokenizer.fit_on_texts(texts) # 把所有序列填充成等长（按最长的补 0）
        sequences = tokenizer.texts_to_sequences(texts)
        # 把 NumPy 的二维数组 padded 每一行变成一个 list，然后整体作为一个新的列表赋值给 data[feature]，从而更新 DataFrame 的某一列，
        # 每个元素是一个 list（而不是一个 ndarray）
        data[feature] = list(pad_sequences(sequences, padding='post'))    # shape: (num_samples, max_seq_len)
        # 130  136         57       60         16         54        0       0     0         0      71  0.475954       1.437616  [2, 0]   [11, 6]
        # 131  249         33       21        168         12        0       1     0        50     223  0.507134      -0.392077  [4, 6]    [9, 8]
        tokenizers[feature] = tokenizer

    # 对于离散pad数据，某一行元素就是[2, 3, 0, 0, 0, 0]这种格式
    return data, tokenizers




def process_history_sequence_feats(data, history_sequence_feats, history_sequence_label_encoder_map_config, label_encoders_dict, maxlen=5):
    """
    对已经是整数 ID 表示的历史序列特征进行处理：分割 -> 编码 -> padding。

    参数说明：
    :param data: pd.DataFrame，原始数据集，包含若干历史序列特征列（如 'history_item_ids'）
    :param history_sequence_feats: List[str]，历史序列特征名列表，例如 ['history_item_ids', 'history_citys']
    :param history_sequence_label_encoder_map_config: Dict[str, str]，每个历史序列特征对应其主特征的编码器名，例如：
           {
               "history_item_ids": "item_id",
               "history_citys": "item_city"
           }
    :param label_encoders: Dict[str, LabelEncoder]，主特征名 -> 已训练好的 LabelEncoder 对象
    :param maxlen: int，序列最大长度，多余部分会被截断，不足会用 0 补齐（post-padding）

    :return: pd.DataFrame，处理后的数据，历史序列列中的值为长度固定的整数列表（padding 后结果）
    """
    for feature in history_sequence_feats:
        # 1. 分割原始字符串为 token 序列（例如 '12,45,7' -> ['12', '45', '7']）
        sequences = data[feature].fillna('').apply(lambda x: x.split(','))

        # 2. 获取当前序列特征所对应的主特征编码器（如 history_item_ids -> item_id 的 LabelEncoder）
        encoder = label_encoders_dict[history_sequence_label_encoder_map_config[feature]]

        # 3. 构建 token -> index 映射表（使用 str(cls) 是因为有的 LabelEncoder 中的类为字符串）
        token2index = {str(cls): idx for idx, cls in enumerate(encoder.classes_)}

        # 4. 对每个 token 编码，若未登录则映射为 0
        encoded_sequences = sequences.apply(lambda seq: [token2index.get(token, 0) for token in seq])

        # 5. 进行 padding（后补零）统一长度
        padded = pad_sequences(encoded_sequences.tolist(), padding='post', maxlen=maxlen)

        # 6. 用填充后的整数序列更新原始列
        data[feature] = list(padded)

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
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel","finish", "like", "music_id", "device", "time", "duration_time", "actors", "genres", "history_item_ids", "history_citys"]
    data = pd.read_csv(file_path, sep='\t', names=column_names)  # 大数据集
    print(data.head(5))

    # 区分数值特征和稀疏特征
    dense_feats    = ["time", "duration_time"]
    sparse_feats   = ["uid", "user_city", "item_id", "author_id", "item_city", "channel","music_id", "device"]
    sequence_feats = ['actors', 'genres']   # 有的时候可能为空列表 []
    history_sequence_feats = ["history_item_ids", "history_citys"]   # 增加变长历史序列数据
    # 下面2个配置其实可以合并为一个，历史原因:
    # 历史序列特征 -> 主特征编码器名称的映射配置（用于 LabelEncoder 查找）
    history_sequence_label_encoder_map_config = {"history_item_ids":'item_id',"history_citys":'item_city'}
    # 历史序列特征 -> 主特征 embedding 源的映射配置（用于模型构建时查找共享 embedding）
    history_sequence_emb_map_config = [{'feat': 'history_item_ids', 'target_emb_column': 'item_id', 'target_item_index': 2}, {'feat': 'history_citys','target_emb_column': 'item_city','target_item_index': 4}]

    # 对数值特征、稀疏特征、序列特征进行处理
    data = process_dense_feats(data,  dense_feats)
    # 你必须对历史序列字段和当前字段使用相同的 LabelEncoder 实例进行 transform，不能重新 fit
    data, label_encoders_dict  = process_sparse_feats(data, sparse_feats)

    if sequence_feats:
        data, tokenizers = process_sequence_feats(data, sequence_feats)
    else:
        tokenizers = {}

    if history_sequence_feats:
        data = process_history_sequence_feats(data,history_sequence_feats,history_sequence_label_encoder_map_config,label_encoders_dict)

    print(data.head(5))
    """
       uid  user_city  item_id  author_id  item_city  channel  finish  like  music_id  device      time  duration_time  actors   genres        history_item_ids         history_citys
    0  259         31       26        210          1        0       0     0        82     202  0.250521       3.830290  [4, 0]   [9, 0]  [247, 75, 176, 197, 0]  [14, 28, 29, 103, 0]
    1   24          4       27        244          2        0       1     0        85     151  0.467755       0.593142  [8, 7]   [6, 0]      [149, 95, 0, 0, 0]     [33, 54, 0, 0, 0]
    2   10        126      130        265          3        0       0     0       113      97  0.516757      -1.095805  [7, 3]  [11, 2]     [137, 284, 0, 0, 0]     [68, 81, 0, 0, 0]
    3   90          6      131        266          4        0       0     0         0     112  0.519817      -0.392077  [5, 6]  [11, 7]       [139, 0, 0, 0, 0]      [32, 0, 0, 0, 0]
    4  242         50      132        202          5        0       1     0         0     155  0.502151       1.015379  [6, 0]   [4, 5]   [244, 214, 241, 0, 0]   [86, 87, 108, 0, 0]
    """

    # 构建特征字典列表，用于模型输入
    feat_columns = [
        [dense_feat(feat) for feat in dense_feats],    # 数值特征的字典列表
        [sparse_feat(feat, len(data[feat].unique())) for feat in sparse_feats],  # 稀疏特征的字典列表
    ]
    if sequence_feats:
        feat_columns.append([sparse_feat(feat, len(tokenizers[feat].word_index) + 1) for feat in sequence_feats])   # 序列稀疏特征的字典列表

    # 离散历史序列特征需要自己进行改这里代码
    if history_sequence_feats:
        feat_columns.append(history_sequence_emb_map_config)

    # [[{'feat': 'time'}, {'feat': 'duration_time'}],
    #  [{'feat': 'uid', 'feat_num': 289},....,{'feat': 'author_id', 'feat_num': 289}],
    #  [{'feat': 'actors', 'feat_num': 10}, {'feat': 'genres', 'feat_num': 13}]]

    train_data, test_data  = train_test_split(data,       test_size=test_size, random_state=42)
    train_data, valid_data = train_test_split(train_data, test_size=test_size, random_state=42)

    # 将pandas 的 DataFrame 数据转换成 TensorFlow 的 tf.data.Dataset，以供模型训练使用
    def df_to_dataset(df):
        sparse_tensor = tf.convert_to_tensor(df[sparse_feats].values, dtype=tf.int32)
        dense_tensor  = tf.convert_to_tensor(df[dense_feats].values, dtype=tf.float32)

        # 分别处理 sequence_feats 和 history_sequence_feats
        seq_input_tensors = []
        for feat in sequence_feats:
            tensor = tf.convert_to_tensor(df[feat].tolist(), dtype=tf.int32)
            seq_input_tensors.append(tensor)

        history_input_tensors = []
        for feat in history_sequence_feats:
            tensor = tf.convert_to_tensor(df[feat].tolist(), dtype=tf.int32)
            history_input_tensors.append(tensor)

        # 构建最终输入顺序（必须和模型中 call 函数保持一致）
        input_features = (sparse_tensor,dense_tensor,*seq_input_tensors,*history_input_tensors)

        labels = tf.convert_to_tensor(df[["finish", "like"]].values, dtype=tf.float32)
        # 将输入 (sparse_tensor, dense_tensor) 和标签 labels 打包成一个 tf.data.Dataset
        labels_finish = labels[:, 0]
        labels_like   = labels[:, 1]

        # *sequence_tensors 相当于传入sequence_tensors[0]、sequence_tensors[1].....
        return tf.data.Dataset.from_tensor_slices((input_features, {'finish': labels_finish, 'like': labels_like}))


    train_ds = df_to_dataset(train_data)
    valid_ds = df_to_dataset(valid_data)
    test_ds  = df_to_dataset(test_data)

    train_ds = train_ds.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return data, train_ds, valid_ds,test_ds, feat_columns