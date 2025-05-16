# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media
# 排序任务

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
import os
from tensorflow.keras.models import load_model

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
        texts = data[feature].fillna("").apply(lambda x: x.replace('|', ' ')).tolist()
        tokenizer = Tokenizer(oov_token='OOV')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, padding='post')
        pad_sequences_dict[feature] = padded
        tokenizers[feature] = tokenizer
        pad_len_dict[feature] = padded.shape[1]

    return pad_sequences_dict, tokenizers, pad_len_dict



#------------------------接入DeepFM的预测流程如下:------------------------

# 第1步：准备 DSSM 召回的候选集数据（假设你拿到如下格式的召回数据（DSSM 输出））
# 2 [{'item_id': 238, 'score': 0.8049}, {'item_id': 123, 'score': 0.7505}, {'item_id': 199, 'score': 0.7226}, {'item_id': 20, 'score': 0.7224}, {'item_id': 255, 'score': 0.7192}]
# 153 [{'item_id': 147, 'score': 0.7923}, {'item_id': 286, 'score': 0.7895}, {'item_id': 44, 'score': 0.7708}, {'item_id': 230, 'score': 0.7634}, {'item_id': 220, 'score': 0.7569}]
# 135 [{'item_id': 106, 'score': 0.7972}, {'item_id': 124, 'score': 0.7661}, {'item_id': 194, 'score': 0.7625}, {'item_id': 203, 'score': 0.7362}, {'item_id': 238, 'score': 0.7334}]
# 130 [{'item_id': 238, 'score': 0.7683}, {'item_id': 255, 'score': 0.7457}, {'item_id': 123, 'score': 0.7436}, {'item_id': 20, 'score': 0.7301}, {'item_id': 230, 'score': 0.7231}]
# 93 [{'item_id': 233, 'score': 0.8776}, {'item_id': 50, 'score': 0.8662}, {'item_id': 167, 'score': 0.8643}, {'item_id': 225, 'score': 0.8583}, {'item_id': 181, 'score': 0.8504}]
# 154 [{'item_id': 238, 'score': 0.8299}, {'item_id': 194, 'score': 0.7699}, {'item_id': 106, 'score': 0.7609}, {'item_id': 123, 'score': 0.7477}, {'item_id': 281, 'score': 0.7461}]
# 123 [{'item_id': 205, 'score': 0.9002}, {'item_id': 150, 'score': 0.887}, {'item_id': 35, 'score': 0.8672}, {'item_id': 178, 'score': 0.8658}, {'item_id': 245, 'score': 0.8574}]
# 204 [{'item_id': 238, 'score': 0.7558}, {'item_id': 255, 'score': 0.7472}, {'item_id': 123, 'score': 0.7371}, {'item_id': 230, 'score': 0.7304}, {'item_id': 20, 'score': 0.7268}]
# 188 [{'item_id': 270, 'score': 0.8424}, {'item_id': 30, 'score': 0.8218}, {'item_id': 241, 'score': 0.8022}, {'item_id': 203, 'score': 0.8006}, {'item_id': 17, 'score': 0.7887}]


# 第2步：为这些 user/item 拼接特征，通过user_profile_df和item_profile_df关联画像特征，为每个 (uid, item_id) 构建完整的特征向量：
# input_records = [
#     {
#         "uid": "u1",
#         "item_id": "i1",
#         "user_city": 12,
#         "device": 3,
#         "time": 200,
#         "item_city": 21,
#         "author_id": 101,
#         "music_id": 501,
#         "genres": "A|B",
#         "actors": "X|Y|Z",
#         "duration_time": 20
#     },
#     {
#         "uid": "u1",
#         "item_id": "i2",
#         "user_city": 12,
#         "device": 3,
#         "time": 200,
#         "item_city": 21,
#         "author_id": 102,
#         "music_id": 502,
#         "genres": "C|D",
#         "actors": "Y",
#         "duration_time": 30,
#     }
# ]
#
# import pandas as pd
# input_df = pd.DataFrame(input_records)
# input_df
# uid	item_id	user_city	device	time	item_city	author_id	music_id	genres	actors	duration_time
# 0	u1	i1	12	3	200	21	101	501	A|B	X|Y|Z	20
# 1	u1	i2	12	3	200	21	102	502	C|D	Y	30


# 为了模拟数据，假设我们现在直接读取原样本数据:
column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like", "music_id","device", "time", "duration_time", "actors", "genres"]
data = pd.read_csv(r"D:\\software\\pycharm_repository\\StarMaker\\MultiRecSys\\data_files\\train_2.csv", sep='\t',names=column_names)
print(data.head(5))

# 特征定义
sparse_feats = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
dense_feats = ["time", "duration_time"]
sequence_feats = ['actors', 'genres']
target = ['finish']  # 推荐任务目标（预测的时候没有这列）


# 相同的数据预处理方式
# 模型训练和预测必须 共享同一个处理器，否则数值不一致会导致预测结果错误：但是这个步骤具体怎么做，我还没有想好
data = process_dense_feats(data, dense_feats)
data = process_sparse_feats(data, sparse_feats)
pad_sequences_dict, tokenizers, pad_len_dict = process_sequence_feats(data, sequence_feats) if sequence_feats else ({}, {}, {})

fixlen_feature_names = sparse_feats + dense_feats + sequence_feats
test_model_input = {name: data[name] for name in fixlen_feature_names}
if sequence_feats:
    for feat in sequence_feats:
        test_model_input[feat] = pad_sequences_dict[feat]


# === 模型评估(加载训练好的模型进行测试) ===
from deepctr.models import DeepFM
fixlen_feature_columns = [
                             SparseFeat(feat, data[feat].nunique() + 1, embedding_dim=4) for feat in sparse_feats
                         ] + [
                             DenseFeat(feat, 1) for feat in dense_feats
                         ]
if sequence_feats:
    for feat in sequence_feats:
        fixlen_feature_columns.append(
            VarLenSparseFeat(
                SparseFeat(feat, vocabulary_size=len(tokenizers[feat].word_index) + 1, embedding_dim=4),
                maxlen=pad_len_dict[feat], combiner='mean')
        )


linear_feature_columns = fixlen_feature_columns
dnn_feature_columns    = fixlen_feature_columns


model = DeepFM(
    linear_feature_columns=linear_feature_columns,
    dnn_feature_columns=dnn_feature_columns,
    task='binary'
)

output_model_file = os.path.join(r'D:\software\pycharm_repository\StarMaker\MultiRecSys\outputs\callbacks', 'best_model.h5')
model.load_weights(output_model_file)  # 只加载权重，不加载结构


# 虽然在预测时用了batch_size=256，但是模型只是分批地处理数据，但最终会拼接每个 batch 的结果，返回一个完整的预测结果数组**
pred_scores  = model.predict(test_model_input, batch_size=256)
data["pred_scores"] = pred_scores
final_ranked = data.sort_values(by=["uid", "pred_scores"], ascending=[True, False])
print(final_ranked)

#      uid  user_city  item_id  ...  actors    genres  pred_scores
# 105    0         24       65  ...      动作    李连杰,巩俐     0.404151
# 119    1         45       75  ...      喜剧    小李子,舒淇     0.460335
# 285    2         11      265  ...      惊悚     汤姆·哈迪     0.389636
# 173    3          0       99  ...   喜剧,悬疑  汤姆·哈迪,舒淇     0.419267
# 227    4         98      276  ...   战争,剧情       刘德华     0.473192

# todo:
# import pickle
#
# class PreprocessorManager:
#     def __init__(self, scaler=None, encoders=None, tokenizers=None):
#         self.scaler = scaler
#         self.encoders = encoders or {}
#         self.tokenizers = tokenizers or {}
#
#     def save(self, filepath):
#         with open(filepath, 'wb') as f:
#             pickle.dump(self.__dict__, f)
#
#     @classmethod
#     def load(cls, filepath):
#         with open(filepath, 'rb') as f:
#             data = pickle.load(f)
#         return cls(**data)
