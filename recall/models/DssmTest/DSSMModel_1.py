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



class DSSMModel_1(tf.keras.Model):
    def __init__(self, user_feature_dims, item_feature_dims, embedding_dim=8):
        super(DSSMModel_1, self).__init__()

        # 为每个用户特征创建一个 Embedding 层
        # input_dim 是词表大小 + 1（预留 0 给 padding），output_dim 是嵌入维度
        self.user_embeddings = {
            feat: Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, name=f"user_emb_{feat}")  for feat, vocab_size in user_feature_dims.items()
        }

        # 为每个物品特征创建一个 Embedding 层
        self.item_embeddings = {
            feat: Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, name=f"item_emb_{feat}")  for feat, vocab_size in item_feature_dims.items()
        }

        # 用户向量的两层全连接网络
        self.user_dense1 = Dense(128, activation='relu')
        self.user_dense2 = Dense(64, activation='relu')

        # 物品向量的两层全连接网络
        self.item_dense1 = Dense(128, activation='relu')
        self.item_dense2 = Dense(64, activation='relu')

        # 最终的输出层，输出点击概率（0~1）
        self.output_layer = Dense(1, activation='sigmoid')



    def call(self, inputs, training=False):
        # 拆分输入为用户特征和物品特征部分
        user_inputs = inputs[:len(self.user_embeddings)]
        item_inputs = inputs[len(self.user_embeddings):]

        # 通过用户 Embedding 层并 Flatten 展平
        user_embeds = [Flatten()(self.user_embeddings[feat](inp))for feat, inp in zip(self.user_embeddings, user_inputs)]
        # 拼接所有用户 embedding 得到用户向量，并输入全连接网络
        user_vector = Concatenate()(user_embeds)
        # print(user_vector.shape)   # (2,3,8) -> (2, 24)
        user_vector = self.user_dense1(user_vector)
        user_vector = self.user_dense2(user_vector)   #  (2, 64)

        # 通过物品 Embedding 层并 Flatten 展平
        item_embeds = [Flatten()(self.item_embeddings[feat](inp)) for feat, inp in zip(self.item_embeddings, item_inputs)]
        # 拼接所有物品 embedding 得到物品向量，并输入全连接网络
        item_vector = Concatenate()(item_embeds)
        # print(item_vector.shape)   # (2,5,8) -> (2, 40)
        item_vector = self.item_dense1(item_vector)
        item_vector = self.item_dense2(item_vector)  #  (2, 64)

        # 使用余弦相似度计算用户向量和物品向量的相似度（推荐核心）
        # 每个用户只跟其对应的物品计算相似度
        cosine_sim = tf.reduce_sum(user_vector * item_vector, axis=-1, keepdims=True) / (tf.norm(user_vector, axis=-1, keepdims=True) * tf.norm(item_vector, axis=-1, keepdims=True) + K.epsilon())
        # print(cosine_sim)
        # tf.Tensor(
        #     [[0.14836383]
        #      [0.26066038]], shape=(2, 1), dtype=float32)

        # cosine_sim = -tf.keras.losses.cosine_similarity(user_vector, item_vector, axis=1)
        # cosine_sim = tf.expand_dims(cosine_sim, axis=1)  # 保持 shape 和原来一致 (batch_size, 1)
        # print(cosine_sim)
        # tf.Tensor(
        #     [[0.35295278]
        #      [0.2738061]], shape=(2, 1), dtype=float32)


        # 模型在每一步会计算一个batch内该用户与所有物品之间的相似度，并不是一一配对的情况，但是这里面我们不用这种方式
        # cosine_sim = tf.matmul(user_vector, item_vector, transpose_b=True) / (tf.norm(user_vector, axis=-1, keepdims=True) * tf.norm(item_vector, axis=-1, keepdims=True) + K.epsilon())

        # 将相似度作为输入传入输出层，输出为点击概率
        output = self.output_layer(cosine_sim)
        return output





if __name__ == '__main__':
    # 1. 构建用户与物品的特征维度信息（即每个类别特征的最大ID + 1，用于Embedding层）
    user_feature_dims = {
        'uid': 10000,     # 用户ID，共有10000种取值
        'user_city': 50,  # 用户所在城市，共50个城市
        'device': 10      # 用户设备类型，共10类设备
    }

    item_feature_dims = {
        'item_id': 20000,   # 物品ID，共20000种物品
        'author_id': 5000,  # 作者ID，共5000个作者
        'item_city': 50,    # 物品所在城市，共50个城市
        'channel': 20,      # 内容频道，共20个频道
        'music_id': 30000   # 配乐ID，共30000种音乐
    }

    # 2. 构建用户输入：每个特征是一个 shape 为 (2, 1) 的张量，表示两个样本的取值
    user_inputs = [
        tf.constant([[1], [2]]),  # uid：第一个用户ID为1，第二个为2
        tf.constant([[3], [4]]),  # user_city：分别来自城市3和4
        tf.constant([[2], [1]]),  # device：设备类型分别为2和1
    ]

    # print(user_inputs)
    # [ < tf.Tensor: shape = (2, 1), dtype = int32, numpy =
    # array([[1],
    #        [2]]) >, < tf.Tensor: shape = (2, 1), dtype = int32, numpy =
    # array([[3],
    #        [4]]) >, < tf.Tensor: shape = (2, 1), dtype = int32, numpy =
    # array([[2],
    #        [1]]) >]

    # 转换成DataFrame
    # feature_names = ['uid', 'user_city', 'device']
    # user_df = pd.DataFrame(
    #     {name: tensor.numpy().flatten() for name, tensor in zip(feature_names, user_inputs)}
    # )
    # print(user_df)
    #    uid  user_city  device
    # 0    1          3       2
    # 1    2          4       1


    # 3. 构建物品输入：每个特征是一个 shape 为 (2, 1) 的张量，表示两个样本的取值
    item_inputs = [
        tf.constant([[100], [200]]),    # item_id：两个样本的物品ID分别为100和200
        tf.constant([[300], [400]]),    # author_id：分别为作者300和400
        tf.constant([[5], [6]]),        # item_city：分别来自城市5和6
        tf.constant([[1], [2]]),        # channel：分别属于频道1和2
        tf.constant([[1000], [2000]]),  # music_id：分别配乐1000和2000
    ]
    # print(item_inputs)
    # [ < tf.Tensor: shape = (2, 1), dtype = int32, numpy =
    # array([[100],
    #        [200]]) >, < tf.Tensor: shape = (2, 1), dtype = int32, numpy =
    # array([[300],
    #        [400]]) >, < tf.Tensor: shape = (2, 1), dtype = int32, numpy =
    # array([[5],
    #        [6]]) >, < tf.Tensor: shape = (2, 1), dtype = int32, numpy =
    # array([[1],
    #        [2]]) >, < tf.Tensor: shape = (2, 1), dtype = int32, numpy =
    # array([[1000],
    #        [2000]]) >]

    # 转换成DataFrame
    # item_feature_names = ['item_id', 'author_id', 'item_city', 'channel', 'music_id']
    # # 转换为 DataFrame
    # item_df = pd.DataFrame({name: tensor.numpy().flatten() for name, tensor in zip(item_feature_names, item_inputs)})
    # print(item_df)
    #    item_id  author_id  item_city  channel  music_id
    # 0      100        300          5        1      1000
    # 1      200        400          6        2      2000

    # 4. 创建 DSSM 模型实例，传入用户和物品特征维度信息
    model = DSSMModel_1(user_feature_dims, item_feature_dims)

    # 5. 调用模型进行推理（前向计算），注意输入需要拼接成一个整体列表传入
    outputs = model(user_inputs + item_inputs, training=False)

    # 6. 打印模型输出的形状和实际预测值（通常是一个 [batch_size, 1] 的概率值）
    print("Model output shape:", outputs.shape)
    print("Model predictions:", outputs.numpy())
