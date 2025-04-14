# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


# models/DeepFm.py
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
from config.data_config import *


class AFM_Embedding(Model):
    def __init__(self, feat_columns, emb_size):
        super().__init__()
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.dense_size = len(self.dense_feats)
        self.emb_size = emb_size

        self.linear_dense = layers.Dense(1)

        self.first_order_sparse_emb = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=1)
            for feat in self.sparse_feats
        ]

        self.second_order_sparse_emb = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size)
            for feat in self.sparse_feats
        ]

        # Attention网络层-权重参数
        self.attention_dense_1 = layers.Dense(units=64, activation='relu')
        self.attention_dense_2 = layers.Dense(units=1, activation=None)

        self.finish_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output_layer   = tf.keras.layers.Dense(1, activation='sigmoid', name='like')

    def call(self, inputs, training=False):
        sparse_inputs, dense_inputs = inputs

        # 第一部分：线性部分
        linear_dense_out = self.linear_dense(dense_inputs)
        linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.first_order_sparse_emb)], axis=1)
        linear_sparse_out = tf.reduce_sum(linear_sparse_out, axis=1, keepdims=True)
        first_order_output = linear_dense_out + linear_sparse_out

        # 第二部分：Attentional FM 交互项 - 只对稀疏特征部分进行了处理
        embeddings = tf.stack([
            emb(sparse_inputs[:, i]) for i, emb in enumerate(self.second_order_sparse_emb)
        ], axis=1)  # shape: [batch, field_num, emb_size]

        field_num = embeddings.shape[1]
        element_wise_products = []
        for i in range(field_num):
            for j in range(i + 1, field_num):
                element_wise_products.append(embeddings[:, i] * embeddings[:, j])
        element_wise_products = tf.stack(element_wise_products, axis=1)  # [batch, num_pairs, emb_size]

        # attention scores - 不同样本 - 不同特征之间权重值都是不一样的
        attention_temp = self.attention_dense_1(element_wise_products)
        attention_scores = self.attention_dense_2(attention_temp)    # [batch, num_pairs, 1]
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # [batch, num_pairs, 1]

        # 加权求和交叉向量
        attention_output = tf.reduce_sum(attention_weights * element_wise_products, axis=1)  # [batch, emb_size]
        afm_out = tf.reduce_sum(attention_output, axis=1, keepdims=True)  # [batch, 1]

        logits = first_order_output + afm_out

        finish_output = self.finish_output_layer(logits)
        like_output   = self.like_output_layer(logits)

        return {'finish': finish_output, 'like': like_output}




if __name__ == '__main__':
    # 假设有 2 个 dense 特征，3 个 sparse 特征
    dense_feats = ['I1', 'I2']
    sparse_feats = ['C1', 'C2', 'C3']

    # 每个 sparse 特征的唯一值个数分别为 10, 8, 6
    feat_columns = [
        [{'feat': 'I1'}, {'feat': 'I2'}],
        [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
    ]

    # 初始化模型
    model = AFM_Embedding(feat_columns=feat_columns, emb_size=5)

    # 模拟 batch size 为 3 的输入
    batch_size = 3
    dense_input = tf.random.uniform(shape=(batch_size, len(dense_feats)), dtype=tf.float32)
    sparse_input = tf.random.uniform(shape=(batch_size, len(sparse_feats)), maxval=6, dtype=tf.int32)

    # 前向传播
    output = model((sparse_input, dense_input), training=False)

    # 打印结果
    print("Dense 输入:")
    print(dense_input.numpy())
    print("Sparse 输入:")
    print(sparse_input.numpy())
    print("\n模型输出:")
    print(output)


    # Dense 输入:
    # [[0.19882536 0.9919691 ]
    #  [0.14089882 0.6178216 ]
    #  [0.59311116 0.79255974]]
    # Sparse 输入:
    # [[1 3 3]
    #  [4 4 0]
    #  [1 3 2]]
    #
    # 模型输出:
    # (<tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    # array([[0.47370112],
    #        [0.4806958 ],
    #        [0.4883328 ]], dtype=float32)>, <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    # array([[0.43123466],
    #        [0.4493978 ],
    #        [0.4693598 ]], dtype=float32)>)
