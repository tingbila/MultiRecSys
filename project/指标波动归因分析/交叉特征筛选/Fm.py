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
import torch


class Fm (Model):
    def __init__(self, feat_columns, emb_size=5):
        super().__init__()
        # feat_columns = [
        #     [{'feat': 'I1'}, {'feat': 'I2'}],
        #     [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
        # ]
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.emb_size = emb_size

        self.linear_dense = layers.Dense(1)

        self.V = self.add_weight(
            shape=(len(self.dense_feats) + len(self.sparse_feats), self.emb_size),
            initializer="random_normal",
            trainable=True,
            name="fm_embedding_weights"  # 👈 加上这一行
        )

        # print(self.V)
        # print(self.V.shape)  # (234, 5)

        # 每个任务一个输出层
        # self.finish_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='finish')
        self.finish_output_layer = tf.keras.layers.Dense(1, activation=None, name='finish')
        # self.like_output_layer   = tf.keras.layers.Dense(1, activation='sigmoid', name='like')



    def call(self, inputs, training=False):
        sparse_inputs, dense_inputs = inputs
        # Dense 输入:
        # [[0.211972   0.3256514 ]
        #  [0.58325326 0.5058359 ]]
        # Sparse 输入:
        # [[5 4 1]
        #  [3 2 3]]

        # 拼接离散特征和连续特征
        X = tf.concat([tf.cast(sparse_inputs, tf.float32), dense_inputs], axis=1)
        # print(X)
        """
        tf.Tensor(
        [[2.         2.         2.         0.3107803  0.3713479 ]
         [1.         1.         0.         0.0973109  0.9294832 ]
         [0.         2.         3.         0.28172433 0.3227619 ]], shape=(3, 5), dtype=float32)
        """

        # 第一部分：线性部分(离散变量和连续都要走线性模型)
        linear_out = self.linear_dense(X)

        # 第二部分：FM交互项部分（下面的这是效率低的写法）
        # fm_out = 0
        # n = X.shape[1]  # 统计有几列
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         # print(i,j)
        #         fm_out += tf.tensordot(self.V[i], self.V[j], axes=1) * X[:, i] * X[:, j]


        # 第二部分：FM交互项部分（下面的这是公式优化写法）
        # a*b = [(a+b)^2 - (a^2+b^2)]/2
        xv_square         = tf.square(tf.matmul(X, self.V))
        x_square_v_square = tf.matmul(tf.square(X), tf.square(self.V))
        fm_out = 0.5 * tf.reduce_sum(xv_square - x_square_v_square, axis=1, keepdims=True)


        logits = linear_out + fm_out

        # 分支输出
        finish_output = self.finish_output_layer(logits)

        return {'finish': finish_output}



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
    model = Fm(feat_columns=feat_columns, emb_size=3)

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
