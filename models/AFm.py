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


class AFm (Model):
    def __init__(self, feat_columns, emb_size=5):
        super().__init__()
        # feat_columns = [
        #     [{'feat': 'I1'}, {'feat': 'I2'}],
        #     [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
        # ]
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]

        self.emb_size = emb_size

        # FM当中离散和连续字段是用一个V矩阵
        self.V = self.add_weight(
            shape=(len(self.dense_feats) + len(self.sparse_feats), self.emb_size),
            initializer="random_normal",
            trainable=True
        )

        # Linear
        self.linear_dense = layers.Dense(1)

        # Attention层参数（权重参数）
        self.attention_dense = layers.Dense(units=64, activation='relu',name='attention_dense')
        self.attention_score = layers.Dense(units=1, activation=None,name='attention_score')

        # 每个任务一个输出层
        self.finish_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output_layer   = tf.keras.layers.Dense(1, activation='sigmoid', name='like')



    def call(self, inputs, training=False):
        sparse_inputs = inputs[0]   # shape: (batch_size, num_sparse)
        dense_inputs  = inputs[1]   # shape: (batch_size, num_dense)
        # Dense 输入:
        # [[0.211972   0.3256514 ]
        #  [0.58325326 0.5058359 ]]
        # Sparse 输入:
        # [[5 4 1]
        #  [3 2 3]]

        # 拼接离散特征和连续特征
        X = tf.concat([tf.cast(sparse_inputs, tf.float32), dense_inputs], axis=1)
        """
        tf.Tensor(
        [[2.         2.         2.         0.3107803  0.3713479 ]
         [1.         1.         0.         0.0973109  0.9294832 ]
         [0.         2.         3.         0.28172433 0.3227619 ]], shape=(3, 5), dtype=float32)
        """

        # 第一部分：线性部分(离散变量和连续都要走线性模型)
        linear_out = self.linear_dense(X)


        # 第二部分：AFM交互项部分（下面的这是公式优化写法）
        # 计算embedding后的交叉向量对  shape=(3, 5, 3)
        embeddings = tf.expand_dims(X, axis=2) * tf.expand_dims(self.V, axis=0)  # shape: [batch, num_sparse + num_dense, emb_size]

        pairwise_interactions = []
        num_feat = embeddings.shape[1]
        for i in range(num_feat):
            for j in range(i + 1, num_feat):
                pairwise_interactions.append(embeddings[:, i, :] * embeddings[:, j, :])  # element-wise product
        interactions = tf.stack(pairwise_interactions, axis=1)       # [batch, num_pairs, emb_size]   (3, 10, 3)

        # Attention部分-计算每个pair交互的权重
        attention_temp = self.attention_dense(interactions)          # [batch, num_pairs, 64]
        attention_scores = self.attention_score(attention_temp)      # [batch, num_pairs, 1]
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # [batch, num_pairs, 1]
        # print(attention_weights)
        # print(attention_weights.shape)
        # tf.Tensor(
        #     [[[0.0998857]
        #       [0.10056859]
        #       [0.09996547]
        #       [0.09998109]
        #       [0.09959196]
        #       [0.10000238]
        #       [0.09999505]
        #       [0.10000414]
        #       [0.10000595]
        #       [0.09999973]]
        #
        #      [[0.09989122]
        #         [0.10011978]
        #         [0.09933807]
        #         [0.09995251]
        #         [0.10011978]
        #         [0.10029529]
        #         [0.09998476]
        #         [0.10011978]
        #         [0.10011978]
        #         [0.1000591]]
        #
        #      [[0.10003331]
        #         [0.10003331]
        #         [0.09979664]
        #         [0.0999464]
        #         [0.10003331]
        #         [0.10003331]
        #         [0.10003331]
        #         [0.10003331]
        #         [0.10003331]
        #         [0.10002378]]], shape=(3, 10, 1), dtype=float32)

        # 基于权重计算最终的结果 交互的结果重新更新
        # [batch, num_pairs, emb_size] *  [batch, num_pairs, 1] = [batch, num_pairs, emb_size] ==> [batch, emb_size]
        attention_output = tf.reduce_sum(attention_weights * interactions, axis=1)   # [batch, emb_size]
        afm_out = tf.reduce_sum(attention_output, axis=1, keepdims=True)             # [batch, 1]

        logits = linear_out + afm_out

        # 分支输出
        finish_output = self.finish_output_layer(logits)
        like_output   = self.like_output_layer(logits)

        return {'finish': finish_output, 'like': like_output}



if __name__ == '__main__':
    # 1. 不使用序列特征
    dense_feats = ['I1', 'I2']
    sparse_feats = ['C1', 'C2', 'C3']
    feat_columns = [
        [{'feat': 'I1'}, {'feat': 'I2'}],
        [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
    ]

    model = AFm(feat_columns=feat_columns, emb_size=5)
    sparse_input = np.array([[1, 2, 3], [4, 5, 5], [1, 2, 3]])
    dense_input = np.random.random((3, len(dense_feats)))
    output = model((sparse_input, dense_input), training=False)

    print("Dense 输入:")
    print(dense_input)
    print("Sparse 输入:")
    print(sparse_input)
    print("\n模型输出:")
    print(output)
    model.summary()


# Dense 输入:
# [[0.09572926 0.87675378]
#  [0.60038836 0.92242923]
#  [0.22504737 0.09761649]]
# Sparse 输入:
# [[1 2 3]
#  [4 5 5]
#  [1 2 3]]
#
# 模型输出:
# {'finish': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.9464742 ],
#        [0.99231625],
#        [0.9178204 ]], dtype=float32)>, 'like': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.6779792 ],
#        [0.77899694],
#        [0.6514487 ]], dtype=float32)>}
# Model: "a_fm"
