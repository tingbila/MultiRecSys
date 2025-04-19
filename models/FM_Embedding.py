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


class FM_MTL(Model):
    def __init__(self, feat_columns, emb_size):
        super().__init__()
        # feat_columns = [
        #     [{'feat': 'I1'}, {'feat': 'I2'}],
        #     [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
        # ]
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        
        self.emb_size = emb_size
        
        # Linear
        self.linear_dense = layers.Dense(1)
        self.linear_sparse_embeds = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=1)         for feat in self.sparse_feats
        ]

        # FM embedding
        self.fm_sparse_embeds = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size)  for feat in self.sparse_feats
        ]


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

        # 第一部分：线性部分
        linear_dense_out  = self.linear_dense(dense_inputs)
        linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.linear_sparse_embeds)],axis=1)
        # print("--------------1-----------")
        # print(linear_sparse_out)
        # [[ 0.01539519 -0.04832922 -0.02953873]
        #  [ 0.02253106  0.00941402  0.04219601]]

        linear_sparse_out = tf.reduce_sum(linear_sparse_out, axis=1, keepdims=True)
        first_order_output = linear_dense_out + linear_sparse_out
        # print(first_order_output)
        # [[0.22632796]
        #  [0.6256093 ]]

        # 第二部分：FM部分
        sparse_embeds = tf.stack([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.fm_sparse_embeds)], axis=1)
        # print(sparse_embeds) shape: (batch_size=2, field_num=3, embedding_dim=5)
        # Tensor("stack:0", shape=(2, 3, 5), dtype=float32)
        # tf.Tensor(
        # [[[-0.0292243   0.03134212 -0.00664638  0.0308771   0.03662998]
        #   [-0.02252715  0.00618609 -0.0408314   0.0155008   0.00702292]
        #   [ 0.02527222  0.02442351  0.01027572  0.04815536  0.01610643]]

        #  [[ 0.01005882 -0.03315624 -0.0195043   0.01774564  0.03821408]
        #   [-0.03824542  0.00229248  0.00047214  0.0488669  -0.04776417]
        #   [-0.01696395 -0.00136379  0.04921383  0.04019973 -0.00026955]]], shape=(2, 3, 5), dtype=float32)

        fm_input = sparse_embeds
        # 这一步对每个样本的所有 sparse embedding 向量在特征维度上求和：对于某个样本，就是把它的三个特征的 embedding 向量相加，变成一个总的表示
        # (2, 3, 5) → [2, 5]
        summed = tf.reduce_sum(fm_input, axis=1)
        # (2, 5) → [2, 5]
        squared_sum = tf.square(summed)
        # 先对每个 embedding 向量做逐元素平方，然后再对所有 sparse 特征做求和 (2, 3, 5) → [2, 5]
        squared = tf.reduce_sum(tf.square(fm_input), axis=1)
        # [2, 1]，表示每个样本的二阶交叉值（标量）
        second_order = 0.5 * tf.reduce_sum(squared_sum - squared, axis=1, keepdims=True)
        # print(second_order)
        # [[0.00537243]
        #  [0.00075581]]

        # 合成最终输出
        logits = first_order_output + second_order

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

    model = FM_MTL(feat_columns=feat_columns, emb_size=5)
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
# [[0.29898568 0.37788569]
#  [0.30009296 0.51782235]
#  [0.06622059 0.74233538]]
# Sparse 输入:
# [[1 2 3]
#  [4 5 5]
#  [1 2 3]]
#
# 模型输出:
# {'finish': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.5775809 ],
#        [0.6161572 ],
#        [0.75753856]], dtype=float32)>, 'like': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.46416104],
#        [0.4459036 ],
#        [0.37216955]], dtype=float32)>}