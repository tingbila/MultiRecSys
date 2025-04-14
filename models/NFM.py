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
from config.data_config import batch_size


# NFM 的关键创新点就是：把 FM 的交叉从“内积”变成“逐元素乘法 + DNN”，保留更多交叉信息的细节。
# NFM 把所有 embedding 做两两维度上的乘法，保留原始维度的交叉信息，然后让 DNN 自主学习每一维
# 的组合权重。这就是它比 FM 更 表达能力强 的原因！
# e1 = [1.0, 2.0]   # 特征1的embedding
# e2 = [3.0, 4.0]   # 特征2的embedding
# e3 = [5.0, 6.0]   # 特征3的embedding
# Pair	      逐元素乘法结果
# e1 * e2	  [1×3, 2×4] = [3, 8]
# e1 * e3	  [1×5, 2×6] = [5, 12]
# e2 * e3	  [3×5, 4×6] = [15, 24]
# flatten   = [3, 8, 5, 12, 15, 24]  然后flatten作为DNN的输入
class NFM(Model):
    def __init__(self, feat_columns, emb_size, batch_size = batch_size):
        super().__init__()
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.dense_size  = len(self.dense_feats)
        self.sparse_size = len(self.sparse_feats)

        # 这行代码的作用是计算在 NFM 模型 中，给定 sparse_size 个特征（即有多少个离散字段参与 embedding），两两组合的交叉数量（即有多少种不同的二阶特征交互对）
        # num_pairs=C(n,2)
        self.num_pairs  = self.sparse_size * (self.sparse_size - 1) // 2  # pair-wise 交叉数量

        self.emb_size = emb_size
        self.batch    = batch_size

        self.linear_dense = layers.Dense(1)   # 参数数量不是1哦

        self.first_order_sparse_emb = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=1)
            for feat in self.sparse_feats
        ]

        self.second_order_sparse_emb = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=self.emb_size)
            for feat in self.sparse_feats
        ]

        self.dnn = tf.keras.Sequential([
            tf.keras.Input(shape=(self.num_pairs * self.emb_size,)),  # 添加这行，其实我感觉不加也行，但是在fit的时候会报错
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])

        # 每个任务一个输出层
        self.finish_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output_layer   = tf.keras.layers.Dense(1, activation='sigmoid', name='like')


    def call(self, inputs, training=False):
        sparse_inputs, dense_inputs = inputs

        # 第一部分：线性部分
        linear_dense_out = self.linear_dense(dense_inputs)
        linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.first_order_sparse_emb)], axis=1)
        linear_sparse_out = tf.reduce_sum(linear_sparse_out, axis=1, keepdims=True)
        first_order_output = linear_dense_out + linear_sparse_out


        # 第二部分：NFM-DNN部分
        """
        embeddings: Tensor, shape=[batch_size, num_fields, embedding_dim]
        """
        embeddings = tf.stack([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.second_order_sparse_emb)], axis=1) # shape=(3, 3, 5)

        num_fields = len(self.sparse_feats)
        interaction_list = []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                # 向量两两交叉（逐元素乘法）注意*的结果不是标量,而是向量； 向量e1 * e2  ===>	[1×3, 2×4] = [3, 8]
                interaction = embeddings[:, i, :] * embeddings[:, j, :]   # shape=[batch_size, embedding_dim]
                interaction_list.append(interaction)
        # 堆叠所有 pair-wise interaction 向量
        bi_interaction    = tf.stack(interaction_list, axis=1)                                   # [batch_size, num_pairs, embedding_dim]
        nfm_flatten_input = tf.reshape(bi_interaction, shape=(tf.shape(bi_interaction)[0], -1))  # [batch_size, num_pairs * embedding_dim]

        dnn_output = self.dnn(nfm_flatten_input, training=training)

        # 合成最终输出
        logits = first_order_output  + dnn_output

        # 分支输出
        finish_output = self.finish_output_layer(logits)
        like_output   = self.like_output_layer(logits)

        return {'finish': finish_output, 'like': like_output}


if __name__ == '__main__':
    # 假设有 2 个 dense 特征，3 个 sparse 特征
    dense_feats  = ['I1', 'I2']
    sparse_feats = ['C1', 'C2', 'C3']

    # 每个 sparse 特征的唯一值个数分别为 10, 8, 6
    feat_columns = [
        [{'feat': 'I1'}, {'feat': 'I2'}],
        [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
    ]

    # 模拟 batch size 为 3 的输入
    batch_size = 3
    dense_input = tf.random.uniform(shape=(batch_size, len(dense_feats)), dtype=tf.float32)
    sparse_input = tf.random.uniform(shape=(batch_size, len(sparse_feats)), maxval=6, dtype=tf.int32)

    # 初始化模型，CIN层设置为2层，每层输出维度为16，当然7,15这种也行
    model = NFM(feat_columns=feat_columns, emb_size=5,batch_size=batch_size)

    # 前向传播
    output = model((sparse_input, dense_input), training=False)

    # 打印结果
    print("Dense 输入:")
    print(dense_input.numpy())
    print("Sparse 输入:")
    print(sparse_input.numpy())
    print("\n模型输出:")
    print(output)

    model.summary()

# Dense 输入:
# [[0.84716856 0.18821383]
#  [0.76769197 0.06524563]
#  [0.09913898 0.591589  ]]
# Sparse 输入:
# [[2 5 3]
#  [4 5 4]
#  [0 1 0]]
#
# 模型输出:
# {'finish': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.5712998 ],
#        [0.5808364 ],
#        [0.40740198]], dtype=float32)>, 'like': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.4043136 ],
#        [0.39168742],
#        [0.6237949 ]], dtype=float32)>}
# Model: "deep_fm_x_deep_fm_mtl"
# ┌─────────────────────────────────┬────────────────────────┬───────────────┐
# │ Layer (type)                    │ Output Shape           │       Param # │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense (Dense)                   │ (3, 1)                 │             3 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding (Embedding)           │ (3, 1)                 │            10 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_1 (Embedding)         │ (3, 1)                 │             8 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_2 (Embedding)         │ (3, 1)                 │             6 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_3 (Embedding)         │ (3, 5)                 │            50 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_4 (Embedding)         │ (3, 5)                 │            40 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_5 (Embedding)         │ (3, 5)                 │            30 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ sequential (Sequential)         │ (3, 1)                 │        84,201 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ functional (Functional)         │ (None, 1)              │           423 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ finish (Dense)                  │ (3, 1)                 │             2 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ like (Dense)                    │ (3, 1)                 │             2 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 84,775 (331.15 KB)
#  Trainable params: 84,775 (331.15 KB)
#  Non-trainable params: 0 (0.00 B)

