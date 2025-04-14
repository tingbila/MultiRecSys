# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv1D, Dense, Lambda, Multiply, Reshape, Permute, Concatenate
from models.Cin_Keras import CIN
from models.Attention import Attention


# 定义XDeepFM + Transform当中的Attention
class XDeepFM_Transform_MTL(Model):
    def __init__(self, feat_columns, emb_size, cin_layers):
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

        # Attention 模块
        self.attention = Attention(num_heads=2, key_dim=5)

        self.dnn = tf.keras.Sequential([
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])

        # CIN 模块
        # input_dim是输入特征的数量
        self.cin = CIN(input_dim=len(self.sparse_feats), embedding_dim=emb_size, layer_dims=cin_layers)

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

        # 第二部分：FM部分
        embeddings = tf.stack([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.second_order_sparse_emb)], axis=1)
        summed = tf.reduce_sum(embeddings, axis=1)
        squared_sum = tf.square(summed)
        squared = tf.reduce_sum(tf.square(embeddings), axis=1)
        second_order = 0.5 * tf.reduce_sum(squared_sum - squared, axis=1, keepdims=True)

        # 第三部分：CIN部分
        # CIN部分 输入是离散变量embeddings之后的结果(2,3,5)
        cin_input = tf.reshape(embeddings, shape=(-1, len(self.sparse_feats), self.emb_size))
        # print(cin_input)
        # print(cin_input.shape)
        # [[[0.04140563  0.03176231  0.01356237 - 0.04648466  0.04604844]
        #   [-0.04380695  0.0055673 - 0.04512671 - 0.00325407  0.04986053]
        #  [-0.04896227 - 0.02058487  0.04993394  0.02401217 - 0.01342272]]
        #
        # [[0.04140563  0.03176231  0.01356237 - 0.04648466  0.04604844]
        #  [-0.03652285  0.00218137  0.01824282 - 0.03070251  0.02715142]
        # [-0.0392504 - 0.04152218
        # 0.00013808
        # 0.04893823 - 0.00362258]]
        #
        # [[0.00019486  0.02938681 - 0.0446441   0.0115645   0.03913646]
        #  [-0.0091401 - 0.04220575 - 0.00148644 - 0.03097728 - 0.04496309]
        #  [0.01833469  0.01332567 - 0.04477177  0.03069586 - 0.00196253]]], shape = (3, 3, 5), dtype = float32)
        # (3, 3, 5)
        cin_output = self.cin(cin_input)

        # 第四部分：Transform_Attention部分
        # Attention特征提取部分
        attention_out = self.attention(embeddings)  # shape=(3, 3, 5)
        pooled_attention = tf.keras.layers.GlobalAveragePooling1D()(attention_out)  # shape=(3, 5)
        # print("----------attention_out----------")
        # print(attention_out)
        # print("----------pooled_attention----------")
        # print(pooled_attention)

        # 第五部分：DNN 部分
        # DNN 部分
        flatten_embeddings = tf.reshape(embeddings, shape=(-1, len(self.sparse_feats) * self.emb_size))  # （3,15）
        dnn_input = tf.concat([dense_inputs, flatten_embeddings, pooled_attention], axis=1)
        dnn_output = self.dnn(dnn_input, training=training)

        # 合成最终输出
        logits = first_order_output + second_order + dnn_output + cin_output

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

    # 初始化模型，CIN层设置为2层，每层输出维度为16，当然7,15这种也行
    model = XDeepFM_Transform_MTL(feat_columns=feat_columns, emb_size=5, cin_layers=[7,15])

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

    model.summary()



# ----------attention_out----------
# Tensor("attention_1/multi_head_attention_1/attention_output_1/add:0", shape=(3, 3, 5), dtype=float32)
# ----------pooled_attention----------
# Tensor("global_average_pooling1d_1/Mean:0", shape=(3, 5), dtype=float32)
# ----------attention_out----------
# tf.Tensor(
# [[[ 0.0073271  -0.01063516 -0.00380943 -0.00636701  0.00220868]
#   [ 0.00733004 -0.01063512 -0.0038076  -0.00636793  0.00221005]
#   [ 0.00732735 -0.01063572 -0.0038063  -0.00636909  0.0022065 ]]
#
#  [[ 0.01502962  0.00565303  0.0088194  -0.00126621  0.01489555]
#   [ 0.0150284   0.00564757  0.00882035 -0.00127211  0.01489167]
#   [ 0.01502859  0.00565184  0.00881962 -0.00126768  0.01489415]]
#
#  [[ 0.00253227 -0.00208628 -0.00672294  0.00228702  0.00014438]
#   [ 0.00252936 -0.00208656 -0.00672245  0.0022867   0.00013971]
#   [ 0.00252939 -0.00208626 -0.00672186  0.00228659  0.0001389 ]]], shape=(3, 3, 5), dtype=float32)
# ----------pooled_attention----------
# tf.Tensor(
# [[ 0.00732816 -0.01063533 -0.00380777 -0.00636801  0.00220841]
#  [ 0.01502887  0.00565081  0.00881979 -0.00126866  0.01489379]
#  [ 0.00253034 -0.00208637 -0.00672241  0.00228677  0.000141  ]], shape=(3, 5), dtype=float32)
# Dense 输入:
# [[0.0710113  0.7410916 ]
#  [0.3367871  0.0239917 ]
#  [0.9464096  0.84482574]]
# Sparse 输入:
# [[2 1 0]
#  [0 4 5]
#  [2 0 0]]
#
# 模型输出:
# {'finish': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.28823373],
#        [0.45229355],
#        [0.17946997]], dtype=float32)>, 'like': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.2887841 ],
#        [0.45243418],
#        [0.1801348 ]], dtype=float32)>}
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
# │ attention (Attention)           │ ?                      │           235 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ sequential (Sequential)         │ (3, 1)                 │        85,201 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ functional (Functional)         │ (None, 1)              │           423 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ finish (Dense)                  │ (3, 1)                 │             2 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ like (Dense)                    │ (3, 1)                 │             2 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 86,010 (335.98 KB)
#  Trainable params: 86,010 (335.98 KB)
#  Non-trainable params: 0 (0.00 B)