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


class DeepCrossing_Residual(Model):
    def __init__(self, feat_columns, emb_size, hidden_units=[128, 64, 32]):
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


        self.residual_blocks = []
        for units in hidden_units:
            self.residual_blocks.append(
                tf.keras.Sequential([
                    layers.Dense(units, activation='relu'),
                    layers.Dense(self.dense_size + len(self.sparse_feats) * self.emb_size)    # 输出维度与输入一致，便于残差连接
                ])
            )
        self.residual_output_layer = tf.keras.layers.Dense(1)

        # 每个任务一个输出层
        self.finish_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='like')

    def call(self, inputs, training=False):
        """
        :param inputs:
        :param training: Keras在 fit() 时并没有显式指定 training=True，因为它自动管理了这个标志位！
        :return:
        """
        sparse_inputs, dense_inputs = inputs
        # Dense 输入:
        # [[0.211972   0.3256514 ]
        #  [0.58325326 0.5058359 ]]
        # Sparse 输入:
        # [[5 4 1]
        #  [3 2 3]]

        # 第一部分：线性部分（注意：论文当中的就是一个纯深度模型，是没有线性这部分的）
        linear_dense_out = self.linear_dense(dense_inputs)
        linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.first_order_sparse_emb)],axis=1)
        # print("--------------1-----------")
        # print(linear_sparse_out)
        # [[ 0.01539519 -0.04832922 -0.02953873]
        #  [ 0.02253106  0.00941402  0.04219601]]

        linear_sparse_out = tf.reduce_sum(linear_sparse_out, axis=1, keepdims=True)
        first_order_output = linear_dense_out + linear_sparse_out
        # print(first_order_output)
        # [[0.22632796]
        #  [0.6256093 ]]

        # 第二部分：DNN 部分（residual部分）
        embeddings = tf.stack([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.second_order_sparse_emb)], axis=1)
        # print(embeddings)
        # Tensor("stack:0", shape=(2, 3, 5), dtype=float32)
        # tf.Tensor(
        # [[[-0.0292243   0.03134212 -0.00664638  0.0308771   0.03662998]
        #   [-0.02252715  0.00618609 -0.0408314   0.0155008   0.00702292]
        #   [ 0.02527222  0.02442351  0.01027572  0.04815536  0.01610643]]

        #  [[ 0.01005882 -0.03315624 -0.0195043   0.01774564  0.03821408]
        #   [-0.03824542  0.00229248  0.00047214  0.0488669  -0.04776417]
        #   [-0.01696395 -0.00136379  0.04921383  0.04019973 -0.00026955]]], shape=(2, 3, 5), dtype=float32)


        flatten_embeddings = tf.reshape(embeddings, shape=(-1, len(self.sparse_feats) * self.emb_size))
        # print(flatten_embeddings)
        # Tensor("Reshape:0", shape=(2, 15), dtype=float32)
        # tf.Tensor(
        # [[-0.03405142  0.01116457 -0.00488006  0.03367222 -0.04788997  0.0262876 0.04647902 -0.01162871  0.03328068  0.04433748  0.02085209  0.01660527
        #   -0.02046416  0.00683039  0.04853446]
        #  [-0.03806484  0.0479795   0.0132894  -0.03121579 -0.0166074   0.00733398
        #    0.00708617 -0.00899755  0.02732437 -0.00605234 -0.02896208  0.02931662
        #    0.0044607   0.03854013  0.04758653]], shape=(2, 15), dtype=float32)
        dnn_input = tf.concat([dense_inputs, flatten_embeddings], axis=1)
        x = dnn_input
        for block in self.residual_blocks:
            x = x + block(x, training=training)  # 残差连接
        dnn_output = self.residual_output_layer(x)

        logits = first_order_output + dnn_output

        # 分支输出
        finish_output = self.finish_output_layer(logits)
        like_output = self.like_output_layer(logits)

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
    model = DeepCrossing_Residual(feat_columns=feat_columns, emb_size=5, hidden_units=[128, 64, 32])

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
    print(model.summary())

    # 用法举例:
    # model = DeepCrossing(input_dim=128)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    # model.fit(train_data, train_labels, batch_size=32, epochs=10)

    # Dense
    # 输入:
    # [[0.67293584 0.2541728]
    #  [0.96817994 0.2788309]
    # [0.8373424
    # 0.55251396]]
    # Sparse
    # 输入:
    # [[1 3 0]
    #  [0 0 0]
    # [2
    # 0
    # 3]]
    #
    # 模型输出:
    # {'finish': < tf.Tensor: shape = (3, 1), dtype = float32, numpy =
    # array([[0.35982525],
    #        [0.34824157],
    #        [0.26548928]], dtype=float32) >, 'like': < tf.Tensor: shape = (3, 1), dtype = float32, numpy =
    # array([[0.58888644],
    #        [0.59651387],
    #        [0.65356797]], dtype=float32) >}
# Model: "deep_crossing__residual"
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
# │ sequential (Sequential)         │ (3, 17)                │         4,497 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ sequential_1 (Sequential)       │ (3, 17)                │         2,257 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ sequential_2 (Sequential)       │ (3, 17)                │         1,137 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense_7 (Dense)                 │ (3, 1)                 │            18 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ finish (Dense)                  │ (3, 1)                 │             2 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ like (Dense)                    │ (3, 1)                 │             2 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 8,060 (31.48 KB)
#  Trainable params: 8,060 (31.48 KB)
#  Non-trainable params: 0 (0.00 B)
# None
