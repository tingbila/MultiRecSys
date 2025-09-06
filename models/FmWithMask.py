# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media
"""
Mask 的作用:
mask 是 [num_features, num_features] 的 0-1 矩阵
用来控制哪些特征对的交互可以计算
支持：
上三角（i < j）
屏蔽特定组合（如业务不关注的交互）
"""


# models/DeepFm.py
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
from config.data_config import *
import torch


class FmWithMask (Model):
    def __init__(self, feat_columns, emb_size=5, mask_matrix=None):
        """
        :param feat_columns:
        :param emb_size:
        :param mask_matrix:新增一个 Mask 矩阵参数 ，表示允许的交互。
        M∈{0,1}n×n
        M∈{0,1}
        n×n
        """
        super().__init__()
        # feat_columns = [
        #     [{'feat': 'I1'}, {'feat': 'I2'}],
        #     [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
        # ]
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.emb_size = emb_size

        self.linear_dense = layers.Dense(1)

        self.V = self.add_weight(
            name="fm_embeddings",  # ⭐️ 加一个明确的名字
            shape=(len(self.dense_feats) + len(self.sparse_feats), self.emb_size),
            initializer="random_normal",
            trainable=True
        )

        print(self.V)
        print(self.V.shape)  # (5, 3)
        # < tf.Variable'Variable:0'
        # shape = (5, 3)
        # dtype = float32, numpy =
        # array([[0.02899534, -0.04050206, 0.03443773],
        #        [-0.01291281, -0.06589996, -0.0117657],
        #        [-0.0475354, -0.01391905, -0.00871712],
        #        [-0.0351649, 0.00506363, -0.00288101],
        #        [-0.11686293, 0.08292484, -0.0692029]], dtype=float32) >
        # (5, 3)

        # 将输入的 Mask 矩阵转换成 TensorFlow 常量张量，用于在模型的计算中控制特征交互，并且保证它不参与训练。
        # if mask_matrix is None:
        #     n_feats = len(self.dense_feats) + len(self.sparse_feats)
        #     mask_matrix = np.ones((n_feats, n_feats), dtype=np.float32)
        # tf.constant 创建的是 常量张量，它的值 固定不变，在训练过程中不会被优化器更新，也不会计算梯度。
        self.mask = tf.constant(mask_matrix, dtype=tf.float32)
        print("self.mask",self.mask)

        # 每个任务一个输出层
        self.finish_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output_layer   = tf.keras.layers.Dense(1, activation='sigmoid', name='like')



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
        # print('linear_out',linear_out)
        # tf.Tensor(
        #     [[-2.4728346]
        #      [-3.7007458]
        #      [-2.7860653]], shape=(3, 1), dtype=float32)

        # 第二部分：FM交互项部分（下面的这是效率低的写法）
        """
        tf.tensordot(self.V[i], self.V[j], axes=1) → 计算 embedding 内积 <v_i, v_j> → 标量
        X[:, i] * X[:, j] → 样本对应的特征乘积 → 向量
        fm_out += ... → 累加所有特征对的二阶交互 → 向量 -> [batch_size, 1]
        """
        fm_out = 0
        n = X.shape[1]  # 统计有几列
        for i in range(n):
            for j in range(i + 1, n):
                if self.mask[i, j] == 1:  # 只计算 mask=1 的交互
                    print(i,j)
                    fm_out += tf.tensordot(self.V[i], self.V[j], axes=1) * X[:, i] * X[:, j]
        print("fm_out",fm_out)


        # 第二部分：FM交互项部分（下面的这是公式优化写法）
        # a*b = [(a+b)^2 - (a^2+b^2)]/2
        # xv_square         = tf.square(tf.matmul(X, self.V))
        # x_square_v_square = tf.matmul(tf.square(X), tf.square(self.V))
        # fm_out = 0.5 * tf.reduce_sum(xv_square - x_square_v_square, axis=1, keepdims=True)
        # print('fm_out',fm_out)
        # tf.Tensor(
        #     [[0.04543651]
        #      [0.06157517]
        #      [0.05966767]], shape=(3, 1), dtype=float32)

        logits = linear_out + fm_out

        # 分支输出
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

    # FM 的交互项只需要 上三角 (i < j)，下三角 (j < i) 是重复的，可以直接置 0。
    """
    这是你传入的 Mask 矩阵，形状为 [n_features, n_features]。
    内容是 0 或 1：
    1 → 允许计算该特征对交互
    0 → 屏蔽该交互
    """
    n_feats = len(dense_feats) + len(sparse_feats)
    mask = np.ones((n_feats, n_feats), dtype=np.float32)
    np.fill_diagonal(mask, 0)  # 去掉对角线
    mask = np.triu(mask, 1)    # 只保留上三角（i < j），下三角全置 0

    # 屏蔽额外交互-这个部分需要业务方自己定义！！
    mask[0, 4] = 0
    mask[1, 3] = 0
    print(mask)
    # [[0. 1. 1. 1. 0.]
    #  [0. 0. 1. 0. 1.]
    #  [0. 0. 0. 1. 1.]
    #  [0. 0. 0. 0. 1.]
    #  [0. 0. 0. 0. 0.]]


    # 初始化模型
    model = FmWithMask(feat_columns=feat_columns, emb_size=3,mask_matrix=mask)

    # 模拟 batch size 为 3 的输入
    batch_size = 3
    dense_input = tf.random.uniform(shape=(batch_size, len(dense_feats)), dtype=tf.float32)
    sparse_input = tf.random.uniform(shape=(batch_size, len(sparse_feats)), maxval=6, dtype=tf.int32)
    # 打印结果
    print("Dense 输入:")
    print(dense_input.numpy())
    print("Sparse 输入:")
    print(sparse_input.numpy())

    # Dense 输入:
    # [[0.19882536 0.9919691 ]
    #  [0.14089882 0.6178216 ]
    #  [0.59311116 0.79255974]]
    # Sparse 输入:
    # [[1 3 3]
    #  [4 4 0]
    #  [1 3 2]]

    # 前向传播
    output = model((sparse_input, dense_input), training=False)

    print("\n模型输出:")
    print(output)

    # 模型输出:
    # (<tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    # array([[0.47370112],
    #        [0.4806958 ],
    #        [0.4883328 ]], dtype=float32)>, <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    # array([[0.43123466],
    #        [0.4493978 ],
    #        [0.4693598 ]], dtype=float32)>)


    # 模型训练评估完之后输出top-N交叉特征-进行数据挖掘-2025年9月6日18:49:58新增内容
    V_matrix = model.V.numpy()  # shape: (num_features, emb_size)
    print(V_matrix)
    # [[-0.02703715  0.01528769 -0.01400328]
    #  [ 0.01407009 -0.03462351 -0.0223598 ]
    #  [-0.03217772 -0.06346221 -0.03414213]
    #  [-0.0573472  -0.10437017 -0.00375469]
    #  [ 0.01148079  0.02796272  0.01585371]]

    # 计算交叉权重矩阵
    cross_weights = np.dot(V_matrix, V_matrix.T)  # shape: (num_features, num_features)
    print(cross_weights)
    print(cross_weights.shape)
    # [[ 0.01674403  0.0044216   0.00503649 -0.0061559  -0.00661108]
    #  [ 0.0044216   0.00236985  0.00069299 -0.0032963  -0.00326076]
    #  [ 0.00503649  0.00069299  0.00269178 -0.00184033 -0.00034924]
    #  [-0.0061559  -0.0032963  -0.00184033  0.00549482  0.00366477]
    #  [-0.00661108 -0.00326076 -0.00034924  0.00366477  0.00535326]]
    # (5, 5)

    # cross_weights 是对称矩阵，V_matrix 是 (num_features, emb_size)
    num_features = cross_weights.shape[0]  # 5
    # num_features和column_names的数量是一致的！！
    column_names = ["platform", "app_name", "app_version", "country", "region"]

    # 保存所有特征对及其交互值（只保留上三角非对角）
    interactions = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if mask[i, j] == 1:  # 只计算 mask=1 的交互权重：只保留训练时允许的交互组合
                interactions.append(((i, j), cross_weights[i, j]))
    print(interactions)
    # [((0, 1), -0.0007552213), ((0, 2), 0.00019944718), ((0, 3), -0.00080472825), ((0, 4), 0.002731351),
    #  ((1, 2), -0.00057267095), ((1, 3), 0.0028101264), ((1, 4), -0.0008638674), ((2, 3), 0.0018121665),
    #  ((2, 4), -0.0049506244), ((3, 4), -0.007953306)]

    # 按绝对值排序（从大到小）输出 Top_k 特征交互对
    print("特征交互对（按交互强度）:")
    top_k_interact = 5
    top_k_list = sorted(interactions, key=lambda x: abs(x[1]), reverse=True)[:top_k_interact]
    print(top_k_list)
    # [((3, 4), -0.007953306), ((2, 4), -0.0049506244), ((1, 3), 0.0028101264), ((0, 4), 0.002731351), ((2, 3), 0.0018121665)]
    # 输出-具体名称-特征交互对

    for (i, j), weight in top_k_list:
        name_i = column_names[i]
        name_j = column_names[j]
        print(f"{i, j}  {name_i} × {name_j} : 权重 = {weight:.6f}")

    # (3, 4)  country × region : 权重 = -0.007953
    # (2, 4)  app_version × region : 权重 = -0.004951
    # (1, 3)  app_name × country : 权重 = 0.002810
    # (0, 4)  platform × region : 权重 = 0.002731
    # (2, 3)  app_version × country : 权重 = 0.001812