# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Lambda, Multiply, Reshape, Permute, Concatenate
from tensorflow.keras import backend as K


# 定义 XdeepFM中的CIN组件
def CIN(input_dim, embedding_dim, layer_dims):
    """
    :param input_dim:     # 特征的数量
    :param embedding_dim: # 每个特征的embedding向量维度
    :param layer_dims:    # 几层CIN => 存储每一层的卷积层
    :return:
    """
    # < KerasTensor
    # shape = (None, 3, 5), dtype = float32, sparse = False, ragged = False, name = keras_tensor >
    input_x = Input(shape=(input_dim, embedding_dim))
    # print(input_x)
    h = input_x
    field_nums = [input_dim]
    xs = []

    for dim in layer_dims:
        F0 = field_nums[0]
        Fk = field_nums[-1]

        # 生成外积交互
        x0_expanded = Lambda(lambda x: K.expand_dims(x, axis=2))(input_x)  # (batch, F0, 1, D)
        h_expanded = Lambda(lambda x: K.expand_dims(x, axis=1))(h)         # (batch, 1, Fk, D)

        # 外积计算
        product = Multiply()([x0_expanded, h_expanded])  # (batch, F0, Fk, D)

        # 调整维度适配Conv1D
        product_reshaped = Reshape((F0 * Fk, embedding_dim))(product)  # (batch, F0*Fk, D)
        product_transposed = Permute((2, 1))(product_reshaped)         # (batch, D, F0*Fk)

        # 特征交叉卷积
        conv_transposed = Conv1D(filters=dim, kernel_size=1, activation='relu')(product_transposed)  # (batch, D, dim)
        conv = Permute((2, 1))(conv_transposed)   # (batch, dim, D)

        xs.append(conv)
        h = conv
        field_nums.append(dim)

    # 聚合所有层输出
    concatenated = Concatenate(axis=1)(xs)  # (batch, sum(layer_dims), D)
    sum_pool = Lambda(lambda x: K.sum(x, axis=2))(concatenated)  # (batch, sum(layer_dims))

    # 最终输出层
    output = Dense(1, use_bias=True)(sum_pool)

    return Model(inputs=input_x, outputs=output)


# 测试代码
if __name__ == "__main__":
    input_dim = 3     # 特征的数量
    embedding_dim = 5 # 每个特征的embedding向量维度
    layer_dims = [7, 9]

    # 创建模型
    model = CIN(input_dim, embedding_dim, layer_dims)
    model.summary()

    # 生成测试数据
    test_input = tf.random.normal(shape=(2, input_dim, embedding_dim))
    print(test_input)

    # 前向传播
    output = model(test_input)
    print("\n测试输出形状:", output.shape)
    print("示例输出值:\n", output.numpy())


# Model: "functional"
# ┌─────────────────────┬───────────────────┬────────────┬───────────────────┐
# │ Layer (type)        │ Output Shape      │    Param # │ Connected to      │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ input_layer         │ (None, 3, 5)      │          0 │ -                 │
# │ (InputLayer)        │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ lambda (Lambda)     │ (None, 3, 1, 5)   │          0 │ input_layer[0][0] │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ lambda_1 (Lambda)   │ (None, 1, 3, 5)   │          0 │ input_layer[0][0] │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ multiply (Multiply) │ (None, 3, 3, 5)   │          0 │ lambda[0][0],     │
# │                     │                   │            │ lambda_1[0][0]    │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ reshape (Reshape)   │ (None, 9, 5)      │          0 │ multiply[0][0]    │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ permute (Permute)   │ (None, 5, 9)      │          0 │ reshape[0][0]     │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ conv1d (Conv1D)     │ (None, 5, 7)      │         70 │ permute[0][0]     │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ permute_1 (Permute) │ (None, 7, 5)      │          0 │ conv1d[0][0]      │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ lambda_2 (Lambda)   │ (None, 3, 1, 5)   │          0 │ input_layer[0][0] │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ lambda_3 (Lambda)   │ (None, 1, 7, 5)   │          0 │ permute_1[0][0]   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ multiply_1          │ (None, 3, 7, 5)   │          0 │ lambda_2[0][0],   │
# │ (Multiply)          │                   │            │ lambda_3[0][0]    │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ reshape_1 (Reshape) │ (None, 21, 5)     │          0 │ multiply_1[0][0]  │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ permute_2 (Permute) │ (None, 5, 21)     │          0 │ reshape_1[0][0]   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ conv1d_1 (Conv1D)   │ (None, 5, 9)      │        198 │ permute_2[0][0]   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ permute_3 (Permute) │ (None, 9, 5)      │          0 │ conv1d_1[0][0]    │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ concatenate         │ (None, 16, 5)     │          0 │ permute_1[0][0],  │
# │ (Concatenate)       │                   │            │ permute_3[0][0]   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ lambda_4 (Lambda)   │ (None, 16)        │          0 │ concatenate[0][0] │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ dense (Dense)       │ (None, 1)         │         17 │ lambda_4[0][0]    │
# └─────────────────────┴───────────────────┴────────────┴───────────────────┘
#  Total params: 285 (1.11 KB)
#  Trainable params: 285 (1.11 KB)
#  Non-trainable params: 0 (0.00 B)
#
# 测试输出形状: (2, 1)
# 示例输出值:
#  [[-2.0499468 ]
#  [-0.85336304]]
