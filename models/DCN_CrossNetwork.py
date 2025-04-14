# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers



# 定义 CrossNetwork的核心组件
class CrossNetwork(layers.Layer):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        # 定义多个交叉层的权重和偏置
        self.ws = [self.add_weight(
            shape=(input_dim, 1),
            initializer='random_normal',
            trainable=True,
            name=f'cross_weight_{i}'
        ) for i in range(num_layers)]

        self.bs = [self.add_weight(
            shape=(input_dim,),
            initializer='zeros',
            trainable=True,
            name=f'cross_bias_{i}'
        ) for i in range(num_layers)]

    # 原始的Cross公式： x = x0 * xw + self.bs[i] + x
    # X1 = X0(X0TW1) + X0
    # X2 = X0(X1TW1) + X1
    # X3 = X0(X2TW1) + X2
    # X4 = X0(X2TW1) + X3
    # 以前都是X2 = X1W 现在是X2 = X0(X1TW1) + X1 相当于定义了一种新的网络 ，传统的网络没有做向量的交叉
    def call(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = tf.matmul(x, self.ws[i])  # (batch_size, 1)
            x = x0 * xw + self.bs[i] + x   # (batch_size, input_dim)
        return x


# ===================== 2. DCN 模型定义 =====================
class DCNModel(tf.keras.Model):
    def __init__(self, input_dim, num_cross_layers=2, hidden_units=[128, 64]):
        super().__init__()
        self.cross_network = CrossNetwork(input_dim=input_dim, num_layers=num_cross_layers)

        self.deep_network = tf.keras.Sequential([
            layers.Dense(hidden_units[0], activation='relu'),
            layers.Dense(hidden_units[1], activation='relu'),
            layers.Dense(1)
        ])

        self.cross_proj = layers.Dense(1)
        self.final_activation = layers.Activation('sigmoid')

    def call(self, inputs):
        cross_out = self.cross_network(inputs)         # (batch, input_dim)
        cross_out = self.cross_proj(cross_out)         # (batch, 1)

        deep_out = self.deep_network(inputs)           # (batch, 1)

        output = cross_out + deep_out
        return self.final_activation(output)




if __name__ == '__main__':
    # 生成试算输入数据
    input_dim   = 3  # 输入特征的维度，输入特征的数量
    num_samples = 10  # 样本数量

    # 随机生成输入数据
    inputs = np.random.randn(num_samples, input_dim).astype(np.float32)
    inputs_tensor = tf.convert_to_tensor(inputs)  # 转换为 TensorFlow 张量
    print(inputs_tensor) # shape=(10, 3)

    # ===================== 3. 创建模型并进行预测 =====================
    # 创建模型
    model = DCNModel(input_dim=input_dim, num_cross_layers=2, hidden_units=[128, 64])

    # 通过模型执行一次前向传播，获取预测结果
    predictions = model(inputs_tensor)

    # 打印预测结果
    print(f"预测结果形状: {predictions.shape}")
    print(f"前5个预测结果: {predictions.numpy()[:5]}")


# tf.Tensor(
# [[-0.9539075  -0.11811007 -0.20525643]
#  [-0.4525233   0.44655284  0.55119354]
#  [ 0.6650196   0.7107557   0.21343742]
#  [-0.08349546  0.9756209  -0.51805735]
#  [-0.5408493  -1.3536524   2.7683518 ]
#  [-1.5353589  -2.5298598  -0.61319643]
#  [-0.9921559   0.5195189   2.0009966 ]
#  [ 0.75709236  1.3475354  -0.00501939]
#  [ 1.0561619   1.3997902   0.20283914]
#  [ 0.09164997 -0.12785815 -0.18233587]], shape=(10, 3), dtype=float32)
# 预测结果形状: (10, 1)
# 前5个预测结果: [[0.70425874]
#  [0.585526  ]
#  [0.3342261 ]
#  [0.38398612]
#  [0.9810456 ]]