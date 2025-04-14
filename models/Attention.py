# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media

import tensorflow as tf
import numpy as np



# 定义 Transform-attention 类（使用 MultiHeadAttention）
class Attention(tf.keras.layers.Layer):
    def __init__(self,num_heads=1, key_dim=5):
        super().__init__()
        # 使用 MultiHeadAttention，设置单头 Attention（num_heads=1）
        # key_dim 设置为 5，TensorFlow 会自动将 Value 的维度设置为 key_dim = 5，虽然Value 的维度不一定和key_dim相同
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads, key_dim)


    def call(self, inputs):
        # 计算 Attention 输出，Query、Key 和 Value 都使用 inputs（自注意力机制）
        attn_output = self.attention(query=inputs, value=inputs, key=inputs)
        # **去掉 Dense 层，直接返回 Attention 输出**
        return attn_output




if __name__ == '__main__':
    # **生成示例输入数据**
    batch_size = 2     # 批次大小
    seq_length = 3     # 序列长度（时间步数）
    feature_dim = 4    # 特征维度，与 depth 相同

    # 随机生成输入数据（形状为 [batch_size, seq_length, feature_dim]）
    np.random.seed(42)
    inputs = np.random.rand(batch_size, seq_length, feature_dim).astype(np.float32)

    print("输入数据形状:", inputs.shape)
    print("输入数据:\n", inputs)

    # **实例化 attention 层并计算 Attention 输出**
    att_layer = Attention(num_heads=2, key_dim=5)
    output = att_layer(inputs)  # 计算 Attention 输出

    print("Attention 输出形状:", output.shape)
    print("Attention 输出:\n", output.numpy())



    # 输入数据形状: (2, 3, 4)
    # 输入数据:
    #  [[[0.37454012 0.9507143  0.7319939  0.5986585 ]
    #   [0.15601864 0.15599452 0.05808361 0.8661761 ]
    #   [0.601115   0.7080726  0.02058449 0.96990985]]
    #
    #  [[0.83244264 0.21233912 0.18182497 0.1834045 ]
    #   [0.30424225 0.52475643 0.43194503 0.29122913]
    #   [0.6118529  0.13949387 0.29214466 0.36636186]]]
    #
    # Attention 输出形状: (2, 3, 4)
    # Attention 输出:
    #  [[[-0.2895045  -0.08634619 -0.19754064 -0.06824519]
    #   [-0.2842324  -0.09126601 -0.19701034 -0.06944697]
    #   [-0.28259432 -0.09098882 -0.19530196 -0.07158254]]
    #
    #  [[-0.05227944 -0.3694144   0.12888485  0.16831942]
    #   [-0.05305324 -0.3704375   0.12951234  0.17002591]
    #   [-0.05244714 -0.3696884   0.1290273   0.16876723]]]