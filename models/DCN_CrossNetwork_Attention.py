#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media

import tensorflow as tf
import numpy as np

class VectorWiseAttentionCross(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads=2, key_dim=8):
        """
        向量级 Attention Cross 层（Vector-wise Attention Cross Layer）

        参数:
        - num_layers: Cross 层数量
        - num_heads: 多头注意力头数
        - key_dim: 注意力每个头的维度
        """
        super().__init__()
        self.num_layers = num_layers
        self.attention_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
            for _ in range(num_layers)
        ]
        self.biases = [
            self.add_weight(shape=(1, 1, embed_dim), initializer='zeros', trainable=True, name=f"cross_bias_{i}")
            for i in range(num_layers)
        ]

        self.norms = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

    def call(self, x):
        """
        前向传播：向量级注意力交叉计算

        参数:
        - x: Tensor, 输入特征张量，形状为 [batch_size, num_fields, embed_dim]。
            - batch_size: 批量大小，表示样本数量。
            - num_fields: 特征的数量，表示每个样本包含的特征维度数。
            - embed_dim: 嵌入维度，表示每个特征的向量表示的维度。

        返回:
        - x: Tensor, 输出特征张量，形状为 [batch_size, num_fields, embed_dim]。
            - 经过多层向量级交叉计算和注意力机制处理后的特征表示。
            - 形状与输入相同，包含了每个样本在每个特征维度的更新表示。

        备注:
        - 每一层的计算过程包括：
            1. 计算当前特征与初始特征之间的 Attention 权重，使用 `self.attention_layers[i]` 来实现。
            2. 基于 Attention 输出和原始特征进行 **向量级交叉**，更新特征表示。
            3. 对更新后的特征进行 **层归一化**，避免训练中出现不稳定的数值（如梯度爆炸或消失）。
        """
        x0 = x
        for i in range(self.num_layers):
            attn_output = self.attention_layers[i](query=x, key=x0, value=x0)  # [batch, fields, embed_dim]
            x = x0 * attn_output + self.biases[i] + x  # 向量级交叉
            x = self.norms[i](x)  # 层归一化防止多层交叉层之后，值可能爆炸（或太小），导致后续网络不稳定

        return x


# 测试用例
if __name__ == '__main__':
    batch_size = 2
    num_fields = 3
    embed_dim = 5
    num_layers = 5
    num_heads = 2
    key_dim = 4  # 通常设置为 embed_dim

    # 生成示例输入 [batch_size, num_fields, embed_dim]
    np.random.seed(42)
    inputs = np.random.rand(batch_size, num_fields, embed_dim).astype(np.float32)

    print("输入数据形状:", inputs.shape)
    print("输入数据:\n", inputs)

    # 实例化并测试向量级交叉层
    attn_cross = VectorWiseAttentionCross(embed_dim = embed_dim,num_layers=num_layers, num_heads=num_heads, key_dim=key_dim)
    output = attn_cross(inputs)
    print(attn_cross.biases)

    print("\nVectorWiseAttentionCross 输出形状:", output.shape)
    print("输出数据:\n", output.numpy())
