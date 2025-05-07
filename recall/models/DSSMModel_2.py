# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ------------------------------------------------------------------------------
# -- author：张明阳
# -- create：2025年5月8日00:46:53
# -- function：DSSM-练习版本V2-faiss查询用户相似的item
# -- document:
# ------------------------------------------------------------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import GlorotUniform
import faiss

# 定义 DSSM（Deep Structured Semantic Model）模型
class DSSMModel(tf.keras.Model):
    def __init__(self,
                 user_feature_vocab_sizes,  # 用户侧特征的 vocab size 列表
                 item_feature_vocab_sizes,  # 物品侧特征的 vocab size 列表
                 embedding_dim=8,           # embedding 向量维度
                 user_dnn_units=[64, 32],   # 用户侧 DNN 每层的单元数
                 item_dnn_units=[64, 32],   # 物品侧 DNN 每层的单元数
                 dnn_activation='tanh',     # DNN 激活函数
                 dnn_dropout=0.0,           # DNN dropout 比例
                 use_bn=False,              # 是否使用 batch normalization
                 similarity_type='cos'):    # 相似度计算方式：'cos' 或 'dot'
        super(DSSMModel, self).__init__()

        print(user_feature_vocab_sizes,item_feature_vocab_sizes)
        self.similarity_type = similarity_type
        self.use_bn = use_bn

        # 构建用户 embedding 层，每个特征一个 embedding 层
        # GlorotUniform 是一种初始化方法，也叫做 Xavier Uniform 初始化，其核心思想是：
        # 让神经网络的每一层在前向传播和反向传播时，保持激活值和梯度的方差尽可能一致，从而避免梯度爆炸或梯度消失的问题。
        self.user_embeddings = [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer=GlorotUniform())  for vocab_size in user_feature_vocab_sizes
        ]

        # 构建物品 embedding 层，每个特征一个 embedding 层
        self.item_embeddings = [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer=GlorotUniform())  for vocab_size in item_feature_vocab_sizes
        ]

        # 构建用户侧 DNN
        self.user_dnn = self._build_dnn(user_dnn_units, dnn_activation, dnn_dropout, use_bn)

        # 构建物品侧 DNN
        self.item_dnn = self._build_dnn(item_dnn_units, dnn_activation, dnn_dropout, use_bn)

        # 最后接一个 sigmoid 输出
        self.output_layer = Dense(1, activation='sigmoid')


    # 构建 DNN 网络的辅助函数
    def _build_dnn(self, hidden_units, activation, dropout_rate, use_bn):
        layers = []
        for units in hidden_units:
            layers.append(Dense(units, activation=None))
            if use_bn:
                layers.append(BatchNormalization())
            layers.append(tf.keras.layers.Activation(activation))
            if dropout_rate > 0:
                layers.append(Dropout(dropout_rate))
        return tf.keras.Sequential(layers)

        # # 类似:DNN layers
        # self.dnn = tf.keras.Sequential([
        #     layers.Dense(200, activation='relu'),
        #     layers.Dropout(0.3),
        #     layers.Dense(200, activation='relu'),
        #     layers.Dropout(0.2),
        #     layers.Dense(200, activation='relu'),
        #     layers.Dropout(0.2),
        #     layers.Dense(1)
        # ])


    # 前向传播函数
    def call(self, inputs, training=False):
        # inputs传入进来的是:x=[user_inputs, item_inputs]
        user_inputs, item_inputs = inputs  # 拆分用户输入和物品输入

        # 计算用户嵌入并拼接
        user_embed = [emb(tf.cast(inp, tf.int32)) for emb, inp in zip(self.user_embeddings, user_inputs)]  # print(user_embed)    # [shape=(5, 1, 8),shape=(5, 1, 8)]
        user_concat = tf.concat(user_embed, axis=-1)               # print(user_concat)   # shape=(5, 1, 16)
        user_flat = Flatten()(user_concat)                         # print(user_flat)     # shape=(5, 16)
        user_vector = self.user_dnn(user_flat, training=training)  # print(user_vector)   # shape=(5, 32)


        # 计算物品嵌入并拼接
        item_embed = [emb(tf.cast(inp, tf.int32)) for emb, inp in zip(self.item_embeddings, item_inputs)]
        item_concat = tf.concat(item_embed, axis=-1)
        item_flat = Flatten()(item_concat)
        item_vector = self.item_dnn(item_flat, training=training)   # shape=(5, 32)


        # 相似度计算（cosine 或 dot）
        if self.similarity_type == 'cos':
            user_norm = tf.nn.l2_normalize(user_vector, axis=-1)
            item_norm = tf.nn.l2_normalize(item_vector, axis=-1)
            sim = tf.reduce_sum(user_norm * item_norm, axis=-1, keepdims=True)
            # print(sim)
            # tf.Tensor(
            #     [[-0.03882238]
            #      [-0.19483174]
            #      [0.11280522]
            #      [-0.00850978]
            #      [0.3569387]], shape=(5, 1), dtype=float32)
        elif self.similarity_type == 'dot':
            sim = tf.reduce_sum(user_vector * item_vector, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")

        # 通过 sigmoid 输出层输出匹配概率
        output = self.output_layer(sim)
        return output


    """
     为什么 get_user_embedding 和 get_item_embedding 和 call() 中代码几乎一样？
     这是因为：
           在 call() 中你做的是：整体 forward，预测点击/评分/相关性
           而在 get_user_embedding() 和 get_item_embedding() 中，你是：单独获取用户或物品向量（可用于召回、可视化等）
     两者流程本质一致，只是调用目的不同：
    """
    # 获取用户向量
    def get_user_embedding(self, user_inputs, training=False):
        user_embed = [emb(tf.cast(inp, tf.int32)) for emb, inp in zip(self.user_embeddings, user_inputs)]
        user_concat = tf.concat(user_embed, axis=-1)
        user_flat = Flatten()(user_concat)
        return self.user_dnn(user_flat, training=training)


    # 获取物品向量
    def get_item_embedding(self, item_inputs, training=False):
        item_embed = [emb(tf.cast(inp, tf.int32)) for emb, inp in zip(self.item_embeddings, item_inputs)]
        item_concat = tf.concat(item_embed, axis=-1)
        item_flat = Flatten()(item_concat)
        return self.item_dnn(item_flat, training=training)





if __name__ == '__main__':
    # 设置用户和物品特征的词表大小
    user_feature_vocab_sizes = [1000, 3]   # 比如：user_id 取值范围 0-999，gender 取值范围 0-2
    item_feature_vocab_sizes = [2000, 50]  # 比如：item_id 和 item 类别 category

    # 初始化模型
    model = DSSMModel(user_feature_vocab_sizes, item_feature_vocab_sizes)

    # 编译模型，采用 Adam 优化器和二分类交叉熵损失，评估指标为 AUC
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    user_inputs = [
        np.random.randint(0, 1000, size=(5, 1)),  # 模拟 user_id 输入
        np.random.randint(0, 3, size=(5, 1))      # 模拟 gender 输入
    ]
    # print(user_inputs)
    # [array([[267],
    #         [386],
    #         [500],
    #         [643],
    #         [173]]), array([[2],
    #                         [0],
    #                         [1],
    #                         [0],
    #                         [0]])]

    item_inputs = [
        np.random.randint(0, 2000, size=(5, 1)),  # 模拟 item_id 输入
        np.random.randint(0, 50, size=(5, 1))     # 模拟 item category 输入
    ]
    # print(item_inputs)
    # [array([[143],
    #         [1452],
    #         [862],
    #         [1919],
    #         [1551]]), array([[7],
    #                          [2],
    #                          [36],
    #                          [12],
    #                          [4]])]

    labels = np.random.randint(0, 2, size=(5, 1))  # 模拟点击标签 0/1

    # 训练模型
    # model.fit(x=[user_inputs, item_inputs], y=labels, epochs=3, batch_size=32)
    """
    在使用 model.fit(...) 时，即使你在 call 方法中加入了 print(...)，很多时候 打印出来的并不是具体的数值，而是 Tensor 信息（例如：<tf.Tensor: shape=(32, 64), dtype=float32, numpy=...> 或 tf.Tensor(..., shape=..., dtype=...)）。
    这背后的原因是：model.fit() 在训练过程中是运行在 TensorFlow 的 图模式（graph mode） 下的，而不是 Eager 模式。
    所以model.fit并不适合调试，记住这个技巧。
    """
    outputs = model([user_inputs, item_inputs], training=False)
    # print(outputs)
    # tf.Tensor(
    # [[0.52732146]
    #  [0.471136  ]
    #  [0.50886   ]
    #  [0.5142037 ]
    #  [0.51859426]], shape=(5, 1), dtype=float32)



    # 假设此时model已经fit训练完了。。。
    # 获取用户嵌入和商品嵌入
    user_embeddings = model.get_user_embedding(user_inputs).numpy()  # 转为numpy数组  # shape=(5, 32)
    item_embeddings = model.get_item_embedding(item_inputs).numpy()  # 转为numpy数组  # shape=(5, 32)



    """
    基于 L2 距离获取user的相似item
    """
    # ----------------------------- 基于 L2 距离 -----------------------------
    # 1. 创建 Faiss 索引（基于 L2 距离）
    dimension = item_embeddings.shape[1]  # 商品嵌入向量的维度
    index = faiss.IndexFlatL2(dimension)  # 创建 L2 距离索引

    # 2. 将商品嵌入添加到 Faiss 索引中
    index.add(item_embeddings.astype(np.float32))  # 添加商品向量到索引中

    # 3. 查询与某个用户嵌入最相似的商品
    user_index = 0  # 假设你想找到与第一个用户最相似的商品
    user_vector = user_embeddings[user_index].reshape(1, -1).astype(np.float32)  # 获取该用户的嵌入向量

    # 4. 使用 Faiss 查找最相似的商品
    k = 3  # 查找与用户最相似的前3个商品
    distances, indices = index.search(user_vector, k)

    # 5. 输出最相似的商品
    print(f"用户{user_index+1}最相似的商品：")
    for i in range(k):
        print(f"商品{indices[0][i]} (距离: {distances[0][i]:.4f})")
    # 用户1最相似的商品：
    # 商品3(距离: 1.6875)
    # 商品0(距离: 1.9085)
    # 商品4(距离: 2.1107)


    """
    基于余弦相似度获取user的相似item
    """
    # -------------------------- 基于余弦相似度 --------------------------
    # 假设你已经获得 item_embeddings 和 user_embeddings，是 NumPy 数组
    # 归一化函数
    def l2_normalize(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    # Step 1: 向量归一化
    normalized_item_embeddings = l2_normalize(item_embeddings.astype(np.float32))
    normalized_user_embeddings = l2_normalize(user_embeddings.astype(np.float32))

    # Step 2: 创建 Faiss 内积索引
    dimension = normalized_item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 基于内积的索引（需配合归一化使用）

    # Step 3: 添加商品向量（已归一化）
    index.add(normalized_item_embeddings)

    # Step 4: 查询某个用户最相似的商品（同样需要归一化）
    user_index = 0
    query = normalized_user_embeddings[user_index].reshape(1, -1)

    # Step 5: 查询 Top-K 相似商品
    k = 5
    scores, item_indices = index.search(query, k)

    # Step 6: 输出结果
    print(f"用户 {user_index+1} 最相似的商品：")
    for i in range(k):
        print(f"商品 {item_indices[0][i]}（余弦相似度: {scores[0][i]:.4f}）")
    # 用户 1 最相似的商品：
    # 商品 3（余弦相似度: 0.1587）
    # 商品 2（余弦相似度: -0.0046）
    # 商品 0（余弦相似度: -0.0220）
    # 商品 4（余弦相似度: -0.1411）
    # 商品 1（余弦相似度: -0.2129）