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


class DeepFM_MTL(Model):
    def __init__(self, feat_columns, emb_size,sequence_metadata=None):
        """
        :param feat_columns:
        :param emb_size:
        :param sequence_metadata: {'tokenizers': {'actors': <keras_preprocessing.text.Tokenizer object at 0x0000029E0D126250>, 'genres': <keras_preprocessing.text.Tokenizer object at 0x0000029E0D126E50>}, 'pad_len_dict': {'actors': 2, 'genres': 2}}
        """
        super().__init__()
        self.dense_feats, self.sparse_feats, self.sequence_feats = feat_columns[0], feat_columns[1],feat_columns[2]
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

        # 增加序列离散数据的Embedding-V矩阵
        self.sequence_emb_layers  = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size)
            for feat in self.sequence_feats
        ]

        self.dnn = tf.keras.Sequential([
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
        sparse_inputs   = inputs[0]   # shape: (batch_size, num_sparse)
        dense_inputs    = inputs[1]   # shape: (batch_size, num_dense)
        sequence_inputs = inputs[2:]  # list of tensors, each shape: (batch_size, max_seq_len)
        # Dense 输入:
        # [[0.211972   0.3256514 ]
        #  [0.58325326 0.5058359 ]]
        # Sparse 输入:
        # [[5 4 1]
        #  [3 2 3]]

        # 第一部分：线性部分
        linear_dense_out = self.linear_dense(dense_inputs)
        linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.first_order_sparse_emb)],
                                      axis=1)
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
        embeddings = tf.stack([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.second_order_sparse_emb)], axis=1)
        # print(embeddings) shape: (batch_size=2, field_num=3, embedding_dim=5)
        # Tensor("stack:0", shape=(2, 3, 5), dtype=float32)
        # tf.Tensor(
        # [[[-0.0292243   0.03134212 -0.00664638  0.0308771   0.03662998]
        #   [-0.02252715  0.00618609 -0.0408314   0.0155008   0.00702292]
        #   [ 0.02527222  0.02442351  0.01027572  0.04815536  0.01610643]]

        #  [[ 0.01005882 -0.03315624 -0.0195043   0.01774564  0.03821408]
        #   [-0.03824542  0.00229248  0.00047214  0.0488669  -0.04776417]
        #   [-0.01696395 -0.00136379  0.04921383  0.04019973 -0.00026955]]], shape=(2, 3, 5), dtype=float32)

        # ---------- 序列特征部分 ----------
        sequence_embeds = []
        for i, (seq_input, seq_emb_layer) in enumerate(zip(sequence_inputs, self.sequence_emb_layers)):
            seq_embed = seq_emb_layer(seq_input)                            # (batch_size, seq_len, embedding_dim)
            pooled_embed = tf.reduce_mean(seq_embed, axis=1,keepdims=True)  # shape: (batch_size, 1, embedding_dim) 从这可以看出边长序列的字段数据最终整体也是当成一个字段处理
            sequence_embeds.append(pooled_embed)

        # 拼接所有嵌入特征
        merged_sequence_embeds = tf.concat(sequence_embeds, axis=1)        # shape: (batch_size, num_fields, embedding_dim)

        # ---------- 序列特征部分 ----------
        sequence_embeds = []
        for i, (seq_input, seq_emb_layer) in enumerate(zip(sequence_inputs, self.sequence_emb_layers)):
            seq_embed = seq_emb_layer(seq_input)                            # (batch_size, seq_len, embedding_dim)
            pooled_embed = tf.reduce_mean(seq_embed, axis=1,keepdims=True)  # shape: (batch_size, 1, embedding_dim) 从这可以看出边长序列的字段数据最终整体也是当成一个字段处理
            sequence_embeds.append(pooled_embed)

        # 拼接所有嵌入特征
        merged_sequence_embeds = tf.concat(sequence_embeds, axis=1)        # shape: (batch_size, num_fields, embedding_dim)

        combined_embeddings = tf.concat([embeddings, merged_sequence_embeds], axis=1)
        # shape: (batch_size, num_sparse_fields + num_sequence_fields, embedding_dim)


        # 这一步对每个样本的所有 sparse embedding 向量在特征维度上求和：对于某个样本，就是把它的三个特征的 embedding 向量相加，变成一个总的表示
        # (2, 3, 5) → [2, 5]
        summed = tf.reduce_sum(combined_embeddings, axis=1)
        # (2, 5) → [2, 5]
        squared_sum = tf.square(summed)
        # 先对每个 embedding 向量做逐元素平方，然后再对所有 sparse 特征做求和 (2, 3, 5) → [2, 5]
        squared = tf.reduce_sum(tf.square(embeddings), axis=1)
        # [2, 1]，表示每个样本的二阶交叉值（标量）
        second_order = 0.5 * tf.reduce_sum(squared_sum - squared, axis=1, keepdims=True)
        # print(second_order)
        # [[0.00537243]
        #  [0.00075581]]

        # 第三部分：DNN 部分
        flatten_embeddings = tf.reshape(embeddings, shape=(-1, len(self.sparse_feats) * self.emb_size))
        # print(flatten_embeddings)
        # Tensor("Reshape:0", shape=(2, 15), dtype=float32)
        # tf.Tensor(
        # [[-0.03405142  0.01116457 -0.00488006  0.03367222 -0.04788997  0.0262876 0.04647902 -0.01162871  0.03328068  0.04433748  0.02085209  0.01660527
        #   -0.02046416  0.00683039  0.04853446]
        #  [-0.03806484  0.0479795   0.0132894  -0.03121579 -0.0166074   0.00733398
        #    0.00708617 -0.00899755  0.02732437 -0.00605234 -0.02896208  0.02931662
        #    0.0044607   0.03854013  0.04758653]], shape=(2, 15), dtype=float32)


        flatten_merged_sequence_embeds = tf.reshape(merged_sequence_embeds, shape=(-1, merged_sequence_embeds.shape[1] * self.emb_size))  # shape: (batch_size, num_fields * embedding_dim)

        dnn_input = tf.concat([dense_inputs, flatten_embeddings,flatten_merged_sequence_embeds], axis=1) # shape=(2, 15) + shape=(2, 2) => shape=(2, 17)
        dnn_output = self.dnn(dnn_input, training=training)

        logits = first_order_output + second_order + dnn_output

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

    # 初始化模型
    model = DeepFM_MTL(feat_columns=feat_columns, emb_size=5)

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


    # Dense 输入:
    # [[0.19882536 0.9919691 ]
    #  [0.14089882 0.6178216 ]
    #  [0.59311116 0.79255974]]
    # Sparse 输入:
    # [[1 3 3]
    #  [4 4 0]
    #  [1 3 2]]
    #
    # 模型输出:
    # (<tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    # array([[0.47370112],
    #        [0.4806958 ],
    #        [0.4883328 ]], dtype=float32)>, <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    # array([[0.43123466],
    #        [0.4493978 ],
    #        [0.4693598 ]], dtype=float32)>)

