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

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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
    def __init__(self, feat_columns, emb_size=5):
        super().__init__()
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.seq_feats = feat_columns[2] if len(feat_columns) > 2 else []

        self.emb_size = emb_size

        self.linear_dense = layers.Dense(1)

        self.first_sparse_embs  = [layers.Embedding(input_dim=feat['feat_num'], output_dim=1)        for feat in self.sparse_feats]
        self.second_sparse_embs = [layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size) for feat in self.sparse_feats]

        # Sequence embedding layers
        # 仅当存在序列特征时才构造 seq embedding 层
        if self.seq_feats:
            self.seq_embeds = [layers.Embedding(input_dim=feat['feat_num'], output_dim=self.emb_size, mask_zero=True) for feat in self.seq_feats]

        # 这行代码的作用是计算在 NFM 模型 中，给定 sparse_size 个特征（即有多少个离散字段参与 embedding），两两组合的交叉数量（即有多少种不同的二阶特征交互对）
        # num_pairs=C(n,2)
        total_fields = len(self.sparse_feats) + (len(self.seq_feats) if self.seq_feats else 0)
        self.num_pairs = total_fields * (total_fields - 1) // 2
        self.dnn = tf.keras.Sequential([
            tf.keras.Input(shape=(self.num_pairs * self.emb_size,)),  # 添加这行，其实我感觉不加也行，但是在fit的时候如果不添加会报错
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
        sparse_inputs = inputs[0]   # shape: (batch_size, num_sparse)
        dense_inputs  = inputs[1]   # shape: (batch_size, num_dense)
        seq_inputs    = inputs[2:] if self.seq_feats else []  # list of tensors, each shape each shape each shape: (batch_size, max_seq_len)
        # Dense 输入:
        # [[0.211972   0.3256514 ]
        #  [0.58325326 0.5058359 ]]
        # Sparse 输入:
        # [[5 4 1]
        #  [3 2 3]]
        # seq_inputs 输入:
        # [array([[2, 3, 4],
        #         [3, 5, 0],
        #         [2, 4, 6]]), array([[2, 3],
        #                             [3, 4],
        #                             [2, 5]])]

        # 第一部分：线性部分
        linear_dense_out  = self.linear_dense(dense_inputs)
        linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.first_sparse_embs)],axis=1)
        # print("--------------1-----------")
        # print(linear_sparse_out)
        # [[ 0.01539519 -0.04832922 -0.02953873]
        #  [ 0.02253106  0.00941402  0.04219601]]

        linear_sparse_out = tf.reduce_sum(linear_sparse_out, axis=1, keepdims=True)
        first_order_output = linear_dense_out + linear_sparse_out
        # print(first_order_output)
        # [[0.22632796]
        #  [0.6256093 ]]


        # 第二部分：NFM-DNN部分
        sparse_embeddings = tf.stack([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.second_sparse_embs)], axis=1)  # shape: [batch, field_num, emb_size]

        # ---------- 序列特征部分 ----------
        if self.seq_feats:
            seq_embeds = []
            for i, (seq_input, seq_layer) in enumerate(zip(seq_inputs, self.seq_embeds)):
                seq_emb = seq_layer(seq_input)  # (batch_size, seq_len, emb_dim)
                pooled = tf.reduce_mean(seq_emb, axis=1, keepdims=True)  # (batch_size, 1, emb_dim)  从这可以看出边长序列的字段数据最终整体也是当成一个字段处理
                seq_embeds.append(pooled)
            seq_embeds_concat = tf.concat(seq_embeds, axis=1)  # (batch_size, num_seq_fields, emb_dim)
            # 稀疏和离散拼接到一起
            nfm_input =tf.concat([sparse_embeddings, seq_embeds_concat], axis=1)  # 在第二个维度拼接    # shape = (2, 3 + 7, 5) → 即 (2, field_num + num_seq_fields, 5)
        else:
            nfm_input = sparse_embeddings

        field_num = nfm_input.shape[1]
        element_wise_products = []
        for i in range(field_num):
            for j in range(i + 1, field_num):
                element_wise_products.append(nfm_input[:, i] * nfm_input[:, j])
        element_wise_products = tf.stack(element_wise_products, axis=1)                                        # [batch, num_pairs, emb_size]
        nfm_flatten_input = tf.reshape(element_wise_products, shape=(tf.shape(element_wise_products)[0], -1))  # [batch_size, num_pairs * embedding_dim]

        dnn_output = self.dnn(nfm_flatten_input, training=training)

        # 合成最终输出
        logits = first_order_output  + dnn_output

        # 分支输出
        finish_output = self.finish_output_layer(logits)
        like_output   = self.like_output_layer(logits)

        return {'finish': finish_output, 'like': like_output}


if __name__ == '__main__':
    use_sequence = True

    if not use_sequence:
        # 1. 不使用序列特征
        dense_feats = ['I1', 'I2']
        sparse_feats = ['C1', 'C2', 'C3']
        feat_columns = [
            [{'feat': 'I1'}, {'feat': 'I2'}],
            [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
        ]

        model = NFM(feat_columns=feat_columns, emb_size=5)
        sparse_input = np.array([[1, 2, 3], [4, 5, 5], [1, 2, 3]])
        dense_input = np.random.random((3, len(dense_feats)))
        output = model((sparse_input, dense_input), training=False)

        print("Dense 输入:")
        print(dense_input)
        print("Sparse 输入:")
        print(sparse_input)
        print("\n模型输出:")
        print(output)
        model.summary()
    else:
    # 2. 使用序列特征
        dense_feats = ['I1', 'I2']
        sparse_feats = ['C1', 'C2', 'C3']
        sequence_feats = ['S1', 'S2']
        feat_columns = [
            [{'feat': 'I1'}, {'feat': 'I2'}],
            [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}],
            [{'feat': 'S1', 'feat_num': 10}, {'feat': 'S2', 'feat_num': 20}]
        ]

        model = NFM(feat_columns=feat_columns, emb_size=5)
        sparse_input = np.array([[1, 2, 3], [4, 5, 5], [1, 2, 3]])
        dense_input = np.random.random((3, len(dense_feats)))

        S1 = ['movie1|movie2|movie3', 'movie2|movie5', 'movie1|movie3|movie4']
        S2 = ['movie6|movie7', 'movie7|movie8', 'movie6|movie9']
        sequence_inputs = {}
        tokenizers = {}
        for feat, texts in zip(sequence_feats, [S1, S2]):
            tokenizer = Tokenizer(oov_token='OOV')
            tokenizer.fit_on_texts(texts)
            padded = pad_sequences(tokenizer.texts_to_sequences(texts), padding='post')
            sequence_inputs[feat] = padded
            tokenizers[feat] = tokenizer

        seq_input_list = [sequence_inputs[feat] for feat in sequence_feats]
        output = model((sparse_input, dense_input, *seq_input_list), training=False)

        print("Dense 输入:")
        print(dense_input)
        print("Sparse 输入:")
        print(sparse_input)
        for feat in sequence_feats:
            print(f"{feat} 输入:")
            print(sequence_inputs[feat])
        print("\n模型输出:")
        print(output)

        model.summary()
