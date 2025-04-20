# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
from config.data_config import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class WideAndDeep(Model):
    def __init__(self, feat_columns, emb_size=5):
        super().__init__()
        # feat_columns = [
        #     [{'feat': 'I1'}, {'feat': 'I2'}],
        #     [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
        # ]
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.seq_feats = feat_columns[2] if len(feat_columns) > 2 else []

        self.emb_size = emb_size

        # Linear
        self.linear_dense = layers.Dense(1)
        self.linear_sparse_embeds = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=1)         for feat in self.sparse_feats
        ]

        # 离散 embedding
        self.sparse_embeds = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size)  for feat in self.sparse_feats
        ]

        # Sequence embedding layers
        # 仅当存在序列特征时才构造 seq embedding 层
        if self.seq_feats:
            self.seq_embeds = [layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size, mask_zero=True) for feat in self.seq_feats]

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
        self.like_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='like')


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
        linear_dense_out = self.linear_dense(dense_inputs)
        linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.linear_sparse_embeds)],
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

        # 第二部分：DNN 部分
        # 1️⃣ 稀疏特征 embedding
        sparse_embeds = tf.stack([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.sparse_embeds)],axis=1)  # shape: (batch_size, num_sparse_fields, emb_dim)
        # print(sparse_embeds) shape: (batch_size=2, field_num=3, embedding_dim=5)
        # Tensor("stack:0", shape=(2, 3, 5), dtype=float32)
        # tf.Tensor(
        # [[[-0.0292243   0.03134212 -0.00664638  0.0308771   0.03662998]
        #   [-0.02252715  0.00618609 -0.0408314   0.0155008   0.00702292]
        #   [ 0.02527222  0.02442351  0.01027572  0.04815536  0.01610643]]

        #  [[ 0.01005882 -0.03315624 -0.0195043   0.01774564  0.03821408]
        #   [-0.03824542  0.00229248  0.00047214  0.0488669  -0.04776417]
        #   [-0.01696395 -0.00136379  0.04921383  0.04019973 -0.00026955]]], shape=(2, 3, 5), dtype=float32)

        sparse_flatten_embeddings = tf.reshape(sparse_embeds, shape=(-1, len(self.sparse_feats) * self.emb_size))  # shape: (batch_size, num_sparse_fields * emb_dim)
        # print(flatten_embeddings)
        # Tensor("Reshape:0", shape=(2, 15), dtype=float32)
        # tf.Tensor(
        # [[-0.03405142  0.01116457 -0.00488006  0.03367222 -0.04788997  0.0262876 0.04647902 -0.01162871  0.03328068  0.04433748  0.02085209  0.01660527
        #   -0.02046416  0.00683039  0.04853446]
        #  [-0.03806484  0.0479795   0.0132894  -0.03121579 -0.0166074   0.00733398
        #    0.00708617 -0.00899755  0.02732437 -0.00605234 -0.02896208  0.02931662
        #    0.0044607   0.03854013  0.04758653]], shape=(2, 15), dtype=float32)

        # ---------- 序列特征部分 ----------
        if self.seq_feats:
            seq_embeds = []
            for i, (seq_input, seq_layer) in enumerate(zip(seq_inputs, self.seq_embeds)):
                seq_emb = seq_layer(seq_input)  # (batch_size, seq_len, emb_dim)
                pooled = tf.reduce_mean(seq_emb, axis=1,keepdims=True)  # (batch_size, 1, emb_dim)  从这可以看出边长序列的字段数据最终整体也是当成一个字段处理
                seq_embeds.append(pooled)
            # 拼接所有嵌入特征
            seq_embeds_concat = tf.concat(seq_embeds, axis=1)  # (batch_size, num_seq_fields, emb_dim)
            seq_flatten_embeddings = tf.reshape(seq_embeds_concat, shape=(-1, seq_embeds_concat.shape[1] * self.emb_size))  # shape: (batch_size, num_seq_fields * emb_dim)

            # sparse、seq、dense进行拼接
            dnn_input = tf.concat([dense_inputs, sparse_flatten_embeddings, seq_flatten_embeddings],axis=1)  # shape=(2, X) + shape=(2, Y) + shape=(2, Z)  => shape=(2, 17))
        else:
            # sparse、dense进行拼接
            dnn_input = tf.concat([dense_inputs, sparse_flatten_embeddings], axis=1)

        dnn_output = self.dnn(dnn_input, training=training)

        logits = first_order_output + dnn_output

        # 分支输出
        finish_output = self.finish_output_layer(logits)
        like_output = self.like_output_layer(logits)

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

        model = WideAndDeep(feat_columns=feat_columns, emb_size=5)
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

        model = WideAndDeep(feat_columns=feat_columns, emb_size=5)
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
