# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
from config.data_config import *


class DeepFM_Sequence_MTL(Model):
    def __init__(self, feat_columns, emb_size):
        super().__init__()
        self.dense_feats, self.sparse_feats, self.sequence_feats = feat_columns[0], feat_columns[1], feat_columns[2]
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

        self.sequence_emb = {
            feat['feat']: layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size, mask_zero=True)
            for feat in self.sequence_feats
        }

        self.dnn = tf.keras.Sequential([
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])

        self.finish_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='like')



    def call(self, inputs, training=False):
        sparse_inputs, dense_inputs, sequence_inputs = inputs

        # 第一部分：Liner部分
        # dense_inputs shape: (batch_size, dense_size)
        linear_dense_out = self.linear_dense(dense_inputs)  # linear_dense_out shape: (batch_size, 1)

        linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.first_order_sparse_emb)],axis=1)
        # sparse_emb: (batch_size, 1) for each sparse feature, so linear_sparse_out shape: (batch_size, num_sparse_features)

        linear_sparse_out = tf.reduce_sum(linear_sparse_out, axis=1,keepdims=True)  # linear_sparse_out shape: (batch_size, 1)
        first_order_output = linear_dense_out + linear_sparse_out  # first_order_output shape: (batch_size, 1)


        # 第二部分：FM部分
        embeddings = tf.stack([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.second_order_sparse_emb)], axis=1)
        # embeddings shape: (batch_size, num_sparse_features, emb_size)

        summed = tf.reduce_sum(embeddings, axis=1)  # summed shape: (batch_size, emb_size)
        squared_sum = tf.square(summed)  # squared_sum shape: (batch_size, emb_size)
        squared = tf.reduce_sum(tf.square(embeddings), axis=1)  # squared shape: (batch_size, emb_size)
        second_order = 0.5 * tf.reduce_sum(squared_sum - squared, axis=1,
                                           keepdims=True)  # second_order shape: (batch_size, 1)


        # 第三部分：DNN部分
        pooled_sequence_embeddings = []
        for i, feat in enumerate(self.sequence_feats):
            feat_name = feat['feat']
            seq_emb = self.sequence_emb[feat_name](
                sequence_inputs[feat_name])  # seq_emb shape: (batch_size, seq_len, emb_size)
            pooled = tf.reduce_mean(seq_emb, axis=1, keepdims=True)  # pooled shape: (batch_size, 1, emb_size)
            pooled_sequence_embeddings.append(pooled)

        pooled_sequence_embeddings = tf.concat(pooled_sequence_embeddings,axis=1)  # pooled_sequence_embeddings shape: (batch_size, num_sequence_feats, emb_size)
        pooled_sequence_embeddings = tf.reshape(pooled_sequence_embeddings, shape=(-1, len(self.sequence_feats) * self.emb_size))  # pooled_sequence_embeddings shape: (batch_size, num_sequence_feats * emb_size)

        flatten_embeddings = tf.reshape(embeddings, shape=(-1, len(self.sparse_feats) * self.emb_size))  # flatten_embeddings shape: (batch_size, num_sparse_feats * emb_size)

        dnn_input = tf.concat([dense_inputs, flatten_embeddings, pooled_sequence_embeddings], axis=1)
        # dnn_input shape: (batch_size, dense_size + num_sparse_feats * emb_size + num_sequence_feats * emb_size)

        dnn_output = self.dnn(dnn_input, training=training)  # dnn_output shape: (batch_size, 1)

        logits = first_order_output + second_order + dnn_output  # logits shape: (batch_size, 1)

        finish_output = self.finish_output_layer(logits)  # finish_output shape: (batch_size, 1)
        like_output = self.like_output_layer(logits)      # like_output shape: (batch_size, 1)

        return {'finish': finish_output, 'like': like_output}




if __name__ == '__main__':
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    dense_feats = ['I1', 'I2']
    sparse_feats = ['C1', 'C2', 'C3']
    sequence_feats = ['user_history', 'user_history_bak']

    feat_columns = [
        [{'feat': 'I1'}, {'feat': 'I2'}],
        [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}],
        [{'feat': 'user_history', 'feat_num': 20}, {'feat': 'user_history_bak', 'feat_num': 20}]
    ]

    model = DeepFM_Sequence_MTL(feat_columns=feat_columns, emb_size=5)

    user_history = ['movie1|movie2|movie3', 'movie2|movie5', 'movie1|movie3|movie4']
    user_history_bak = ['movie6|movie7', 'movie7|movie8', 'movie6|movie9']

    sequence_inputs = {}
    for feat in sequence_feats:
        texts = eval(feat)  # 转换为变量
        tokenizer = Tokenizer(oov_token='OOV')
        tokenizer.fit_on_texts(texts)
        padded = pad_sequences(tokenizer.texts_to_sequences(texts), padding='post')
        sequence_inputs[feat] = padded


    sparse_input = np.array([[1, 2, 3], [4, 5, 5], [1, 2, 3]])
    dense_input = np.random.random((3, len(dense_feats)))
    print("model sequence_inputs:\n", sequence_inputs)
    print("model sparse_input:\n", sparse_input)
    print("model dense_input:\n", dense_input)


    output = model((sparse_input, dense_input, sequence_inputs), training=False)
    print("model output:\n", output)


# model sequence_inputs:
#  {'user_history': array([[2, 3, 4],
#        [3, 5, 0],
#        [2, 4, 6]]), 'user_history_bak': array([[2, 3],
#        [3, 4],
#        [2, 5]])}
# model sparse_input:
#  [[1 2 3]
#  [4 5 5]
#  [1 2 3]]
# model dense_input:
#  [[0.37100647 0.17859854]
#  [0.85621138 0.16887648]
#  [0.09723737 0.99690205]]
# model output:
#  {'finish': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.50046873],
#        [0.4700782 ],
#        [0.59671247]], dtype=float32)>, 'like': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.5003576 ],
#        [0.47716182],
#        [0.5741702 ]], dtype=float32)>}