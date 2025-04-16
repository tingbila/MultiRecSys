# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import tensorflow as tf
from tensorflow.keras import layers, Model


class MMOE(Model):
    def __init__(self, feat_columns, embed_dim, num_experts=4):
        super().__init__()
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.dense_size = len(self.dense_feats)
        self.emb_size = embed_dim
        self.num_experts = num_experts

        # Embedding for sparse features
        self.second_order_sparse_emb = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=self.emb_size)
            for feat in self.sparse_feats
        ]

        # Expert networks
        self.experts = [
            tf.keras.Sequential([
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu')
            ]) for _ in range(self.num_experts)
        ]

        # Gate networks for each task   ====> 用来计算每个专家对应的gate的权重
        self.gate_finish = layers.Dense(self.num_experts, activation='softmax')
        self.gate_like = layers.Dense(self.num_experts, activation='softmax')

        # Output layers for each task  ====> 用来计算每个塔的输出，对应多输出预测
        self.finish_output = layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output = layers.Dense(1, activation='sigmoid', name='like')

    def call(self, inputs, training=False):
        sparse_inputs, dense_inputs = inputs

        # 1️⃣ 稀疏特征 embedding
        embeddings = tf.stack([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.second_order_sparse_emb)],
                              axis=1)  # shape: (B, num_sparse, emb_dim)

        flatten_emb = tf.reshape(embeddings, shape=(-1, len(self.sparse_feats) * self.emb_size))
        dnn_input = tf.concat([dense_inputs, flatten_emb], axis=1)  # shape: (B, dense + sparse)

        # 2️⃣ 多专家输出
        expert_outputs = tf.stack([expert(dnn_input, training=training) for expert in self.experts],
                                  axis=1)  # shape: (B, num_experts, expert_output_dim)

        # 3️⃣ 每个任务的 gate
        gate_finish_weight = tf.expand_dims(self.gate_finish(dnn_input), axis=-1)  # (B, num_experts, 1)
        gate_like_weight = tf.expand_dims(self.gate_like(dnn_input), axis=-1)  # (B, num_experts, 1)

        # 4️⃣ Gate 加权求和（融合多个 expert）
        task_finish_input = tf.reduce_sum(gate_finish_weight * expert_outputs,
                                          axis=1)  # (B, num_experts, expert_output_dim) ==> (B, expert_output_dim)
        task_like_input = tf.reduce_sum(gate_like_weight * expert_outputs,
                                        axis=1)  # (B, num_experts, expert_output_dim) ==> (B, expert_output_dim)

        # 5️⃣ 输出预测
        finish_logit = task_finish_input
        like_logit = task_like_input

        return {
            'finish': self.finish_output(finish_logit),
            'like': self.like_output(like_logit)
        }



if __name__ == '__main__':
    dense_feats  = ['I1', 'I2']
    sparse_feats = ['C1', 'C2', 'C3']
    feat_columns = [
        [{'feat': 'I1'}, {'feat': 'I2'}],
        [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
    ]

    model = DeepFM_MMOE_MTL(feat_columns=feat_columns, emb_size=5)

    batch_size = 3
    dense_input  = tf.random.uniform(shape=(batch_size, len(dense_feats)), dtype=tf.float32)
    sparse_input = tf.random.uniform(shape=(batch_size, len(sparse_feats)), maxval=6, dtype=tf.int32)

    output = model((sparse_input, dense_input), training=False)
    print("Dense 输入:")
    print(dense_input.numpy())
    print("Sparse 输入:")
    print(sparse_input.numpy())
    print("\n模型输出:")
    print(output)


# Dense 输入:
# [[0.62843096 0.7158071 ]
#  [0.21943963 0.65574443]
#  [0.83801186 0.20903265]]
# Sparse 输入:
# [[4 1 5]
#  [2 0 4]
#  [3 0 3]]
#
# 模型输出:
# {'finish': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.5374543],
#        [0.5324111],
#        [0.5309143]], dtype=float32)>, 'like': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.47835976],
#        [0.48945898],
#        [0.4786319 ]], dtype=float32)>}