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


class DeepFM_MTL(Model):
    def __init__(self, feat_columns, emb_size=5):
        """
        初始化多任务 DeepFM 模型。

        参数:
            feat_columns: list[list[dict]]
                按照特征类型组织的特征配置，包括：
                [0] dense_feats: 连续特征，如 [{'feat': 'time'}, {'feat': 'duration_time'}]
                [1] sparse_feats: 离散特征，需指定 'feat_num'，如 [{'feat': 'uid', 'feat_num': 289}, ...]
                [2] seq_feats: 变长离散序列特征（如标题、标签），需指定 'feat_num'
                [3] history_seq_feats: 历史行为序列特征，需指定 'feat_emb_source' 表示使用哪个稀疏特征的 embedding
            emb_size: int
                FM 和序列特征的嵌入维度大小。
        """
        super().__init__()
        """
        feat_column数据如下:1、连续特征  2、离散特征   3、变长离散  4、历史变长离散
        [{'feat': 'time'}, {'feat': 'duration_time'}]
        [{'feat': 'uid', 'feat_num': 289}, {'feat': 'user_city', 'feat_num': 129}, {'feat': 'item_id', 'feat_num': 291}, {'feat': 'author_id', 'feat_num': 289}, {'feat': 'item_city', 'feat_num': 136}, {'feat': 'channel', 'feat_num': 4}, {'feat': 'music_id', 'feat_num': 115}, {'feat': 'device', 'feat_num': 289}]
        [{'feat': 'actors', 'feat_num': 10}, {'feat': 'genres', 'feat_num': 13}]
        [{'feat': 'history_item_ids', 'feat_emb_source': 'item_id'}, {'feat': 'history_citys', 'feat_emb_source': 'item_city'}]
        """


        # 特征字段划分
        self.dense_feats  = feat_columns[0]  # 连续特征
        self.sparse_feats = feat_columns[1]  # 离散特征
        self.seq_feats = feat_columns[2] if len(feat_columns) > 2 else []  # 可选：变长序列特征
        self.history_seq_feats = feat_columns[3] if len(feat_columns) > 3 else []  # 可选：历史行为序列特征   [{'feat': 'history_item_ids', 'feat_emb_source': 'item_id'}, {'feat': 'history_citys', 'feat_emb_source': 'item_city'}]

        self.emb_size = emb_size

        # ---------- Wide 部分（Linear） ----------
        self.linear_dense = layers.Dense(1)  # 连续特征线性变换
        self.linear_sparse_embed_dict = {
            feat['feat']: layers.Embedding(input_dim=feat['feat_num'], output_dim=1, name=f"{feat['feat']}_linear_embedding")
            for feat in self.sparse_feats
        }

        # ---------- FM 部分 ----------
        # 离散特征的 Embedding 用于 FM 二阶交叉项
        self.fm_sparse_embed_dict = {
            feat['feat']: layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size,
                                           name=f"{feat['feat']}_fm_embedding")
            for feat in self.sparse_feats
        }

        # ---------- 变长序列特征（如标题、标签） ----------
        if self.seq_feats:
            self.seq_embed_dict = {
                feat['feat']: layers.Embedding(
                    input_dim=feat['feat_num'],
                    output_dim=emb_size,
                    mask_zero=True,
                    name=f"{feat['feat']}_seq_embedding"
                )
                for feat in self.seq_feats
            }

        # ---------- 历史行为序列特征 ----------
        # [{'feat': 'history_item_ids', 'feat_emb_source': 'item_id'}, {'feat': 'history_citys', 'feat_emb_source': 'item_city'}]
        # 每个历史序列特征共享其对应的离散特征 embedding（通过 feat_emb_source 指定）
        if self.history_seq_feats:
            self.history_seq_embed_dict = {
                feat['feat']: self.fm_sparse_embed_dict.get(feat['feat_emb_source'])
                for feat in self.history_seq_feats
            }

        # ---------- DNN 部分 ----------
        self.dnn = tf.keras.Sequential([
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])

        # ---------- 多任务输出 ----------
        # 分别为 finish 和 like 任务构建单独的输出层
        self.finish_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='like')


    def call(self, inputs, training=False):
        sparse_inputs      = inputs[0]   # shape: (batch_size, num_sparse)
        dense_inputs       = inputs[1]   # shape: (batch_size, num_dense)

        # 按照数量来切
        seq_inputs = inputs[2: 2 + len(self.seq_feats)]   # list of tensors, each shape each shape each shape: (batch_size, max_seq_len)
        history_seq_inputs = inputs[2 + len(self.seq_feats): 2 + len(self.seq_feats) + len(self.history_seq_feats)]  # list of tensors, each shape each shape each shape: (batch_size, max_seq_len)
        # Dense 输入:
        # [[0.211972   0.3256514 ]
        #  [0.58325326 0.5058359 ]]
        # Sparse 输入:
        # [[5 4 1]
        #  [3 2 3]]
        # seq_inputs 输入:
        # [array([[2, 3, 4],  array([[2, 3],
        #         [3, 5, 0],         [3, 4],
        #         [2, 4, 6]]),       [2, 5]])]
        # history_seq_feats 输入:
        # [array([[2, 3, 4],  array([[2, 3],
        #         [3, 5, 0],         [3, 4],
        #         [2, 4, 6]]),       [2, 5]])]

        # for i, feat in enumerate(self.sparse_feats):
        #     print(i,feat)
        # """
        # 0 {'feat': 'C1', 'feat_num': 10}
        # 1 {'feat': 'C2', 'feat_num': 8}
        # 2 {'feat': 'C3', 'feat_num': 6}
        # """

        # 第一部分：线性部分
        linear_dense_out  = self.linear_dense(dense_inputs)
        # linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.linear_sparse_embeds)],axis=1)
        linear_sparse_out = tf.concat([
            self.linear_sparse_embed_dict[feat['feat']](sparse_inputs[:, i])  for i, feat in enumerate(self.sparse_feats)
        ], axis=1)   # (batch_size, num_sparse, 1)
        # print("--------------1-----------")
        # print(linear_sparse_out)
        # [[ 0.01539519 -0.04832922 -0.02953873]
        #  [ 0.02253106  0.00941402  0.04219601]]

        linear_sparse_out = tf.reduce_sum(linear_sparse_out, axis=1, keepdims=True)  # (batch_size, 1)
        first_order_output = linear_dense_out + linear_sparse_out  # (batch_size, 1)
        # print(first_order_output)
        # [[0.22632796]
        #  [0.6256093 ]]

        # 第二部分：FM部分 (Second Order)
        # sparse_embeds = tf.stack([emb(sparse_inputs[:, i])   for i, emb in enumerate(self.fm_sparse_embeds)],axis=1)  # shape: (batch_size, num_sparse_fields, emb_dim)
        sparse_embeds = tf.stack([self.fm_sparse_embed_dict[feat['feat']](sparse_inputs[:, i]) for i, feat in enumerate(self.sparse_feats)], axis=1)
        # print(sparse_embeds) shape: (batch_size=2, field_num=3, embedding_dim=5)
        # Tensor("stack:0", shape=(2, 3, 5), dtype=float32)
        # tf.Tensor(
        # [[[-0.0292243   0.03134212 -0.00664638  0.0308771   0.03662998]
        #   [-0.02252715  0.00618609 -0.0408314   0.0155008   0.00702292]
        #   [ 0.02527222  0.02442351  0.01027572  0.04815536  0.01610643]]

        #  [[ 0.01005882 -0.03315624 -0.0195043   0.01774564  0.03821408]
        #   [-0.03824542  0.00229248  0.00047214  0.0488669  -0.04776417]
        #   [-0.01696395 -0.00136379  0.04921383  0.04019973 -0.00026955]]], shape=(2, 3, 5), dtype=float32)

        # ---------- 序列特征部分 ----------
        # ---------- 变长序列特征（如标题、标签） ----------
        if self.seq_feats:
            seq_embeds = []
            for i, feat in enumerate(self.seq_feats):
                seq_emb = self.seq_embed_dict[feat['feat']](seq_inputs[i])   # (batch_size, seq_len, emb_dim)
                pooled = tf.reduce_mean(seq_emb, axis=1, keepdims=True)      # (batch_size, 1, emb_dim)  从这可以看出变长序列的字段数据最终整体也是当成一个字段处理
                seq_embeds.append(pooled)                                    # 变长序列特征 embedding 池化后拼接
            # 拼接所有嵌入特征
            seq_embeds_concat = tf.concat(seq_embeds, axis=1)                # (batch_size, num_seq_fields, emb_dim)

        # ---------- 历史行为序列特征 ----------
        if self.history_seq_feats:
            history_seq_embeds = []
            for i, feat in enumerate(self.history_seq_feats):
                seq_emb = self.history_seq_embed_dict[feat['feat']](history_seq_inputs[i])   # (batch_size, seq_len, emb_dim)
                pooled = tf.reduce_mean(seq_emb, axis=1, keepdims=True)                      # (batch_size, 1, emb_dim)
                history_seq_embeds.append(pooled)                                            # 历史行为序列 embedding 池化后拼接
            # 拼接所有嵌入特征
            history_seq_embeds_concat = tf.concat(history_seq_embeds, axis=1)                # shape = (batch_size, num_history_seq_fields, emb_dim)

        # 拼接所有embedding用于FM二阶交叉计算
        fm_input_parts = [sparse_embeds]
        if self.seq_feats:
            fm_input_parts.append(seq_embeds_concat)
        if self.history_seq_feats:
            fm_input_parts.append(history_seq_embeds_concat)

        fm_input = tf.concat(fm_input_parts, axis=1)  # (batch_size, total_fields, emb_dim)

        # 这一步对每个样本的所有 sparse embedding 向量在特征维度上求和：对于某个样本，就是把它的三个特征的 embedding 向量相加，变成一个总的表示
        # (2, 3, 5) → [2, 5]
        summed = tf.reduce_sum(fm_input, axis=1)    # (batch_size, emb_size)
        # (2, 5) → [2, 5]
        squared_sum = tf.square(summed)             # (batch_size, emb_size)
        # 先对每个 embedding 向量做逐元素平方，然后再对所有 sparse 特征做求和 (2, 3, 5) → [2, 5]
        squared = tf.reduce_sum(tf.square(fm_input), axis=1)    # (batch_size, emb_size)
        # [2, 1]，表示每个样本的二阶交叉值（标量）
        second_order = 0.5 * tf.reduce_sum(squared_sum - squared, axis=1, keepdims=True)  # (batch_size, 1)
        # print(second_order)
        # [[0.00537243]
        #  [0.00075581]]

        # 第三部分：DNN 部分
        sparse_flatten_embeddings = tf.reshape(sparse_embeds, shape=(-1, len(self.sparse_feats) * self.emb_size))
        # print(flatten_embeddings)
        # Tensor("Reshape:0", shape=(2, 15), dtype=float32)
        # tf.Tensor(
        # [[-0.03405142  0.01116457 -0.00488006  0.03367222 -0.04788997  0.0262876 0.04647902 -0.01162871  0.03328068  0.04433748  0.02085209  0.01660527
        #   -0.02046416  0.00683039  0.04853446]
        #  [-0.03806484  0.0479795   0.0132894  -0.03121579 -0.0166074   0.00733398
        #    0.00708617 -0.00899755  0.02732437 -0.00605234 -0.02896208  0.02931662
        #    0.0044607   0.03854013  0.04758653]], shape=(2, 15), dtype=float32)
        # 初始化 DNN 输入
        dnn_input_parts = [dense_inputs, sparse_flatten_embeddings]  # [shape=(batch_size, dense_dim), shape=(batch_size, num_sparse_fields * emb_dim)]
        if self.seq_feats:
            # shape: (batch_size, num_seq_fields, emb_dim) → (batch_size, num_seq_fields * emb_dim)
            seq_flat = tf.reshape(seq_embeds_concat, shape=(-1, seq_embeds_concat.shape[1] * self.emb_size))
            dnn_input_parts.append(seq_flat)
        if self.history_seq_feats:
            # shape: (batch_size, num_history_seq_fields, emb_dim) → (batch_size, num_history_seq_fields * emb_dim)
            history_seq_flat = tf.reshape(history_seq_embeds_concat, shape=(-1, history_seq_embeds_concat.shape[1] * self.emb_size))
            dnn_input_parts.append(history_seq_flat)

        # 拼接所有部分作为 DNN 输入
        dnn_input = tf.concat(dnn_input_parts, axis=1)  # 最终 shape: (batch_size, dense_dim + sparse_dim + seq_dim + history_seq_dim)

        # DNN forward
        dnn_output = self.dnn(dnn_input, training=training)

        # 合成最终输出
        logits = first_order_output + second_order + dnn_output

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

        model = DeepFM_MTL(feat_columns=feat_columns, emb_size=5)
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
            [{'feat': 'S1', 'feat_num': 10}, {'feat': 'S2', 'feat_num': 20}],
            [{'feat': 'History_H1', 'feat_emb_source': 'C1'}, {'feat': 'History_H2', 'feat_emb_source': 'C2'}]
        ]

        model = DeepFM_MTL(feat_columns=feat_columns, emb_size=5)

        # 稀疏特征 (batch_size=3)
        sparse_input = np.array([[1, 2, 3], [4, 5, 5], [1, 2, 3]])
        # 稠密特征
        dense_input = np.random.random((3, len(dense_feats)))

        # 变长序列特征
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

        # ✅ 增加历史序列特征 History_H1 和 History_H2
        # 模拟用户过去点击过的C1、C2的历史行为序列（用稀疏特征ID列表表达）
        # 比如 C1 取值范围是 0-9（因为 feat_num=10），所以这里历史行为可以是其中的一些值

        history_seq_inputs = [
            np.array([[1, 2, 0], [3, 0, 0], [4, 5, 6]]),  # History_H1 (from C1)
            np.array([[1, 3, 4], [2, 2, 0], [0, 0, 0]])  # History_H2 (from C2)
        ]

        # ✅ 合并所有输入并传入模型
        sequence_tensors = []
        model_inputs = (sparse_input, dense_input, *seq_input_list, *history_seq_inputs)
        output = model(model_inputs, training=False)

        print("====== Dense 输入 ======",feat_columns[0])
        print(dense_input)
        print()

        print("====== Sparse 输入 ======",feat_columns[1])
        print(sparse_input)
        print()

        print("====== 离散序列特征输入（如标题、标签等） ======")
        for i,feat in enumerate(sequence_feats):
            print(f"{feat} 输入:",feat_columns[2][i])
            print(sequence_inputs[feat])
            print()

        print("====== 历史行为序列输入（History） ======")
        history_feats = ['History_H1', 'History_H2']
        for feat, hist_input in zip(history_feats, history_seq_inputs):
            print(f"{feat} 输入:")
            print(hist_input)
            print()

        print("====== 模型输出 ======")
        print(output)

        model.summary()



# Dense 输入:
# [[0.42735437 0.1205228 ]
#  [0.3352796  0.72378077]
#  [0.35812327 0.34702339]]
# Sparse 输入:
# [[1 2 3]
#  [4 5 5]
#  [1 2 3]]
# S1 输入:
# [[2 3 4]
#  [3 5 0]
#  [2 4 6]]
# S2 输入:
# [[2 3]
#  [3 4]
#  [2 5]]
#
# 模型输出:
# {'finish': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.5794624 ],
#        [0.46379182],
#        [0.52200294]], dtype=float32)>, 'like': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.6530806],
#        [0.4289063],
#        [0.5433398]], dtype=float32)>}
# Model: "deep_fm_mtl"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                multiple                  3
# _________________________________________________________________
# embedding (Embedding)        multiple                  10
# _________________________________________________________________
# embedding_1 (Embedding)      multiple                  8
# _________________________________________________________________
# embedding_2 (Embedding)      multiple                  6
# _________________________________________________________________
# embedding_3 (Embedding)      multiple                  50
# _________________________________________________________________
# embedding_4 (Embedding)      multiple                  40
# _________________________________________________________________
# embedding_5 (Embedding)      multiple                  30
# _________________________________________________________________
# embedding_6 (Embedding)      multiple                  50
# _________________________________________________________________
# embedding_7 (Embedding)      multiple                  100
# _________________________________________________________________
# sequential (Sequential)      (3, 1)                    86201
# _________________________________________________________________
# finish (Dense)               multiple                  2
# _________________________________________________________________
# like (Dense)                 multiple                  2
# =================================================================
# Total params: 86,502
# Trainable params: 86,502
# Non-trainable params: 0
# _________________________________________________________________