# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv1D, Dense, Lambda, Multiply, Reshape, Permute, Concatenate
from models.Cin_Keras import CIN
from models.Attention import Attention

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# 定义XDeepFM + Transform当中的Attention
class XDeepFM_Transform_MTL(Model):
    def __init__(self, feat_columns, emb_size=5, cin_layers=[7,15]):
        super().__init__()
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.seq_feats = feat_columns[2] if len(feat_columns) > 2 else []

        self.emb_size = emb_size

        self.linear_dense = layers.Dense(1)

        self.linear_sparse_embeds = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=1)
            for feat in self.sparse_feats
        ]

        self.fm_sparse_embeds = [
            layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size)
            for feat in self.sparse_feats
        ]

        # Sequence embedding layers
        # 仅当存在序列特征时才构造 seq embedding 层
        if self.seq_feats:
            self.seq_embeds = [layers.Embedding(input_dim=feat['feat_num'], output_dim=emb_size, mask_zero=True) for feat in self.seq_feats]

        # Attention 模块
        self.attention = Attention(num_heads=2, key_dim=5)

        self.dnn = tf.keras.Sequential([
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])

        # CIN 模块
        # input_dim是输入特征的数量
        self.cin = CIN(input_dim=len(self.sparse_feats), embedding_dim=emb_size, layer_dims=cin_layers)

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
        linear_sparse_out = tf.concat([emb(sparse_inputs[:, i]) for i, emb in enumerate(self.linear_sparse_embeds)],axis=1)
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
        sparse_embeds = tf.stack([emb(sparse_inputs[:, i])   for i, emb in enumerate(self.fm_sparse_embeds)],axis=1)  # shape: (batch_size, num_sparse_fields, emb_dim)
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
        if self.seq_feats:
            seq_embeds = []
            for i, (seq_input, seq_layer) in enumerate(zip(seq_inputs, self.seq_embeds)):
                seq_emb = seq_layer(seq_input)  # (batch_size, seq_len, emb_dim)
                pooled = tf.reduce_mean(seq_emb, axis=1, keepdims=True)  # (batch_size, 1, emb_dim)  从这可以看出边长序列的字段数据最终整体也是当成一个字段处理
                seq_embeds.append(pooled)
            # 拼接所有嵌入特征
            seq_embeds_concat = tf.concat(seq_embeds, axis=1)  # (batch_size, num_seq_fields, emb_dim)
            # sparse和seq_sparse进行拼接
            fm_input = tf.concat([sparse_embeds, seq_embeds_concat], axis=1)  # (batch, num_sparse_fields + num_seq_fields, emb_dim)
        else:
            fm_input = sparse_embeds

        # 这一步对每个样本的所有 sparse embedding 向量在特征维度上求和：对于某个样本，就是把它的三个特征的 embedding 向量相加，变成一个总的表示
        # (2, 3, 5) → [2, 5]
        summed = tf.reduce_sum(fm_input, axis=1)
        # (2, 5) → [2, 5]
        squared_sum = tf.square(summed)
        # 先对每个 embedding 向量做逐元素平方，然后再对所有 sparse 特征做求和 (2, 3, 5) → [2, 5]
        squared = tf.reduce_sum(tf.square(fm_input), axis=1)
        # [2, 1]，表示每个样本的二阶交叉值（标量）
        second_order = 0.5 * tf.reduce_sum(squared_sum - squared, axis=1, keepdims=True)
        # print(second_order)
        # [[0.00537243]
        #  [0.00075581]]

        # 第三部分：CIN部分
        # CIN部分 输入是离散变量embeddings之后的结果(2,3,5)  其实可以直接cin_input = sparse_embeds
        cin_input = tf.reshape(sparse_embeds, shape=(-1, len(self.sparse_feats), self.emb_size))
        # print(cin_input)
        # print(cin_input.shape)
        # [[[0.04140563  0.03176231  0.01356237 - 0.04648466  0.04604844]
        #   [-0.04380695  0.0055673 - 0.04512671 - 0.00325407  0.04986053]
        #  [-0.04896227 - 0.02058487  0.04993394  0.02401217 - 0.01342272]]
        #
        # [[0.04140563  0.03176231  0.01356237 - 0.04648466  0.04604844]
        #  [-0.03652285  0.00218137  0.01824282 - 0.03070251  0.02715142]
        # [-0.0392504 - 0.04152218
        # 0.00013808
        # 0.04893823 - 0.00362258]]
        #
        # [[0.00019486  0.02938681 - 0.0446441   0.0115645   0.03913646]
        #  [-0.0091401 - 0.04220575 - 0.00148644 - 0.03097728 - 0.04496309]
        #  [0.01833469  0.01332567 - 0.04477177  0.03069586 - 0.00196253]]], shape = (3, 3, 5), dtype = float32)
        # (3, 3, 5)
        cin_output = self.cin(cin_input)

        # 第四部分：DNN 部分
        sparse_flatten_embeddings = tf.reshape(sparse_embeds, shape=(-1, len(self.sparse_feats) * self.emb_size))
        # print(flatten_embeddings)
        # Tensor("Reshape:0", shape=(2, 15), dtype=float32)
        # tf.Tensor(
        # [[-0.03405142  0.01116457 -0.00488006  0.03367222 -0.04788997  0.0262876 0.04647902 -0.01162871  0.03328068  0.04433748  0.02085209  0.01660527
        #   -0.02046416  0.00683039  0.04853446]
        #  [-0.03806484  0.0479795   0.0132894  -0.03121579 -0.0166074   0.00733398
        #    0.00708617 -0.00899755  0.02732437 -0.00605234 -0.02896208  0.02931662
        #    0.0044607   0.03854013  0.04758653]], shape=(2, 15), dtype=float32)
        if self.seq_feats:
            seq_flat = tf.reshape(seq_embeds_concat, shape=(-1, seq_embeds_concat.shape[1] * self.emb_size)) # shape: (batch_size, num_fields * embedding_dim)
            dnn_input = tf.concat([dense_inputs, sparse_flatten_embeddings, seq_flat], axis=1)    # shape=(2, 15) + shape=(2, 2) + shape=(2, X)  => shape=(2, 17)
        else:
            dnn_input = tf.concat([dense_inputs, sparse_flatten_embeddings], axis=1)
        dnn_output = self.dnn(dnn_input, training=training)

        # 第五部分：Transform_Attention部分
        # Attention特征提取部分
        attention_out = self.attention(fm_input)  # shape=(2, 3, 5)
        pooled_attention_out = tf.keras.layers.GlobalAveragePooling1D()(attention_out)  # shape=(2, 5)

        # 合成最终输出
        logits = first_order_output + second_order + cin_output + dnn_output + pooled_attention_out

        # 分支输出
        finish_output = self.finish_output_layer(logits)
        like_output   = self.like_output_layer(logits)

        return {'finish': finish_output, 'like': like_output}


if __name__ == '__main__':
    use_sequence = False

    if not use_sequence:
        # 1. 不使用序列特征
        dense_feats = ['I1', 'I2']
        sparse_feats = ['C1', 'C2', 'C3']
        feat_columns = [
            [{'feat': 'I1'}, {'feat': 'I2'}],
            [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}]
        ]

        model = XDeepFM_Transform_MTL(feat_columns=feat_columns, emb_size=5)
        sparse_input = np.array([[1, 2, 3], [4, 5, 5], [1, 2, 3]])
        dense_input = np.random.random((3, len(dense_feats)))
        output = model((sparse_input, dense_input), training=False)

        print("Dense 输入:")
        print(dense_input)
        print("Sparse 输入:")
        print(sparse_input)
        print("\n模型输出:")
        print(output)
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

        model = XDeepFM_Transform_MTL(feat_columns=feat_columns, emb_size=5)
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



# ----------attention_out----------
# Tensor("attention_1/multi_head_attention_1/attention_output_1/add:0", shape=(3, 3, 5), dtype=float32)
# ----------pooled_attention----------
# Tensor("global_average_pooling1d_1/Mean:0", shape=(3, 5), dtype=float32)
# ----------attention_out----------
# tf.Tensor(
# [[[ 0.0073271  -0.01063516 -0.00380943 -0.00636701  0.00220868]
#   [ 0.00733004 -0.01063512 -0.0038076  -0.00636793  0.00221005]
#   [ 0.00732735 -0.01063572 -0.0038063  -0.00636909  0.0022065 ]]
#
#  [[ 0.01502962  0.00565303  0.0088194  -0.00126621  0.01489555]
#   [ 0.0150284   0.00564757  0.00882035 -0.00127211  0.01489167]
#   [ 0.01502859  0.00565184  0.00881962 -0.00126768  0.01489415]]
#
#  [[ 0.00253227 -0.00208628 -0.00672294  0.00228702  0.00014438]
#   [ 0.00252936 -0.00208656 -0.00672245  0.0022867   0.00013971]
#   [ 0.00252939 -0.00208626 -0.00672186  0.00228659  0.0001389 ]]], shape=(3, 3, 5), dtype=float32)
# ----------pooled_attention----------
# tf.Tensor(
# [[ 0.00732816 -0.01063533 -0.00380777 -0.00636801  0.00220841]
#  [ 0.01502887  0.00565081  0.00881979 -0.00126866  0.01489379]
#  [ 0.00253034 -0.00208637 -0.00672241  0.00228677  0.000141  ]], shape=(3, 5), dtype=float32)
# Dense 输入:
# [[0.0710113  0.7410916 ]
#  [0.3367871  0.0239917 ]
#  [0.9464096  0.84482574]]
# Sparse 输入:
# [[2 1 0]
#  [0 4 5]
#  [2 0 0]]
#
# 模型输出:
# {'finish': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.28823373],
#        [0.45229355],
#        [0.17946997]], dtype=float32)>, 'like': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.2887841 ],
#        [0.45243418],
#        [0.1801348 ]], dtype=float32)>}
# Model: "deep_fm_x_deep_fm_mtl"
# ┌─────────────────────────────────┬────────────────────────┬───────────────┐
# │ Layer (type)                    │ Output Shape           │       Param # │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ dense (Dense)                   │ (3, 1)                 │             3 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding (Embedding)           │ (3, 1)                 │            10 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_1 (Embedding)         │ (3, 1)                 │             8 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_2 (Embedding)         │ (3, 1)                 │             6 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_3 (Embedding)         │ (3, 5)                 │            50 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_4 (Embedding)         │ (3, 5)                 │            40 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ embedding_5 (Embedding)         │ (3, 5)                 │            30 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ attention (Attention)           │ ?                      │           235 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ sequential (Sequential)         │ (3, 1)                 │        85,201 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ functional (Functional)         │ (None, 1)              │           423 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ finish (Dense)                  │ (3, 1)                 │             2 │
# ├─────────────────────────────────┼────────────────────────┼───────────────┤
# │ like (Dense)                    │ (3, 1)                 │             2 │
# └─────────────────────────────────┴────────────────────────┴───────────────┘
#  Total params: 86,010 (335.98 KB)
#  Trainable params: 86,010 (335.98 KB)
#  Non-trainable params: 0 (0.00 B)