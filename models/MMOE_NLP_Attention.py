# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media

# ✅ 整体结构：
# ✅ 使用了 多个 expert 网络 来提取共享特征。
# ✅ 对每个任务都使用独立的 gate 网络 输出 softmax 权重。
# ✅ 用 gate 权重对 expert 输出加权求和，作为每个任务的输入。
# ✅ 支持 稠密特征 + 离散特征 + 变长序列特征 的融合建模。
# ✅ 每个任务都使用独立的输出层 Dense(1, activation='sigmoid')。

from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
from config.data_config import *
from models.Attention import Attention

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class MMOE_NLP_Attention(Model):
    def __init__(self, feat_columns, emb_size=5, num_experts=4):
        super().__init__()
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.seq_feats = feat_columns[2] if len(feat_columns) > 2 else []
        
        self.emb_size = emb_size
        self.num_experts = num_experts

        # Embedding for sparse features
        self.sparse_embs = [layers.Embedding(input_dim=feat['feat_num'], output_dim=self.emb_size) for feat in self.sparse_feats]

        # Sequence embedding layers
        # 仅当存在序列特征时才构造 seq embedding 层
        if self.seq_feats:
            self.seq_embeds = [layers.Embedding(input_dim=feat['feat_num'], output_dim=self.emb_size, mask_zero=True) for feat in self.seq_feats]

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
        self.gate_like   = layers.Dense(self.num_experts, activation='softmax')

        # Output layers for each task  ====> 用来计算每个塔的输出，对应多输出预测
        self.finish_output = layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output   = layers.Dense(1, activation='sigmoid', name='like')

        # Attention 模块
        self.attention = Attention(num_heads=2, key_dim=5)

        # 归一化模型
        self.norm = tf.keras.layers.LayerNormalization()

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
        
        
        # 1️⃣ 稀疏特征 embedding
        sparse_embeds = tf.stack([emb(sparse_inputs[:, i])   for i, emb in enumerate(self.sparse_embs)],axis=1)  # shape: (batch_size, num_sparse_fields, emb_dim)
        # print(sparse_embeds) shape: (batch_size=2, field_num=3, embedding_dim=5)
        # Tensor("stack:0", shape=(2, 3, 5), dtype=float32)
        # tf.Tensor(
        # [[[-0.0292243   0.03134212 -0.00664638  0.0308771   0.03662998]
        #   [-0.02252715  0.00618609 -0.0408314   0.0155008   0.00702292]
        #   [ 0.02527222  0.02442351  0.01027572  0.04815536  0.01610643]]

        #  [[ 0.01005882 -0.03315624 -0.0195043   0.01774564  0.03821408]
        #   [-0.03824542  0.00229248  0.00047214  0.0488669  -0.04776417]
        #   [-0.01696395 -0.00136379  0.04921383  0.04019973 -0.00026955]]], shape=(2, 3, 5), dtype=float32)

        sparse_flatten_embeddings = tf.reshape(sparse_embeds, shape=(-1, len(self.sparse_feats) * self.emb_size)) # shape: (batch_size, num_sparse_fields * emb_dim)
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
                pooled = tf.reduce_mean(seq_emb, axis=1, keepdims=True)  # (batch_size, 1, emb_dim)  从这可以看出边长序列的字段数据最终整体也是当成一个字段处理
                seq_embeds.append(pooled)
            # 拼接所有嵌入特征
            seq_embeds_concat = tf.concat(seq_embeds, axis=1)  # (batch_size, num_seq_fields, emb_dim)
            seq_flatten_embeddings = tf.reshape(seq_embeds_concat, shape=(-1, seq_embeds_concat.shape[1] * self.emb_size)) # shape: (batch_size, num_seq_fields * emb_dim)

            # sparse、seq、dense进行拼接
            mmoe_input = tf.concat([dense_inputs,sparse_flatten_embeddings, seq_flatten_embeddings], axis=1)    # shape=(2, X) + shape=(2, Y) + shape=(2, Z)  => shape=(2, 17))
        else:
            # sparse、dense进行拼接
            mmoe_input = tf.concat([dense_inputs,sparse_flatten_embeddings], axis=1)



        # 2️⃣ 多专家输出
        expert_outputs = tf.stack([expert(mmoe_input, training=training) for expert in self.experts],axis=1)  # shape: shape=(3, 4, 64) (batch_size, num_experts, expert_output_dim)

        # 多专家输出 增加 Transform_Attention特征提取部分-创新部分
        # 自注意力交互（Self-Attention）
        residual = expert_outputs  # 残差连接
        expert_outputs = self.attention(expert_outputs)  # Q=K=V
        # 残差连接 + 层归一化
        expert_outputs = self.norm(expert_outputs + residual)   #  加 LayerNorm / 残差连接 是优化项（不强制）
        """
        归一化的具体过程:
        对形状为 (2, 3, 5) 的张量，LayerNormalization 的计算过程如下：
        对第一个维度（batch）里的每个样本（共2个）
        对第二个维度（序列长度）里的每个时间步（共3个）
        对第三个维度（特征维度）中的5个元素进行均值和方差计算，归一化
        换句话说，LayerNormalization 是对形状 (5,) 的向量做标准化，每个 (2,3) 位置都会独立计算和归一化。

        🎯 结论：
        对于形状为 (2, 3, 5) 的输入：
            LayerNormalization 是对每个序列中的 5 个特征单独做标准化
            ✅ 不是 把一个样本的 3 × 5 = 15 个数拉成一维后统一归一化
            换句话说，它是沿着最后一个维度（feature_dim）做归一化
        
        🔍 细节解释：
        以 shape = (batch_size=2, seq_len=3, feature_dim=5) 为例，LayerNormalization 的处理方式是：
        for i in range(batch_size):      # 对每个样本
            for j in range(seq_len):     # 对每个时间步/序列位置
                x = inputs[i, j, :]      # 一个长度为5的向量
                norm_x = (x - mean(x)) / sqrt(var(x) + eps)
                # 再乘 gamma，加 beta（可训练参数）
        所以，每个 (i, j) 的位置都独立执行标准化操作。
        """

        # 3️⃣ 每个任务的 gate
        gate_finish_weight = tf.expand_dims(self.gate_finish(mmoe_input), axis=-1)     # (batch_size, num_experts, 1)
        gate_like_weight   = tf.expand_dims(self.gate_like(mmoe_input), axis=-1)       # (batch_size, num_experts, 1)

        # 4️⃣ Gate 加权求和（融合多个 expert）
        task_finish_input = tf.reduce_sum(gate_finish_weight * expert_outputs,axis=1)  # (batch_size, num_experts, expert_output_dim) ==> (batch_size, expert_output_dim)
        task_like_input = tf.reduce_sum(gate_like_weight * expert_outputs,axis=1)      # (batch_size, num_experts, expert_output_dim) ==> (batch_size, expert_output_dim)
        """
        expert_outputs.shape = (2, 3, 4)
        [
            [  # 第一个样本：3个 expert 输出向量
                [1.0, 2.0, 3.0, 4.0],  # expert 1
                [5.0, 6.0, 7.0, 8.0],  # expert 2
                [9.0,10.0,11.0,12.0],  # expert 3
            ],
            [  # 第二个样本
                [0.5, 1.0, 1.5, 2.0],
                [2.0, 2.5, 3.0, 3.5],
                [4.0, 4.5, 5.0, 5.5],
            ]
        ]
        
        这个操作表示：对于每个样本，沿着第 1 维（也就是专家数量）对每个位置的值求和   
        第一个样本：对三个 expert 的向量按列求和：[1.0 + 5.0 + 9.0, 2.0 + 6.0 + 10.0, 3.0 + 7.0 + 11.0, 4.0 + 8.0 + 12.0] = [15.0, 18.0, 21.0, 24.0]        
        第二个样本：                            [0.5 + 2.0 + 4.0, 1.0 + 2.5 + 4.5, 1.5 + 3.0 + 5.0, 2.0 + 3.5 + 5.5]    = [6.5, 8.0, 9.5, 11.0]        

        输出是 shape (2, 4) 的张量：
        [
            [15.0, 18.0, 21.0, 24.0],
            [ 6.5,  8.0,  9.5, 11.0]
        ]
        reduce_sum(axis=1) 就是 把每个样本的多个 expert 输出向量逐元素相加，最终每个样本只留下一个长度为 expert_output_dim 的融合向量。
        """


        # 5️⃣ 输出预测
        finish_logit = task_finish_input
        like_logit = task_like_input

        return {
            'finish': self.finish_output(finish_logit),
            'like': self.like_output(like_logit)
        }



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

        model = MMOE_NLP_Attention(feat_columns=feat_columns, emb_size=5)
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

        model = MMOE_NLP_Attention(feat_columns=feat_columns, emb_size=5)
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

# Dense 输入:
# [[0.113212   0.63990683]
#  [0.83283576 0.00455242]
#  [0.40675872 0.30803841]]
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
# array([[0.5065163 ],
#        [0.51338285],
#        [0.50972116]], dtype=float32)>, 'like': <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
# array([[0.47611362],
#        [0.47083804],
#        [0.48120457]], dtype=float32)>}
# Model: "MMOE_NLP_Attention"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        multiple                  50
# _________________________________________________________________
# embedding_1 (Embedding)      multiple                  40
# _________________________________________________________________
# embedding_2 (Embedding)      multiple                  30
# _________________________________________________________________
# embedding_3 (Embedding)      multiple                  50
# _________________________________________________________________
# embedding_4 (Embedding)      multiple                  100
# _________________________________________________________________
# sequential (Sequential)      (3, 64)                   11840
# _________________________________________________________________
# sequential_1 (Sequential)    (3, 64)                   11840
# _________________________________________________________________
# sequential_2 (Sequential)    (3, 64)                   11840
# _________________________________________________________________
# sequential_3 (Sequential)    (3, 64)                   11840
# _________________________________________________________________
# dense_8 (Dense)              multiple                  112
# _________________________________________________________________
# dense_9 (Dense)              multiple                  112
# _________________________________________________________________
# finish (Dense)               multiple                  65
# _________________________________________________________________
# like (Dense)                 multiple                  65
# =================================================================
# Total params: 47,984
# Trainable params: 47,984
# Non-trainable params: 0
# _________________________________________________________________