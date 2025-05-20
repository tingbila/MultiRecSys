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
# ✅ 相比之前的MMOE，增加了2点:增加历史序列数据、增加历史序列数据的加权兴趣


from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
from config.data_config import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class MMOE_IN_DIEN(Model):
    def __init__(self, feat_columns, emb_size=5, num_experts=4):
        super().__init__()
        self.dense_feats, self.sparse_feats = feat_columns[0], feat_columns[1]
        self.seq_feats = feat_columns[2] if len(feat_columns) > 2 else []
        self.history_seq_feats = feat_columns[3] if len(
            feat_columns) > 3 else []  # 可选：历史行为序列特征   [{'feat': 'history_item_ids', 'target_emb_column': 'item_id'}, {'feat': 'history_citys', 'target_emb_column': 'item_city'}]

        self.emb_size = emb_size
        self.num_experts = num_experts

        # Embedding for sparse features
        self.fm_sparse_embed_dict = {
            feat['feat']: layers.Embedding(input_dim=feat['feat_num'], output_dim=self.emb_size,
                                           name=f"{feat['feat']}_fm_embedding")
            for feat in self.sparse_feats
        }

        # ---------- 变长序列特征（如标题、标签） ----------
        # 仅当存在序列特征时才构造 seq embedding 层
        if self.seq_feats:
            self.seq_embed_dict = {
                feat['feat']: layers.Embedding(input_dim=feat['feat_num'], output_dim=self.emb_size, mask_zero=True,
                                               name=f"{feat['feat']}_seq_embedding")
                for feat in self.seq_feats
            }

        # ---------- 历史行为序列特征 ----------
        # [{'feat': 'history_item_ids', 'target_emb_column': 'item_id'}, {'feat': 'history_citys', 'target_emb_column': 'item_city'}]
        # 每个历史序列特征共享其对应的离散特征 embedding（通过 target_emb_column 指定）
        if self.history_seq_feats:
            self.history_seq_embed_dict = {
                feat['feat']: self.fm_sparse_embed_dict.get(feat['target_emb_column'])
                for feat in self.history_seq_feats
            }
            # 这里补充各历史序列特征对应的 embedding 层和 GRU 层
            self.history_seq_gru_dict = {
                feat['feat']: tf.keras.layers.GRU(units=emb_size, return_sequences=False, name=f"{feat['feat']}_gru")
                # 所有 GRU 都设置了 return_sequences=False，确保你拿到的是最后一步的兴趣向量（简化版 DIEN 结构）。
                for feat in self.history_seq_feats
            }

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

        # DIN用的注意力网络-DIN专用
        self.din_attention_mlp = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation=None)  # 输出注意力得分，后面用softmax归一化
        ])

        # Output layers for each task  ====> 用来计算每个塔的输出，对应多输出预测
        self.finish_output = layers.Dense(1, activation='sigmoid', name='finish')
        self.like_output = layers.Dense(1, activation='sigmoid', name='like')

    def din_attention(self, query, keys, mask=None):
        """
        这个函数 din_attention(query, keys, mask) 是 DIN（Deep Interest Network）注意力机制的核心部分，用于计算：
        当前目标商品（query）和历史点击商品序列（keys）之间的兴趣相关性加权向量。
        简单说就是：
             用户历史看过 A、B、C，现在给他看 D，我们要判断：
             A 和 D 相关？B 和 D 相关？C 和 D 相关？
            然后给每个历史行为打个权重（attention），加权得到一个 兴趣表达向量（interest summary）。

        query->item_emb->当前目标 item 的 embedding 向量: 目标物品embedding，shape = (batch_size, emb_dim)     (3, 5)
                比如item，下面相当于是给每行对应的item向量都给获取到了
                (3, 5)
                tf.Tensor(
                [[ 0.00409347  0.01375012 -0.00956724  0.04792858  0.04354867]
                 [ 0.03006909 -0.04394181  0.00638409 -0.04737679  0.04043685]
                 [ 0.00409347  0.01375012 -0.00956724  0.04792858  0.04354867]], shape=(3, 5), dtype=float32)

        keys->history_items_emb->历史点击序列，每个 item 是一个 embedding: 历史序列embedding，shape = (batch_size, seq_len, emb_dim)   (3, 3, 5)

        mask->history_items->	指示历史序列中哪些是 padding，0 表示 padding，1 表示有效 : 原始数据掩码处理: 掩码，shape = (batch_size, seq_len)，1表示有效，0表示padding 对应keys真实数据的掩码
                在 DIN 注意力模块中，计算注意力分数后会用这个 mask 来把 padding 的位置打上极小值（-inf），防止这些无效位置对注意力分数和最终兴趣向量产生影响。
                这一步是为了构造注意力机制中使用的 mask，使模型只聚焦于用户真实的历史行为，忽略 padding 部分，避免引入无效信息干扰。
                原始数据输入:                               mask数值:
                tf.Tensor(                                 tf.Tensor(
                [[1 2 0]                                   [[1. 1. 0.]
                 [3 0 0]                                    [1. 0. 0.]
                 [4 5 6]], shape=(3, 3), dtype=int32)       [1. 1. 1.]], shape=(3, 3), dtype=float32)
        返回加权求和后的兴趣向量，shape = (batch_size, 1, emb_dim)  --> 和以前deepfm的处理格式保持一致
        """

        # 1. 将目标 item 的向量复制成一个序列长度的矩阵，准备与历史序列中的每个 item 进行逐个对比。
        query = tf.expand_dims(query, axis=1)  # (3, 5) -> (3, 1, 5)    # (B, E) -> (B, 1, E)
        query = tf.tile(query, [1, tf.shape(keys)[1], 1])  # (batch_size, seq_len, emb_dim)  复制考虑seq_len份   -> (B, S, E)
        """
        (3, 3, 5)
        tf.Tensor(
        [[[-0.00044398 -0.02416375 -0.02272075  0.01461015  0.04648725]
          [-0.00044398 -0.02416375 -0.02272075  0.01461015  0.04648725]
          [-0.00044398 -0.02416375 -0.02272075  0.01461015  0.04648725]]

         [[ 0.03528274  0.02952408 -0.02367508  0.00254655 -0.04886616]
          [ 0.03528274  0.02952408 -0.02367508  0.00254655 -0.04886616]
          [ 0.03528274  0.02952408 -0.02367508  0.00254655 -0.04886616]]

         [[-0.00044398 -0.02416375 -0.02272075  0.01461015  0.04648725]
          [-0.00044398 -0.02416375 -0.02272075  0.01461015  0.04648725]
          [-0.00044398 -0.02416375 -0.02272075  0.01461015  0.04648725]]], shape=(3, 3, 5), dtype=float32)
        """

        # 2. 构建注意力输入特征：query 和 key 的组合特征  这是 DIN 的核心设计！用 [q, k, q-k, q*k] 四个部分建模 query 和 key 之间的复杂关系。
        # shape=(3, 3, 5) 和 shape=(3, 3, 5)进行交互处理
        att_input = tf.concat([query, keys, query - keys, query * keys], axis=-1)  # (batch_size, seq_len, 4*emb_dim)

        # 3. 通过一个小 MLP 得到注意力分数:用一个浅层网络（可学习）得到每个历史行为对当前目标 item 的注意力分数。
        att_scores = self.din_attention_mlp(att_input)  # (batch_size, seq_len, 1)  ====> 其实看到这里其实就是权重了。。
        """
        (3, 3, 1)
        tf.Tensor(
        [[[-0.00737097]
          [-0.00759939]
          [-0.00788507]]

         [[-0.02021017]
          [-0.01353764]
          [-0.01353764]]

         [[-0.01749066]
          [-0.01572918]
          [-0.00743625]]], shape=(3, 3, 1), dtype=float32)
        """

        # 4. mask处理 下面这几个步骤都是mask处理，其实可以跳过这几个步骤。
        att_scores = tf.squeeze(att_scores, axis=-1)  # (batch_size, seq_len)
        """
        [[ 0.01111379  0.0005887  -0.01211756]
         [-0.01127466 -0.01005966 -0.01005966]
         [-0.00443098 -0.00823664  0.00463225]], shape=(3, 3), dtype=float32)
        """
        # 使用 mask 去除 padding 的影响
        # 你不希望模型考虑 padding（空白）的 item，所以需要 把它的注意力分数变成极小值（相当于 -∞），
        # 让 softmax 的时候它的概率变成 接近 0，不会影响结果。
        if mask is not None:
            paddings = tf.ones_like(att_scores) * (-2 ** 32 + 1)
            att_scores = tf.where(mask > 0, att_scores, paddings)
            """
            (3, 3)
            tf.Tensor(
            [[ 1.6117751e-03  1.3761593e-03 -4.2949673e+09]
             [-5.3150370e-03 -4.2949673e+09 -4.2949673e+09]
             [-1.3341270e-03  7.2927773e-03  9.4263274e-03]], shape=(3, 3), dtype=float32)
            """
        # softmax 得到权重  转化为概率分布，代表当前目标 item 与每个历史 item 的相关性权重（总和为 1）
        att_weights = tf.nn.softmax(att_scores, axis=1)  # (batch_size, seq_len)
        """
        att_weights：
        (3, 3)
        tf.Tensor(
        [[0.5011909  0.49880904 0.        ]
         [1.         0.         0.        ]
         [0.33306825 0.3316404  0.3352914 ]], shape=(3, 3), dtype=float32)

        print(tf.expand_dims(att_weights, -1).shape)：
        print(tf.expand_dims(att_weights, -1))：
        (3, 3, 1)
        tf.Tensor(
        [[[0.5008997 ]
          [0.49910033]
          [0.        ]]

         [[1.        ]
          [0.        ]
          [0.        ]]

         [[0.33396128]
          [0.33337814]
          [0.33266056]]], shape=(3, 3, 1), dtype=float32)
        """

        # 5. 加权求和历史序列
        # 不好理解的话就这么理解：考虑只有一个样本  # (1, 3, 1) * (1, 3, 5) => (1, 3, 5) => (1,5)
        # 每个历史 item 的 embedding 被乘上注意力权重
        # 然后在序列维度上求和，得到一个加权后的 “兴趣表示向量”
        # print((tf.expand_dims(att_weights, -1) * keys).shape)  (3, 3, 5)
        output = tf.reduce_sum(tf.expand_dims(att_weights, -1) * keys,
                               axis=1)  # (3, 3, 1) *(3, 3, 5) => (batch_size, seq_len, emb_dim) => (batch_size, emb_dim)

        output = tf.expand_dims(output, axis=1)  # ✅ 添加这行，结果是 (batch_size, 1, emb_dim)
        return output

    def call(self, inputs, training=False):
        sparse_inputs = inputs[0]  # shape: (batch_size, num_sparse)
        dense_inputs = inputs[1]  # shape: (batch_size, num_dense)

        # 按照数量来切
        seq_inputs = inputs[2: 2 + len(
            self.seq_feats)]  # list of tensors, each shape each shape each shape: (batch_size, max_seq_len)
        history_seq_inputs = inputs[2 + len(self.seq_feats): 2 + len(self.seq_feats) + len(
            self.history_seq_feats)]  # list of tensors, each shape each shape each shape: (batch_size, max_seq_len)
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

        # ---------- 1、稀疏特征处理 ----------
        sparse_embeds = tf.stack(
            [self.fm_sparse_embed_dict[feat['feat']](sparse_inputs[:, i]) for i, feat in enumerate(self.sparse_feats)],
            axis=1)
        # ---------- 序列特征部分（1+2） ----------
        # ---------- 2、变长序列特征（如标题、标签） ----------
        if self.seq_feats:
            seq_embeds = []
            for i, feat in enumerate(self.seq_feats):
                seq_emb = self.seq_embed_dict[feat['feat']](seq_inputs[i])  # (batch_size, seq_len, emb_dim)
                pooled = tf.reduce_mean(seq_emb, axis=1,
                                        keepdims=True)  # (batch_size, 1, emb_dim)  从这可以看出变长序列的字段数据最终整体也是当成一个字段处理
                seq_embeds.append(pooled)  # 变长序列特征 embedding 池化后拼接
            # 拼接所有嵌入特征
            seq_embeds_concat = tf.concat(seq_embeds, axis=1)  # (batch_size, num_seq_fields, emb_dim)

        # ---------- 3-1、历史行为序列特征-历史行为序列特征不再采用简单的池化方式，而是引入了基于目标物品的 DIN 注意力机制进行兴趣提取 ----------
        if self.history_seq_feats:
            history_seq_embeds = []
            for i, feat in enumerate(self.history_seq_feats):
                # 获取当前历史序列的 embedding 表示
                history_embeds = self.history_seq_embed_dict[feat['feat']](
                    history_seq_inputs[i])  # (batch_size, seq_len, emb_dim)   (3, 3, 5)

                # 构建序列的有效位置 mask（padding 值为 0 的位置将被 mask 掉）
                mask = tf.cast(tf.not_equal(history_seq_inputs[i], 0), tf.float32)  # (batch_size, seq_len)
                """
                在 DIN 注意力模块中，计算注意力分数后会用这个 mask 来把 padding 的位置打上极小值（-inf），防止这些无效位置对注意力分数和最终兴趣向量产生影响。
                这一步是为了构造注意力机制中使用的 mask，使模型只聚焦于用户真实的历史行为，忽略 padding 部分，避免引入无效信息干扰。
                原始数据输入:                               mask数值:
                tf.Tensor(                                 tf.Tensor(
                [[1 2 0]                                   [[1. 1. 0.]
                 [3 0 0]                                    [1. 0. 0.]
                 [4 5 6]], shape=(3, 3), dtype=int32)       [1. 1. 1.]], shape=(3, 3), dtype=float32)
                """
                # 提取与该历史序列对应的目标 item embedding（用于注意力对齐） target_emb_column 表示目标 item 特征名，target_item_index 表示其在 sparse_inputs 中的列索引
                target_item_embed = self.fm_sparse_embed_dict[feat['target_emb_column']](
                    sparse_inputs[:, feat['target_item_index']])  # (batch_size, emb_dim)
                """
                比如item，下面相当于是给每行对应的item向量都给获取到了
                (3, 5)
                tf.Tensor(
                [[ 0.00409347  0.01375012 -0.00956724  0.04792858  0.04354867]
                 [ 0.03006909 -0.04394181  0.00638409 -0.04737679  0.04043685]
                 [ 0.00409347  0.01375012 -0.00956724  0.04792858  0.04354867]], shape=(3, 5), dtype=float32)
                """
                # 使用 DIN 注意力机制，将历史行为序列根据目标 item 表示进行加权聚合
                pooled = self.din_attention(target_item_embed, history_embeds, mask)  # (batch_size, 1, emb_dim)
                history_seq_embeds.append(pooled)  # 收集每个序列对应的兴趣表示

            # 拼接所有嵌入特征:多个序列兴趣表示拼接
            history_seq_embeds_concat = tf.concat(history_seq_embeds,
                                                  axis=1)  # shape = (batch_size, num_history_seq_fields, emb_dim)

        # ---------- 3-2、历史行为序列特征-历史行为序列特征通过GRU的最后一个隐状态获取序列整体的兴趣演变表示 ----------
        if self.history_seq_feats:
            history_seq_gru_embeds = []
            for i, feat in enumerate(self.history_seq_feats):
                # 获取当前历史序列的 embedding 表示
                history_embeds = self.history_seq_embed_dict[feat['feat']](
                    history_seq_inputs[i])  # (batch_size, seq_len, emb_dim)  (3, 3, 5)  # (B, T, D)

                # 构建 mask（可选：你可以控制是否在 GRU 中使用 mask）
                # 传给 GRU 的 mask 必须是 tf.bool, 用于计算或加权时用 tf.float32
                mask = tf.cast(tf.not_equal(history_seq_inputs[i], 0), tf.bool)  # (B, T)
                # 调用专属的 GRU 模块，返回每条序列的最终兴趣表示
                # return_sequences=True  → 输出 shape = (2, 3, 8)
                # return_sequences=False → 输出 shape = (2, 8)，表示每个序列的最后一个状态向量。
                gru_output = self.history_seq_gru_dict[feat['feat']](history_embeds,
                                                                     mask=mask)  # (B, D)  (batch_size, emb_dim)   (3, 5)

                pooled = tf.expand_dims(gru_output,
                                        axis=1)  # 保持维度一致tf.expand_dims(gru_output, axis=1) : (B, D) → (B, 1, D)  (batch_size, 1, emb_dim)
                history_seq_gru_embeds.append(pooled)

            # 拼接所有嵌入特征:多个序列兴趣表示拼接
            history_seq_gru_embeds_concat = tf.concat(history_seq_gru_embeds,
                                                      axis=1)  # shape = (batch_size, num_history_seq_fields, emb_dim)

        # 拼接生成各种DNN的输入
        sparse_flat = tf.reshape(sparse_embeds,
                                 shape=(-1, len(self.sparse_feats) * self.emb_size))  # (2,3,5) => (2,3*5)

        # 初始化 DNN 输入
        dnn_input_parts = [dense_inputs,
                           sparse_flat]  # [shape=(batch_size, dense_dim), shape=(batch_size, num_sparse_fields * emb_dim)]

        if self.seq_feats:
            # shape: (batch_size, num_seq_fields, emb_dim) → (batch_size, num_seq_fields * emb_dim)
            seq_flat = tf.reshape(seq_embeds_concat, shape=(-1, seq_embeds_concat.shape[1] * self.emb_size))
            dnn_input_parts.append(seq_flat)

        if self.history_seq_feats:
            # 和目标加权后的兴趣向量（如 DIN）添加到DNN：shape: (batch_size, num_history_seq_fields, emb_dim) → (batch_size, num_history_seq_fields * emb_dim)
            history_seq_flat = tf.reshape(history_seq_embeds_concat,
                                          shape=(-1, history_seq_embeds_concat.shape[1] * self.emb_size))
            dnn_input_parts.append(history_seq_flat)
            # GRU 最后隐状态的兴趣表示：                shape: (batch_size, num_history_seq_fields, emb_dim) → (batch_size, num_history_seq_fields * emb_dim)
            history_seq_gru_flat = tf.reshape(history_seq_gru_embeds_concat,
                                              shape=(-1, history_seq_gru_embeds_concat.shape[1] * self.emb_size))
            dnn_input_parts.append(history_seq_gru_flat)

        # 拼接所有部分作为 DNN 输入
        mmoe_input = tf.concat(dnn_input_parts,
                               axis=1)  # 最终 shape: (batch_size, dense_dim + sparse_dim + seq_dim + history_seq_dim + history_seq_gru_flat)

        # 2️⃣ 多专家输出
        expert_outputs = tf.stack([expert(mmoe_input, training=training) for expert in self.experts],
                                  axis=1)  # shape: (batch_size, num_experts, expert_output_dim)

        # 3️⃣ 每个任务的 gate
        gate_finish_weight = tf.expand_dims(self.gate_finish(mmoe_input), axis=-1)  # (batch_size, num_experts, 1)
        gate_like_weight = tf.expand_dims(self.gate_like(mmoe_input), axis=-1)  # (batch_size, num_experts, 1)

        # 4️⃣ Gate 加权求和（融合多个 expert）
        task_finish_input = tf.reduce_sum(gate_finish_weight * expert_outputs,
                                          axis=1)  # (batch_size, num_experts, expert_output_dim) ==> (batch_size, expert_output_dim)
        task_like_input = tf.reduce_sum(gate_like_weight * expert_outputs,
                                        axis=1)  # (batch_size, num_experts, expert_output_dim) ==> (batch_size, expert_output_dim)
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
    dense_feats = ['I1', 'I2']
    sparse_feats = ['C1', 'C2', 'C3']
    sequence_feats = ['S1', 'S2']
    feat_columns = [
        [{'feat': 'I1'}, {'feat': 'I2'}],
        [{'feat': 'C1', 'feat_num': 10}, {'feat': 'C2', 'feat_num': 8}, {'feat': 'C3', 'feat_num': 6}],
        [{'feat': 'S1', 'feat_num': 10}, {'feat': 'S2', 'feat_num': 20}],
        [{'feat': 'History_H1', 'target_emb_column': 'C1', 'target_item_index': 0},
         {'feat': 'History_H2', 'target_emb_column': 'C2', 'target_item_index': 1}]
    ]
    # target_emb_column
    model = MMOE_IN_DIEN(feat_columns=feat_columns, emb_size=5)

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

    print("====== Dense 输入 ======", feat_columns[0])
    print(dense_input)
    print()

    print("====== Sparse 输入 ======", feat_columns[1])
    print(sparse_input)
    print()

    print("====== 离散序列特征输入（如标题、标签等） ======")
    for i, feat in enumerate(sequence_feats):
        print(f"{feat} 输入:", feat_columns[2][i])
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