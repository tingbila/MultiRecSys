# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ------------------------------------------------------------------------------
# -- author：张明阳
# -- create：2025年5月8日00:46:53
# -- function：DSSM-练习版本V3-faiss查询用户相似的item+DeepCTR
# -- document:
# ------------------------------------------------------------------------------


from tensorflow.keras.models import Model
from deepctr.layers.core import DNN
from deepctr.feature_column import build_input_features, input_from_feature_columns, SparseFeat, create_embedding_matrix
from deepctr.layers.utils import combined_dnn_input
from deepctr.layers.core import PredictionLayer

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
import faiss

from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class DSSMModel(Model):
    """
    基于 DeepCTR 的 DSSM 实现，用于计算用户与物品向量之间的相似度（如余弦相似度），适合推荐/匹配场景。
    """
    def __init__(self, user_feature_columns, item_feature_columns,
                 user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32),
                 dnn_activation='tanh', dnn_use_bn=False,
                 l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0,
                 seed=1024, metric='cos'):
        super(DSSMModel, self).__init__()

        # print(user_feature_columns,item_feature_columns)
        # 保存输入特征列和正则化超参数
        self.user_feature_columns = user_feature_columns  # [SparseFeat('user_id', vocabulary_size=10, embedding_dim=8)]
        self.item_feature_columns = item_feature_columns  # [SparseFeat('item_id', vocabulary_size=20, embedding_dim=8)]
        self.l2_reg_embedding = l2_reg_embedding

        # 在DeepCTR的更新版本中，embedding_matrix_dict参数已经被移除。embedding_matrix_dict是用来初始化预训练的嵌入矩阵的，但
        # DeepCTR现在的设计中会自动处理嵌入矩阵的初始化。因此，你不需要再手动传入embedding_matrix_dict
        # self.embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
        #                                                 seed=seed,
        #                                                 seq_mask_zero=True)

        # 构建用户和物品的输入特征字典（用于 Keras Functional API）
        # build_input_features(feature_columns) 会根据传入的特征列（如 SparseFeat, DenseFeat 等）自动生成对应的 tf.keras.Input 层，并返回一个字典，key 是特征名，value 是输入层张量
        # {'user_id': <KerasTensor: shape=(None, 1) dtype=int32 ...>}
        # user_id_input = tf.keras.Input(shape=(1,), name='user_id', dtype='int32')
        # 无论是离散特征（SparseFeat）还是连续特征（DenseFeat），build_input_features 默认都会创建 shape=(None, 1) 的输入层
        # 变长序列特征（VarLenSparseFeat）对应的是Input(shape=(10,), dtype='int32', name='hist_item_id') 序列的长度
        self.user_input_features = build_input_features(user_feature_columns)
        self.item_input_features = build_input_features(item_feature_columns)
        # print(self.user_input_features)
        # print(self.item_input_features)
        # [SparseFeat(name='user_id', vocabulary_size=10, embedding_dim=8, use_hash=False, vocabulary_path=None, dtype='int32', embeddings_initializer=<tensorflow.python.keras.initializers.initializers_v1.RandomNormal object at 0x00000296C5878700>, embedding_name='user_id', group_name='default_group', trainable=True), DenseFeat(name='user_age', dimension=1, dtype='float32', transform_fn=None)]
        # [SparseFeat(name='item_id', vocabulary_size=20, embedding_dim=8, use_hash=False, vocabulary_path=None, dtype='int32', embeddings_initializer=<tensorflow.python.keras.initializers.initializers_v1.RandomNormal object at 0x00000296C59B5DC0>, embedding_name='item_id', group_name='default_group', trainable=True), DenseFeat(name='item_price', dimension=1, dtype='float32', transform_fn=None)]


        # 构建用户和物品 DNN 模块（共享结构也可根据需要共享）
        # l2_reg 是 L2 正则化（L2 regularization），用于防止 神经网络过拟合。在 DNN 里，它会对 每一层的权重参数 加一个正则项（惩罚项）来控制模型复杂度
        # l2_reg 的惩罚项 确实是体现在 loss 损失函数里，它并不会直接改变模型的输出结果，而是通过 影响损失函数 来间接调整模型的学习方向
        # loss = loss_data + λ * Σ||W||²
        self.user_dnn = DNN(user_dnn_hidden_units, activation=dnn_activation,
                            l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                            use_bn=dnn_use_bn, seed=seed)
        self.item_dnn = DNN(item_dnn_hidden_units, activation=dnn_activation,
                            l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                            use_bn=dnn_use_bn, seed=seed)

        # 使用二分类预测层（sigmoid 输出）
        self.prediction = PredictionLayer("binary")


    def cosine_similarity(self, a, b):
        """
        计算两个向量之间的余弦相似度
        a, b: shape=(batch_size, embedding_dim)
        返回 shape=(batch_size, 1)
        """
        a = tf.nn.l2_normalize(a, axis=1)  # 对每个向量按行归一化
        b = tf.nn.l2_normalize(b, axis=1)
        return tf.reduce_sum(tf.multiply(a, b), axis=1, keepdims=True)


    def call(self, inputs, training=None):
        """
        前向传播逻辑
        inputs: 字典，包含用户和物品的所有输入特征
        training: 标志是否为训练模式
        """
        # print(inputs) # {'user_id': <tf.Tensor: shape=(4,), dtype=int32, numpy=array([1, 2, 3, 4])>, 'item_id': <tf.Tensor: shape=(4,), dtype=int32, numpy=array([10, 11, 12, 13])>}
        # 将输入拆分为用户和物品部分
        user_inputs = {k: inputs[k] for k in self.user_input_features}
        item_inputs = {k: inputs[k] for k in self.item_input_features}
        # print(user_inputs)
        # print(item_inputs)
        # {'user_id': array([1, 2, 3, 4]), 'user_age': array([25., 30., 22., 28.], dtype=float32)}
        # {'item_id': array([10, 11, 12, 13]), 'item_price': array([100., 200., 150., 175.], dtype=float32)}

        # 获取嵌入和数值特征列表，支持稠密特征，添加 L2 正则项
        # user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(
        #     user_inputs,                # 输入的用户特征字典（key 是特征名，value 是张量）
        #     self.user_feature_columns,  # 用户侧的特征列定义（SparseFeat/DenseFeat/VarLenSparseFeat）
        #     self.l2_reg_embedding,      # 嵌入层使用的 L2 正则强度（仅对 SparseFeat 有效）
        #     support_dense=True,         # 是否支持稠密（连续数值）特征
        #     seed=1024                   # 随机种子，保证可重复
        # )
        # 于每个 DenseFeat（连续特征）：直接将原始值从 user_inputs 中取出，不做变换，
        # 不在需要传入embedding_matrix_dict=self.embedding_matrix_dict
        user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(
            user_inputs, self.user_feature_columns, self.l2_reg_embedding,
            support_dense=True, seed=1024
        )   # print(user_sparse_embedding_list, user_dense_value_list) # [shape=(4, 8),shape=(4, 1, 8),shape=(4,)]
        # 如果有形状为 [4, 1, 8] 的张量，移除额外的维度，使其变为 [4, 8]
        user_sparse_embedding_list = [tf.squeeze(embed, axis=1) if embed.shape[1] == 1 else embed for embed in user_sparse_embedding_list]


        item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(
            item_inputs, self.item_feature_columns, self.l2_reg_embedding,
            support_dense=True, seed=1024
        )  # print(item_sparse_embedding_list, item_dense_value_list)  # [shape=(4, 8),shape=(4, 1, 8),shape=(4,)]
        # 如果有形状为 [4, 1, 8] 的张量，移除额外的维度，使其变为 [4, 8]
        item_sparse_embedding_list = [tf.squeeze(embed, axis=1) if embed.shape[1] == 1 else embed for embed in item_sparse_embedding_list]


        # 拼接稀疏嵌入和稠密特征，作为 DNN 输入:可以理解为将离散特征的Embedding向量和连续特征进行concat
        user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
        item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
        # print(user_dnn_input.shape)  # (4, 17)
        # print(item_dnn_input.shape)  # (4, 17)

        # 通过用户和物品 DNN 获得表示向量
        user_dnn_output = self.user_dnn(user_dnn_input, training=training)
        item_dnn_output = self.item_dnn(item_dnn_input, training=training)
        # print(user_dnn_output.shape,item_dnn_output.shape)  # (4, 32) (4, 32)

        # 计算相似度分数（如余弦）
        score = self.cosine_similarity(user_dnn_output, item_dnn_output)
        # print(score) # shape=(4, 1)

        # 使用 sigmoid 得到二分类概率输出
        output = self.prediction(score)

        # 保存中间向量（用于调试/可视化）
        self.user_embedding = user_dnn_output
        self.item_embedding = item_dnn_output

        return output



if __name__ == '__main__':
    # 定义用户和物品特征列，使用 SparseFeat（稀疏离散特征）、DenseFeat（连续特征）和 VarLenSparseFeat（变长序列特征）
    user_feature_columns = [
        SparseFeat('user_id', vocabulary_size=10, embedding_dim=8),
        DenseFeat('user_age', 1),  # 连续特征：用户年龄
        VarLenSparseFeat(SparseFeat('user_history', vocabulary_size=20, embedding_dim=8), maxlen=5)  # 用户的历史浏览物品（变长序列）
    ]

    item_feature_columns = [
        SparseFeat('item_id', vocabulary_size=20, embedding_dim=8),
        DenseFeat('item_price', 1),  # 连续特征：物品价格
        VarLenSparseFeat(SparseFeat('item_tags', vocabulary_size=30, embedding_dim=8), maxlen=3)  # 物品的标签（变长序列）
    ]

    print(user_feature_columns)
    print(item_feature_columns)

    # 构造输入数据
    user_model_input = {
        'user_id': np.array([1, 2, 3, 4]),
        'user_age': np.array([25, 30, 22, 28], dtype=np.float32),  # 连续特征：用户年龄
        'user_history': pad_sequences([[1, 3, 4], [5, 6, 7, 8], [9], [10, 11, 12]], maxlen=5, padding='post', value=0)
        # 用户历史浏览物品（变长序列）
    }
    # {'user_id': array([1, 2, 3, 4]), 'user_age': array([25., 30., 22., 28.], dtype=float32),
    #  'user_history': array([[1, 3, 4, 0, 0],
    #                         [5, 6, 7, 8, 0],
    #                         [9, 0, 0, 0, 0],
    #                         [10, 11, 12, 0, 0]])}
    item_model_input = {
        'item_id': np.array([10, 11, 12, 13]),
        'item_price': np.array([100, 200, 150, 175], dtype=np.float32),  # 连续特征：物品价格
        'item_tags': pad_sequences([[15, 17], [18, 20, 21], [22, 23], [24, 25]], maxlen=3, padding='post', value=0)
        # 物品标签（变长序列）
    }
    # {'item_id': array([10, 11, 12, 13]), 'item_price': array([100., 200., 150., 175.], dtype=float32),
    #  'item_tags': array([[15, 17, 0],
    #                      [18, 20, 21],
    #                      [22, 23, 0],
    #                      [24, 25, 0]])}
    labels = np.array([1, 0, 1, 0])

    print(user_model_input)
    print(item_model_input)
    print(labels)

    # 实例化模型
    model = DSSMModel(user_feature_columns, item_feature_columns)

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

    # 模型预测（inference 模式）
    outputs = model({**user_model_input, **item_model_input}, training=False)
    print("预测输出：", outputs.numpy())

    # 输出用户和物品的向量（可以用于向量检索）
    user_embeddings = model.user_embedding.numpy()
    item_embeddings = model.item_embedding.numpy()
    print("User Embedding:\n", user_embeddings.shape)  # (4, 32)
    print("Item Embedding:\n", item_embeddings.shape)  # (4, 32)


    # -------------------------- 基于余弦相似度获取user的相似item --------------------------
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

