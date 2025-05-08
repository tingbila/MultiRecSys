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
from deepctr.feature_column import build_input_features, input_from_feature_columns, SparseFeat
from deepctr.layers.utils import combined_dnn_input
from deepctr.layers.core import PredictionLayer

import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
import faiss

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

        print(user_feature_columns,item_feature_columns)
        # 保存输入特征列和正则化超参数
        self.user_feature_columns = user_feature_columns  # [SparseFeat('user_id', vocabulary_size=10, embedding_dim=8)]
        self.item_feature_columns = item_feature_columns  # [SparseFeat('item_id', vocabulary_size=20, embedding_dim=8)]
        self.l2_reg_embedding = l2_reg_embedding

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
        # OrderedDict([('user_id', <KerasTensor: shape=(None, 1) dtype=int32 (created by layer 'user_id')>)])
        # OrderedDict([('item_id', <KerasTensor: shape=(None, 1) dtype=int32 (created by layer 'item_id')>)])

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
        # print(user_inputs)   {'user_id': < tf.Tensor: shape = (4,), dtype = int32, numpy = array([1, 2, 3, 4]) >}
        # print(item_inputs)   {'item_id': < tf.Tensor: shape = (4,), dtype = int32, numpy = array([10, 11, 12, 13]) >}

        # 获取嵌入和数值特征列表，支持稠密特征，添加 L2 正则项
        # user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(
        #     user_inputs,                # 输入的用户特征字典（key 是特征名，value 是张量）
        #     self.user_feature_columns,  # 用户侧的特征列定义（SparseFeat/DenseFeat/VarLenSparseFeat）
        #     self.l2_reg_embedding,      # 嵌入层使用的 L2 正则强度（仅对 SparseFeat 有效）
        #     support_dense=True,         # 是否支持稠密（连续数值）特征
        #     seed=1024                   # 随机种子，保证可重复
        # )
        # 于每个 DenseFeat（连续特征）：直接将原始值从 user_inputs 中取出，不做变换
        user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(
            user_inputs, self.user_feature_columns, self.l2_reg_embedding,
            support_dense=True, seed=1024
        )
        # print(user_sparse_embedding_list,user_dense_value_list) # shape=(4, 8)

        item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(
            item_inputs, self.item_feature_columns, self.l2_reg_embedding,
            support_dense=True, seed=1024
        )
        print(item_sparse_embedding_list, item_dense_value_list) # shape=(4, 8)

        # 拼接稀疏嵌入和稠密特征，作为 DNN 输入
        user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
        item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
        # print(user_dnn_input.shape)
        # print(item_dnn_input.shape)
        # (4, 8)
        # (4, 8)

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
    # 定义用户和物品特征列，使用 SparseFeat（稀疏离散特征）
    user_feature_columns = [SparseFeat('user_id', vocabulary_size=10, embedding_dim=8)]
    item_feature_columns = [SparseFeat('item_id', vocabulary_size=20, embedding_dim=8)]
    print(user_feature_columns)
    print(item_feature_columns)

    # 构造输入数据
    user_model_input = {'user_id': np.array([1, 2, 3, 4])}
    item_model_input = {'item_id': np.array([10, 11, 12, 13])}
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
    # 为什么要将 DNN 的输出作为用户和物品的特征向量，这是理解 DSSM（Deep Structured Semantic Model） 或基于向量检索推荐系统的核心之一
    # DSSM 的目标是学习一个共同的语义空间，把用户和物品映射为向量，使得相关的 user-item 向量更相似（如余弦相似度更高）
    user_embeddings = model.user_embedding.numpy()
    item_embeddings = model.item_embedding.numpy()
    print("User Embedding:\n", user_embeddings.shape)  #  (4, 32)
    print("Item Embedding:\n", item_embeddings.shape)  #  (4, 32)



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

