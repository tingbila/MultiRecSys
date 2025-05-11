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
# 启用即时执行（eager execution） 如果不添加下面的参数，会报错： ValueError: tf.function-decorated function tried to create variables on non-first call.
# 对于 召回 阶段的任务，性能要求通常没有排序（Rank）或在线服务阶段那么高，因为召回更多是在训练过程中进行的。你可以在训练时先开启 Eager Execution，确保模型的正确性，并进行逐步优化。
tf.config.experimental_run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


import numpy as np
from tensorflow.keras.optimizers import Adam
import faiss

from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # ✅ 替代 experimental_run_functions_eagerly


def cosine_similarity(a, b):
    """
    计算两个向量之间的余弦相似度
    a, b: shape=(batch_size, embedding_dim)
    返回 shape=(batch_size, 1)
    """
    a = tf.nn.l2_normalize(a, axis=1)  # 对每个向量按行归一化
    b = tf.nn.l2_normalize(b, axis=1)
    return tf.reduce_sum(tf.multiply(a, b), axis=1, keepdims=True)



def DSSM(user_feature_columns, item_feature_columns,
              user_dnn_hidden_units=(64, 32), item_dnn_hidden_units=(64, 32),
              dnn_activation='tanh', dnn_use_bn=False,
              l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0,
              seed=1024, metric='cos'):
    """
    构建 DSSM（Deep Structured Semantic Model）模型，支持用户塔和物品塔的分离 DNN 网络建模。

    参数说明：
    - user_feature_columns, item_feature_columns: 特征列定义（稀疏特征、密集特征）
    - user_dnn_hidden_units, item_dnn_hidden_units: 用户/物品塔的 DNN 层结构
    - dnn_activation: DNN 激活函数，如 'relu' 或 'tanh'
    - dnn_use_bn: 是否使用 BatchNormalization
    - l2_reg_dnn, l2_reg_embedding: DNN 和 Embedding 层的 L2 正则
    - dnn_dropout: DNN Dropout 比例
    - seed: 随机种子
    - metric: 相似度函数，默认 'cos' 表示使用余弦相似度
    """

    # 1. 构建用户特征输入层（字典）
    user_features = build_input_features(user_feature_columns)
    # user_inputs_list： [<KerasTensor: shape=(None, 1) dtype=int32 (created by layer 'user_id')>, <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'user_age')>, <KerasTensor: shape=(None, 5) dtype=int32 (created by layer 'user_history')>]
    user_inputs_list = list(user_features.values())  # 提取为输入层列表

    # 2. 根据用户特征列提取 embedding 向量和数值特征
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(
        user_features, user_feature_columns, l2_reg_embedding,
        support_dense=True, seed=1024
    )
    # print(user_sparse_embedding_list)   # [<KerasTensor: shape=(None, 1, 8) dtype=float32 (created by layer 'sparse_emb_user_id')>, <KerasTensor: shape=(None, 1, 8) dtype=float32 (created by layer 'sequence_pooling_layer')>]
    # print(user_dense_value_list)        # [<KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'user_age')>]

    # 3. 将稀疏向量和数值特征拼接，作为用户塔 DNN 的输入
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
    # print(user_dnn_input)  # KerasTensor(type_spec=TensorSpec(shape=(None, 17), dtype=tf.float32, name=None), name='concat_1/concat:0', description="created by layer 'concat_1'")

    # 4. 构建物品特征输入层
    item_features = build_input_features(item_feature_columns)
    # item_inputs_list: [<KerasTensor: shape=(None, 1) dtype=int32 (created by layer 'item_id')>, <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'item_price')>, <KerasTensor: shape=(None, 3) dtype=int32 (created by layer 'item_tags')>]
    item_inputs_list = list(item_features.values())

    # 5. 根据物品特征列提取 embedding 向量和数值特征
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(
        item_features, item_feature_columns, l2_reg_embedding,
        support_dense=True, seed=1024
    )
    # print(item_sparse_embedding_list)  # [<KerasTensor: shape=(None, 1, 8) dtype=float32 (created by layer 'sparse_emb_item_id')>, <KerasTensor: shape=(None, 1, 8) dtype=float32 (created by layer 'sequence_pooling_layer_1')>]
    # print(item_dense_value_list)       # [<KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'item_price')>]

    # 6. 拼接物品稀疏和数值特征作为 DNN 输入
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
    # print(item_dnn_input) # KerasTensor(type_spec=TensorSpec(shape=(None, 17), dtype=tf.float32, name=None), name='concat_3/concat:0', description="created by layer 'concat_3'")


    # 7. 用户塔：使用多层 DNN 进行建模，输出用户向量
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(user_dnn_input)

    # KerasTensor(type_spec=TensorSpec(shape=(None, 32), dtype=tf.float32, name=None), name='dnn/dropout_1/Identity:0', description="created by layer 'dnn'")

    # 8. 物品塔：使用多层 DNN 进行建模，输出物品向量
    item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(item_dnn_input)
    # KerasTensor(type_spec=TensorSpec(shape=(None, 32), dtype=tf.float32, name=None), name='dnn_1/dropout_1/Identity:0', description="created by layer 'dnn_1'")

    # 9. 计算用户向量和物品向量之间的相似度（默认使用 cosine）
    score = cosine_similarity(user_dnn_out, item_dnn_out)
    # print(score)  # KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name=None), name='tf.math.reduce_sum/Sum:0', description="created by layer 'tf.math.reduce_sum'")

    # 10. 经过 sigmoid 层输出点击概率（二分类任务）
    output = PredictionLayer("binary")(score)
    # print(output)  # KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.float32, name=None), name='prediction_layer/Reshape:0', description="created by layer 'prediction_layer'")

    # 11. 构建最终模型，输入包括用户和物品的所有输入特征
    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    # 12. 绑定模型内部的中间向量（便于后续访问）
    model.__setattr__("user_input", user_inputs_list)    # 用户原始输入层
    model.__setattr__("item_input", item_inputs_list)    # 物品原始输入层
    model.__setattr__("user_embedding", user_dnn_out)    # 用户向量
    model.__setattr__("item_embedding", item_dnn_out)    # 物品向量

    return model




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
    model = DSSM(user_feature_columns, item_feature_columns)

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

    # 模型预测（inference 模式）
    outputs = model.predict({**user_model_input, **item_model_input})
    print("预测输出：", outputs)

    # print(model.summary())

    print(model.user_input)
    print(model.item_input)
    print(model.user_embedding)
    print(model.item_embedding)
    # ListWrapper([<KerasTensor: shape=(None, 1) dtype=int32 (created by layer 'user_id')>, <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'user_age')>, <KerasTensor: shape=(None, 5) dtype=int32 (created by layer 'user_history')>])
    # ListWrapper([<KerasTensor: shape=(None, 1) dtype=int32 (created by layer 'item_id')>, <KerasTensor: shape=(None, 1) dtype=float32 (created by layer 'item_price')>, <KerasTensor: shape=(None, 3) dtype=int32 (created by layer 'item_tags')>])
    # KerasTensor(type_spec=TensorSpec(shape=(None, 32), dtype=tf.float32, name=None), name='dnn/dropout_1/Identity:0', description="created by layer 'dnn'")
    # KerasTensor(type_spec=TensorSpec(shape=(None, 32), dtype=tf.float32, name=None), name='dnn_1/dropout_1/Identity:0', description="created by layer 'dnn_1'")
