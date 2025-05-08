from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
import tensorflow as tf
import numpy as np

# 定义用户和物品特征列
user_feature_columns = [
    SparseFeat('user_id', vocabulary_size=10, embedding_dim=8),
    DenseFeat('user_age', 1),
    VarLenSparseFeat('user_history', vocabulary_size=50, embedding_dim=8, maxlen=5)  # 序列特征
]

item_feature_columns = [
    SparseFeat('item_id', vocabulary_size=20, embedding_dim=8),
    DenseFeat('item_price', 1),
]

# 定义输入数据
user_inputs = {
    'user_id': np.array([1, 2, 3, 4]),
    'user_age': np.array([25, 30, 22, 28], dtype=np.float32),
    'user_history': np.array([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=np.int32)
}

item_inputs = {
    'item_id': np.array([10, 11, 12, 13]),
    'item_price': np.array([100, 200, 150, 175], dtype=np.float32)
}

# 使用 input_from_feature_columns 来获取稀疏特征和稠密特征
user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(
    user_inputs, user_feature_columns, l2_reg_embedding=0.0, support_dense=True, seed=1024, support_masking=True
)

item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(
    item_inputs, item_feature_columns, l2_reg_embedding=0.0, support_dense=True, seed=1024
)

# 打印结果，查看处理后的特征形状
print(user_sparse_embedding_list, user_dense_value_list)
print(item_sparse_embedding_list, item_dense_value_list)
