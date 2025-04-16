# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


# ========== 标准库 ==========
import warnings

# ========== 第三方库 ==========
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score

from tensorflow import keras
import tensorflow.python.keras.engine.data_adapter as data_adapter

# ========== DeepCTR 模块 ==========
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM, MMOE

# ========== 设置选项 & 警告过滤 ==========
# 设置 pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# 忽略 matplotlib 的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")

# 修复 DeepCTR 与新版 TensorFlow 的兼容性问题
def _is_distributed_dataset_fixed(ds):
    return False
data_adapter._is_distributed_dataset = _is_distributed_dataset_fixed

# 1. 构造模拟输入数据
user_id = np.array([[1], [0], [1]])
item_id = np.array([[30], [20], [10]])
pic_vec = np.array([
    [0.1, 0.5, 0.4, 0.3, 0.2],
    [0.1, 0.5, 0.4, 0.3, 0.2],
    [0.1, 0.5, 0.4, 0.3, 0.2]
])
label = np.array([1, 0, 1])

# 构造 DataFrame 数据
data = pd.DataFrame({
    'user_id': user_id.flatten(),
    'item_id': item_id.flatten(),
    'pic_vec_0': pic_vec[:, 0],
    'pic_vec_1': pic_vec[:, 1],
    'pic_vec_2': pic_vec[:, 2],
    'pic_vec_3': pic_vec[:, 3],
    'pic_vec_4': pic_vec[:, 4],
    'label': label
})


# 2. 定义特征列结构
fixlen_feature_columns = [
    SparseFeat('user_id', 120, embedding_dim=4),
    SparseFeat('item_id', 60,  embedding_dim=4),
    DenseFeat('pic_vec_0', 1),
    DenseFeat('pic_vec_1', 1),
    DenseFeat('pic_vec_2', 1),
    DenseFeat('pic_vec_3', 1),
    DenseFeat('pic_vec_4', 1)
]

# 3. 获取特征名列表（用于模型输入字典）
fixlen_feature_names = get_feature_names(fixlen_feature_columns)


# 4. 划分训练集和测试集
train, test = train_test_split(data, test_size=0.2, random_state=2018)

# 5. 构建训练集和测试集的输入格式（字典类型）
train_model_input = {name: train[name] for name in fixlen_feature_names}
test_model_input  = {name: test[name]  for name in fixlen_feature_names}

# 6. 定义 DeepFM 模型
model = DeepFM(
    linear_feature_columns=fixlen_feature_columns,
    dnn_feature_columns=fixlen_feature_columns,
    task='binary'
)

# 7. 编译模型
model.compile(optimizer='adagrad', loss='binary_crossentropy')

# 8. 模型训练
model.fit(train_model_input, train['label'].values, epochs=5, batch_size=2, verbose=1)

# 9. 模型预测与评估
pred_ans = model.predict(test_model_input, batch_size=2)
print("Predictions:", pred_ans)

# Epoch 1/5
# 1/1 [==============================] - 1s 908ms/step - loss: 0.6159
# Epoch 2/5
# 1/1 [==============================] - 0s 2ms/step - loss: 0.6113
# Epoch 3/5
# 1/1 [==============================] - 0s 2ms/step - loss: 0.6071
# Epoch 4/5
# 1/1 [==============================] - 0s 2ms/step - loss: 0.6032
# Epoch 5/5
# 1/1 [==============================] - 0s 2ms/step - loss: 0.5995
# Predictions: [[0.54960644]]