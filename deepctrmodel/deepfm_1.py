# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM
from tensorflow import keras
import tensorflow.python.keras.engine.data_adapter as data_adapter
import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个

# --- 修复 DeepCTR 与新版 TensorFlow 的数据适配器兼容性 ---
def _is_distributed_dataset_fixed(ds):
    return False

data_adapter._is_distributed_dataset = _is_distributed_dataset_fixed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")


# --- 1. 读取数据 ---
# 定义数据集的列名并加载数据
column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like", "music_id", "device", "time", "duration_time"]
data = pd.read_csv(r'D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\train_1.csv', sep='\t', names=column_names)


# --- 2. 特征预处理 ---
# 定义稀疏特征和密集特征
sparse_feats  = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
dense_feats   = ["time", "duration_time"]

# 定义处理稀疏特征的函数：对每个稀疏特征进行LabelEncoder编码
def process_sparse_feats(data, sparse_feats):
    for feat in sparse_feats:
        le = LabelEncoder()  # 初始化LabelEncoder
        data[feat] = le.fit_transform(data[feat])  # 对每个特征进行编码
    return data

# 定义处理密集特征的函数：对每个密集特征进行标准化
def process_dense_feats(data, dense_feats):
    scaler = StandardScaler()  # 初始化标准化Scaler
    data[dense_feats] = scaler.fit_transform(data[dense_feats])  # 对每个密集特征进行标准化
    return data

# 对数据进行预处理
data = process_dense_feats(data, dense_feats)    # 标准化密集特征
data = process_sparse_feats(data, sparse_feats)  # 编码稀疏特征


# --- 3. 定义特征列 ---
# 定义固定长度的特征列，包括稀疏特征和密集特征
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique()+1, embedding_dim=4) for feat in sparse_feats] \
                        + [DenseFeat(feat, 1) for feat in dense_feats]

# 获取所有特征名，用于构建输入数据
fixlen_feature_names = get_feature_names(fixlen_feature_columns)


# --- 4. 划分训练集和测试集 ---
# 使用train_test_split将数据集划分为训练集和测试集，测试集比例为20%
train, test = train_test_split(data, test_size=0.2, random_state=2018)


# --- 5. 构建模型输入 ---
# 构建训练集和测试集的输入数据，按特征名选择对应的数据列
train_model_input = {name: train[name] for name in fixlen_feature_names}
test_model_input  = {name: test[name]  for name in fixlen_feature_names}


# --- 6. 初始化 DeepFM 模型结构 ---
# 创建一个DeepFM模型实例，linear和dnn的特征列都是fixlen_feature_columns
model = DeepFM(
    linear_feature_columns=fixlen_feature_columns,
    dnn_feature_columns=fixlen_feature_columns,
    task='binary'  # 任务是二分类
)


# --- 7. 编译模型 ---
# 编译模型，使用'Adagrad'优化器，二元交叉熵损失函数，并设置评估指标为'accuracy'和'AUC'
model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC(name='auc')])


# --- 8. 模型训练 ---
# 定义目标标签列，目标为'finish'，即用户是否完成操作
target = ['finish']
# 训练模型，设置训练批次大小为256，训练10个epoch，使用20%的数据作为验证集
history = model.fit(
    train_model_input,
    train[target].values,  # 训练目标标签
    batch_size=256,
    epochs=10,
    verbose=2,
    validation_split=0.2  # 设定验证集的比例
)


# --- 9. 模型预测与评估 ---
# 使用模型对测试集进行预测，批次大小为256
pred_ans = model.predict(test_model_input, batch_size=256)
# 输出模型评估结果：LogLoss 和 AUC
print("test LogLoss", round(log_loss(test[target].values, pred_ans, labels=[0, 1]), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))



# --- 10. 可视化数据 ---
print(pd.DataFrame(history.history))
print(history.epoch)
plt.figure(figsize=(12, 5))


# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.epoch, history.history['loss'], label='Train Loss')  # 训练损失
plt.plot(history.epoch, history.history['val_loss'], label='Val Loss')  # 验证损失
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()


# 绘制AUC和准确度曲线
plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history['auc'], label='Train AUC')  # 训练AUC
plt.plot(history.epoch, history.history['accuracy'], label='Train Accuracy')  # 训练准确度
plt.plot(history.epoch, history.history['val_auc'], label='Val AUC')  # 验证AUC
plt.plot(history.epoch, history.history['val_accuracy'], label='Val Accuracy')  # 验证准确度
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('AUC and Accuracy over Epochs')
plt.legend()

# 调整子图布局
plt.tight_layout()

# 显示图表
plt.show()


