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


# This example shows how to use MMOE to solve a multi task learning problem.
# You can get the demo data census-income.sample and run the following codes.
# https://deepctr-doc.readthedocs.io/en/latest/Examples.html

if __name__ == "__main__":
    # 1. 定义数据字段名称
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like", "music_id", "device", "time", "duration_time"]

    # 2. 读取原始数据并命名列
    data = pd.read_csv(r'D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\train_2.csv',sep='\t', names=column_names)
    # 3. 构造多任务标签
    pass

    # 4. 区分稀疏特征和连续特征
    columns = data.columns.values.tolist()
    sparse_features  = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
    dense_features = [col for col in columns if col not in sparse_features and col not in ['finish', 'like']]

    # 5. 缺失值填充：稀疏特征用 "-1" 填充，连续特征用 0 填充
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features]  = data[dense_features].fillna(0)

    # 6. 连续特征归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 7. 稀疏特征编码（LabelEncoder）
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 8. 构建 DeepCTR 所需的特征列
    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4) for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    # 9. 特征列用于 DNN 和 Linear 部分
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    # 10. 获取模型输入所需的特征名列表
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 11. 划分训练集和测试集
    train, test = train_test_split(data, test_size=0.2, random_state=2020)

    # 12. 构造模型输入字典
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 13. 定义 MMOE 模型（多任务输出：finish分类 + like分类）
    model = MMOE(dnn_feature_columns, tower_dnn_hidden_units=[],task_types=['binary', 'binary'],task_names=['finish', 'like'])

    # 14. 编译模型
    model.compile("adam",loss=["binary_crossentropy", "binary_crossentropy"],metrics=['accuracy', keras.metrics.AUC(name='auc')])

    # 15. 模型训练
    history = model.fit(train_model_input,[train['finish'].values, train['like'].values],batch_size=256, epochs=10, verbose=2, validation_split=0.2)

    # 16. 模型预测
    pred_ans = model.predict(test_model_input, batch_size=256)

    # 17. AUC 评估
    print("test finish AUC",  round(roc_auc_score(test['finish'], pred_ans[0]), 4))
    print("test like AUC", round(roc_auc_score(test['like'], pred_ans[1]), 4))

    # test income AUC 0.2696
    # test marital AUC 0.8363

    # 18. 可视化数据:可以自己选择数据进行可视化matplotlib绘画 ---
    print(pd.DataFrame(history.history))
    print(history.epoch)
    plt.figure(figsize=(12, 5))


# test finish AUC 0.4732
# test like AUC nan
#        loss  finish_loss  like_loss  finish_accuracy  finish_auc  like_accuracy  like_auc  val_loss  val_finish_loss  val_like_loss  val_finish_accuracy  val_finish_auc  val_like_accuracy  val_like_auc
# 0  1.396225     0.681090   0.715135         0.702128    0.516610       0.005319  0.223075  1.356287         0.682060       0.674227             0.638298        0.231605                1.0      0.220403
# 1  1.349422     0.675411   0.674010         0.707447    0.286991       0.994681  0.234515  1.315335         0.678660       0.636676             0.638298        0.236035                1.0      0.274124
# 2  1.307051     0.669837   0.637213         0.707447    0.304620       0.994681  0.386512  1.276886         0.675293       0.601593             0.638298        0.373134                1.0      0.397872
# 3  1.267138     0.664461   0.602677         0.707447    0.383303       0.994681  0.455928  1.240132         0.672360       0.567771             0.638298        0.445813                1.0      0.462216
# 4  1.228732     0.659434   0.569297         0.707447    0.450730       0.994681  0.502401  1.204961         0.669531       0.535429             0.638298        0.493144                1.0      0.505145
# 5  1.191813     0.654386   0.537426         0.707447    0.489715       0.994681  0.529420  1.171772         0.666976       0.504795             0.638298        0.521510                1.0      0.530951
# 6  1.157078     0.649680   0.507396         0.707447    0.518464       0.994681  0.550352  1.141329         0.664775       0.476552             0.638298        0.543558                1.0      0.551277
# 7  1.125074     0.645123   0.479949         0.707447    0.538639       0.994681  0.565239  1.113962         0.662432       0.451529             0.638298        0.559199                1.0      0.565722
# 8  1.095742     0.640180   0.455559         0.707447    0.554189       0.994681  0.576930  1.088943         0.660274       0.428667             0.638298        0.571315                1.0      0.576954
# 9  1.067768     0.634972   0.432794         0.707447    0.565561       0.994681  0.585433  1.064854         0.658455       0.406397             0.638298        0.580424                1.0      0.585391
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]