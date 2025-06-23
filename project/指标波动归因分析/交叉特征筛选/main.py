# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email :  mingyang.zhang@ushow.media

# 项目入口 main_recall.py
# 用于训练和评估模型

from datasets.data_loader import load_dataset
from trainers.trainer import train_and_evaluate
import tensorflow as tf
from datasets.utils_tf import create_dataset
from config.data_config import *
from models.Fm import Fm
from models.FM_Embedding import FM_MTL
from models.WideAndDeep import WideAndDeep
from models.DeepFm import DeepFM_MTL
from models.XDeepFM import XDeepFM_MTL
from models.XDeepFM_Transform import XDeepFM_Transform_MTL
from models.DCN_Model_MTL import DCN_Model_MTL
from models.XDeepFM_Transform_DCN_CrossNetwork_Attention_MTL import XDeepFM_Transform_DCN_CrossNetwork_Attention_MTL
from models.DeepCrossing_Residual import DeepCrossing_Residual
from models.NFM import NFM
from models.AFm import AFm
from models.AFM_Embedding import AFM_Embedding
from models.MMOE import MMOE
from models.DeepFm_DIN_GRU import DeepFm_DIN_GRU
from models.MMOE_IN_DIEN import MMOE_IN_DIEN
from models.MMOE_OUT_DIEN import MMOE_OUT_DIEN
from models.MMOE_NLP_Attention import MMOE_NLP_Attention



def get_model(model_name, feat_columns, embed_dim=None, batch_size=None):
    model_factory = {
        "Fm": lambda: Fm(feat_columns),
        "FM_MTL": lambda: FM_MTL(feat_columns),
        "WideAndDeep": lambda: WideAndDeep(feat_columns),
        "DeepFM_MTL": lambda: DeepFM_MTL(feat_columns),
        "XDeepFM_MTL": lambda: XDeepFM_MTL(feat_columns),
        "XDeepFM_Transform_MTL": lambda: XDeepFM_Transform_MTL(feat_columns),
        "DCN_Model_MTL": lambda: DCN_Model_MTL(feat_columns),
        "XDeepFM_Transform_DCN_CrossNetwork_Attention_MTL": lambda: XDeepFM_Transform_DCN_CrossNetwork_Attention_MTL(feat_columns),   # 这是自己进行创新的算法-DCN-CrossNetwork引入了Attention注意力机制
        "DeepCrossing_Residual": lambda: DeepCrossing_Residual(feat_columns),
        "NFM": lambda: NFM(feat_columns),
        "AFm": lambda: AFm(feat_columns),
        "AFM_Embedding": lambda: AFM_Embedding(feat_columns),
        "MMOE": lambda: MMOE(feat_columns),
        "DeepFm_DIN_GRU": lambda: DeepFm_DIN_GRU(feat_columns),
        "MMOE_IN_DIEN": lambda: MMOE_IN_DIEN(feat_columns),
        "MMOE_OUT_DIEN": lambda: MMOE_OUT_DIEN(feat_columns),
        "MMOE_NLP_Attention": lambda: MMOE_NLP_Attention(feat_columns)
    }

    if model_name not in model_factory:
        raise ValueError(f"未知模型: {model_name}")
    return model_factory[model_name]()



if __name__ == "__main__":
    # 1. 加载数据集 （小数据集是用逗号分割单 ，大数据集是用\t分割的）
    data, train_ds, valid_ds,test_ds, feat_columns  = create_dataset(file_path="data_files/20250623.csv", embed_dim=embed_dim)
    print(data.head(5))
    """
       uid  user_city  item_id  author_id  item_city  channel  finish  like  music_id  device      time  duration_time  actors   genres        history_item_ids         history_citys
    0  259         31       26        210          1        0       0     0        82     202  0.250521       3.830290  [4, 0]   [9, 0]  [247, 75, 176, 197, 0]  [14, 28, 29, 103, 0]
    1   24          4       27        244          2        0       1     0        85     151  0.467755       0.593142  [8, 7]   [6, 0]      [149, 95, 0, 0, 0]     [33, 54, 0, 0, 0]
    2   10        126      130        265          3        0       0     0       113      97  0.516757      -1.095805  [7, 3]  [11, 2]     [137, 284, 0, 0, 0]     [68, 81, 0, 0, 0]
    3   90          6      131        266          4        0       0     0         0     112  0.519817      -0.392077  [5, 6]  [11, 7]       [139, 0, 0, 0, 0]      [32, 0, 0, 0, 0]
    4  242         50      132        202          5        0       1     0         0     155  0.502151       1.015379  [6, 0]   [4, 5]   [244, 214, 241, 0, 0]   [86, 87, 108, 0, 0]
    """

    for item in train_ds:
        print(item)
        break
    """
    ((<tf.Tensor: shape=(2, 8), dtype=int32, numpy=
    array([[ 35, 120, 270,  54, 121,   0,   0, 115],
           [ 30,  83,  65, 109,  12,   0,   0, 209]])>, <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[-0.12916128,  0.7338877 ],
           [ 0.37441754, -0.6735681 ]], dtype=float32)>, <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[3, 0],
           [5, 7]])>, <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[10,  0],
           [ 8,  0]])>, <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])>, <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
    array([[34,  7, 24, 40, 10],
           [27, 30,  0,  0,  0]])>), {'finish': <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 0.], dtype=float32)>, 'like': <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 0.], dtype=float32)>})
    """


    for item in feat_columns:
        print(item)
    """
    [{'feat': 'time'}, {'feat': 'duration_time'}]
    [{'feat': 'uid', 'feat_num': 289}, {'feat': 'user_city', 'feat_num': 129}, {'feat': 'item_id', 'feat_num': 291}, {'feat': 'author_id', 'feat_num': 289}, {'feat': 'item_city', 'feat_num': 136}, {'feat': 'channel', 'feat_num': 4}, {'feat': 'music_id', 'feat_num': 115}, {'feat': 'device', 'feat_num': 289}]
    [{'feat': 'actors', 'feat_num': 10}, {'feat': 'genres', 'feat_num': 13}]
    [{'feat': 'history_item_ids', 'feat_emb_source': 'item_id'}, {'feat': 'history_citys', 'feat_emb_source': 'item_city'}]
    """


    # 2. 调用模型
    model_name = "Fm"
    model = get_model(model_name, feat_columns)

    # 3. 训练并评估
    train_and_evaluate(model, train_ds, valid_ds,test_ds)

    # 假设 model 是训练好的 FM 实例
    import numpy as np
    V_matrix = model.V.numpy()  # shape: (num_features, emb_size)
    print(V_matrix)

    # 计算交叉权重矩阵
    cross_weights = np.dot(V_matrix, V_matrix.T)  # shape: (num_features, num_features)
    print(cross_weights)
    print(cross_weights.shape)

    import numpy as np

    # cross_weights 是对称矩阵，V_matrix 是 (num_features, emb_size)
    num_features = cross_weights.shape[0]

    # 保存所有特征对及其交互值（只保留上三角非对角）
    interactions = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            interactions.append(((i, j), cross_weights[i, j]))

    # 按绝对值排序（从大到小）
    top_k = sorted(interactions, key=lambda x: abs(x[1]), reverse=True)[:10]

    # 输出 Top10 特征交互对
    print("Top 10 特征交互对（按交互强度）:")
    column_names = ["platform", "app_name", "app_version", "country", "region", "language", "channel", "create_date","active_last_date"]
    for (i, j), weight in top_k:
        name_i = column_names[i]
        name_j = column_names[j]
        print(f"{name_i} × {name_j} : 权重 = {weight:.6f}")

    """
      platform app_name app_version country   region language              channel create_date active_last_date  target
    0  android       sm      8.87.3      BD  Area_BD       en              unknown  2025-06-13       2022-06-14    1080
    1  android       sm      8.87.3      IN  Area_IN       hi    googleadwords_int  2025-04-15       2025-06-12     109
    2  android       sm      8.87.3      ID  Area_ID       in  bytedanceglobal_int  2025-06-10       2025-06-11    -244
    3  android       sm      8.87.3      IN  Area_IN       en              organic  2025-06-02       2025-06-12     133
    4  android       sm      8.86.4      IN  Area_IN       en              organic  2025-05-30       2025-06-12     124
       platform  app_name  app_version  country  region  language  channel  create_date  active_last_date  target
    0         0         1           20        3       0         4       10          223                 0    1080
    1         0         1           20       23       6         7        3          164                 6     109
    2         0         1           20       22       5         9        2          220                 5    -244
    3         0         1           20       23       6         4        5          212                 6     133
    4         0         1           18       23       6         4        5          209                 6     124
    [[], [{'feat': 'platform', 'feat_num': 3}, {'feat': 'app_name', 'feat_num': 5}, {'feat': 'app_version', 'feat_num': 26}, {'feat': 'country', 'feat_num': 60}, {'feat': 'region', 'feat_num': 21}, {'feat': 'language', 'feat_num': 28}, {'feat': 'channel', 'feat_num': 13}, {'feat': 'create_date', 'feat_num': 224}, {'feat': 'active_last_date', 'feat_num': 7}]]
    process_sparse_feats: 100%|██████████| 9/9 [00:00<00:00, 595.37it/s]
    """

    # Top 10 特征交互对（按交互强度）:
    # create_date × active_last_date : 权重 = 0.066382
    # app_version × active_last_date : 权重 = 0.032802
    # platform × active_last_date : 权重 = -0.029710
    # app_version × create_date : 权重 = 0.015355
    # platform × create_date : 权重 = -0.015010l
    # app_name × app_version : 权重 = 0.011878
    # country × active_last_date : 权重 = -0.010133
    # country × create_date : 权重 = -0.009827
    # app_name × active_last_date : 权重 = 0.009772