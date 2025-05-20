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
    }

    if model_name not in model_factory:
        raise ValueError(f"未知模型: {model_name}")
    return model_factory[model_name]()



if __name__ == "__main__":
    # 1. 加载数据集 （小数据集是用逗号分割单 ，大数据集是用\t分割的）
    data, train_ds, valid_ds,test_ds, feat_columns  = create_dataset(file_path="data_files/train_2_with_history.csv", embed_dim=embed_dim)
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
    model_name = "MMOE_IN_DIEN"
    model = get_model(model_name, feat_columns)

    # 3. 训练并评估
    train_and_evaluate(model, train_ds, valid_ds,test_ds)