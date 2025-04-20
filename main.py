# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email :  mingyang.zhang@ushow.media

# 项目入口 main.py
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



def get_model(model_name, feat_columns, embed_dim=None, batch_size=None):
    model_factory = {
        "Fm": lambda: Fm(feat_columns),
        "FM_MTL": lambda: FM_MTL(feat_columns),
        "WideAndDeep": lambda: WideAndDeep(feat_columns),
        "DeepFM_MTL": lambda: DeepFM_MTL(feat_columns),
        "XDeepFM_MTL": lambda: XDeepFM_MTL(feat_columns),
        "XDeepFM_Transform_MTL": lambda: XDeepFM_Transform_MTL(feat_columns),
        "DCN_Model_MTL": lambda: DCN_Model_MTL(feat_columns),
        "XDeepFM_Transform_DCN_CrossNetwork_Attention_MTL": lambda: XDeepFM_Transform_DCN_CrossNetwork_Attention_MTL(feat_columns),
        "DeepCrossing_Residual": lambda: DeepCrossing_Residual(feat_columns),
        "NFM": lambda: NFM(feat_columns),
        "AFm": lambda: AFm(feat_columns),
        "AFM_Embedding": lambda: AFM_Embedding(feat_columns),
        "MMOE": lambda: MMOE(feat_columns)
    }

    if model_name not in model_factory:
        raise ValueError(f"未知模型: {model_name}")
    return model_factory[model_name]()


if __name__ == "__main__":
    # 1. 加载数据集 （小数据集是用逗号分割单 ，大数据集是用\t分割的）
    data, train_ds, valid_ds,test_ds, feat_columns  = create_dataset(file_path="data_files/train_2.csv", embed_dim=embed_dim)
    # {'tokenizers': {'actors': <keras_preprocessing.text.Tokenizer object at 0x0000029E0D126250>, 'genres': <keras_preprocessing.text.Tokenizer object at 0x0000029E0D126E50>}, 'pad_len_dict': {'actors': 2, 'genres': 2}}

    # 2. 调用模型
    model_name = "AFM_Embedding"
    model = get_model(model_name, feat_columns)

    # 3. 训练并评估
    train_and_evaluate(model, train_ds, valid_ds,test_ds)
    
    #    finish_accuracy  finish_loss  like_accuracy  like_loss      loss  val_finish_accuracy  val_finish_loss  val_like_accuracy  val_like_loss  val_loss
    # 0             0.68     0.633306           0.68   0.675442  1.289731             0.571429         0.790080           0.714286       0.682514  1.501765
    # 1             0.68     0.477380           0.88   0.637175  1.134881             0.428571         0.981986           1.000000       0.629791  1.696161
    # 2             0.80     0.465747           0.96   0.586718  0.986695             0.428571         1.355031           1.000000       0.575927  2.081305
    # 3             0.80     0.369490           0.96   0.518443  0.916458             0.428571         1.668533           1.000000       0.529664  2.387516
    # 4             0.80     0.352170           0.96   0.497951  0.855214             0.428571         2.065776           1.000000       0.476627  2.785610
    # 5             0.80     0.308622           0.96   0.458282  0.770337             0.428571         2.633919           1.000000       0.415921  3.370546