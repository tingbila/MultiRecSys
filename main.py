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



if __name__ == "__main__":
    # 加载数据集 （小数据集是用逗号分割单 ，大数据集是用\t分割的）
    data, train_ds, valid_ds,test_ds, feat_columns  = create_dataset(file_path="data_files/train_2.csv", embed_dim=embed_dim)
    # {'tokenizers': {'actors': <keras_preprocessing.text.Tokenizer object at 0x0000029E0D126250>, 'genres': <keras_preprocessing.text.Tokenizer object at 0x0000029E0D126E50>}, 'pad_len_dict': {'actors': 2, 'genres': 2}}

    # 打印整个 batch 数据（可根据实际需要调整显示内容）
    # 仅取出第一个 batch 并退出循环
    # for item in train_ds:
    #     break
    #
    #     # 打印数据结构的分隔线，并增加详细的输出信息
    # print('====================================================')
    # print('取出的第一个 batch 数据如下：')
    # print('----------------------------------------------------')
    # print(item)

    # 构建模型
    # 1. 调用FM模型-最早学的那个版本
    # model = Fm(feat_columns,embed_dim)

    # 2. 调用FM模型-Embedding版本
    # model = FM_MTL(feat_columns,embed_dim)

    # 3. 调用Wide&Deep模型
    # model = WideAndDeep(feat_columns,embed_dim)

    # 4. 调用DeepFM模型
    model = DeepFM_MTL(feat_columns,embed_dim)
    print(model)

    # 5. 调用XDeepFM模型
    # model = XDeepFM_MTL(feat_columns,embed_dim,cin_layers=[7,15])

    # 6. 调用XDeepFM + Transform_Attention模型
    # finish_accuracy: 0.7333 - finish_auc: 0.8978 - finish_loss: 0.6232 - like_accuracy: 0.6000 - like_auc: 0.0000e+00 - like_loss: 0.6832 - loss: 1.3064
    # model = XDeepFM_Transform_MTL(feat_columns,embed_dim,cin_layers=[7,15])

    # 7. 调用DCN CrossNetwork 网络
    # model = DCN_Model_MTL(feat_columns,embed_dim,cin_layers=[7,15])

    # 8. 调用XDeepFM +  Transform_Attention + DCN_Attentiion模型（这是论文创新的代码，工业上跳过这个）
    # DCN:          finish_accuracy: 0.5000 - finish_auc: 0.4178 - finish_loss: 0.8678 - like_accuracy: 1.0000 - like_auc: 0.0000e+00 - like_loss: 0.1547 - loss: 1.0225
    # DCN_Attention:finish_accuracy: 0.5000 - finish_auc: 0.5000 - finish_loss: 0.6931 - like_accuracy: 1.0000 - like_auc: 0.0000e+00 - like_loss: 0.0504 - loss: 0.7435
    # model = XDeepFM_Transform_DCN_CrossNetwork_Attention_MTL(feat_columns,embed_dim,cin_layers=[7,15])

    # 8. 调用DeepCrossing_Residual模型
    # model = DeepCrossing_Residual(feat_columns,embed_dim,hidden_units=[128, 64, 32])

    # 9. 调用NFM模型
    # model = NFM(feat_columns,embed_dim,batch_size)

    # 10. 调用AFm模型
    # model = AFm(feat_columns,embed_dim)

    # 11. 调用AFM_Embedding模型
    # model = AFM_Embedding(feat_columns,embed_dim)

    # 12. 调用MMOE模型
    # loss: 0.6485 - finish_loss: 0.6384 - like_loss: 0.0101 - finish_auc: 0.4112 - finish_accuracy: 0.6780 - like_auc: 0.0000e+00 - like_accuracy: 1.0000
    # model = MMOE(feat_columns=feat_columns, embed_dim=5)

    # 13. 调用DeepFM模型 + 含有序列Sequence数据
    # model = DeepFM_Sequence_MTL(feat_columns,embed_dim)

    # 训练并评估
    train_and_evaluate(model, train_ds, valid_ds,test_ds)
    
    #    finish_accuracy  finish_loss  like_accuracy  like_loss      loss  val_finish_accuracy  val_finish_loss  val_like_accuracy  val_like_loss  val_loss
    # 0             0.68     0.633306           0.68   0.675442  1.289731             0.571429         0.790080           0.714286       0.682514  1.501765
    # 1             0.68     0.477380           0.88   0.637175  1.134881             0.428571         0.981986           1.000000       0.629791  1.696161
    # 2             0.80     0.465747           0.96   0.586718  0.986695             0.428571         1.355031           1.000000       0.575927  2.081305
    # 3             0.80     0.369490           0.96   0.518443  0.916458             0.428571         1.668533           1.000000       0.529664  2.387516
    # 4             0.80     0.352170           0.96   0.497951  0.855214             0.428571         2.065776           1.000000       0.476627  2.785610
    # 5             0.80     0.308622           0.96   0.458282  0.770337             0.428571         2.633919           1.000000       0.415921  3.370546