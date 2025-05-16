# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm
from tensorflow import keras
import argparse

import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个




def parse_args():
    parser = argparse.ArgumentParser(description="DSSM Recommendation System Parameters")

    parser.add_argument("--data_dir", type=str, default=r"D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\train_2.csv",help="原始输入数据路径")
    parser.add_argument("--data_final_dir", type=str,default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_finash.csv", help="最终处理后的数据路径")

    parser.add_argument("--seq_len", type=int, default=15, help="用户历史序列的最大长度")
    parser.add_argument("--min_count", type=int, default=5, help="商品被点击的最小次数（过滤低频）")
    parser.add_argument("--negsample", type=int, default=3, help="负采样的数量")
    parser.add_argument("--embedding_dim", type=int, default=30, help="Embedding 向量维度")
    parser.add_argument("--batch_size", type=int, default=256, help="训练的批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--test_split", type=float, default=0.2, help="测试集划分比例")
    parser.add_argument("--validation_split", type=float, default=0.2, help="验证集划分比例")
    parser.add_argument("--layer_embeding", type=int, default=32, help="模型中间层嵌入维度")
    parser.add_argument("--pred_topk", type=int, default=200, help="召回预测时选取的 Top-K 数量")
    parser.add_argument("--recall_topk", type=int, default=5, help="评估时的 Top-K 召回覆盖率")

    parser.add_argument("--save_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_u2i.txt",help="主召回结果保存路径")
    parser.add_argument("--save_dir_new", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_new.txt",help="新召回结果保存路径")
    parser.add_argument("--save_sdm_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/sdm_data_u2i.txt",help="SDM 模型召回结果保存路径")
    parser.add_argument("--save_mind_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/mind_data_u2i.txt",help="MIND 模型召回结果保存路径")
    parser.add_argument("--save_final_dir", type=str,default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_final_u2i.txt", help="最终合并召回结果保存路径")

    # ✅ 关键：让 argparse 忽略 Jupyter 注入的无关参数
    # Jupyter 会自动向 argparse 传入 notebook 的内部参数（比如 -f kernel-xxx.json），而你没有设置接受这些参数，所以报错
    args, _ = parser.parse_known_args()
    return args


# csv数据当中的全部字段
column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like", "music_id", "device", "time", "duration_time", "actors", "genres"]

# csv当中的字段分类
sparse_feats = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
dense_feats  = ["time", "duration_time"]
sequence_feats = ['actors', 'genres']
target = ['finish']     # 推荐任务目标

# DSSM当中双塔的DNN层数信息
user_dnn_hidden_units = (64, 32)
item_dnn_hidden_units = (64, 32)

# DSSM双塔当中user塔和item塔的字段信息
user_tower_feature_columns = ["uid", "user_city", "device", "time", "duration_time"]
item_tower_feature_columns = ["item_id", "author_id", "item_city", "channel", "music_id", 'actors', 'genres']

