# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media
# 召回任务


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
from sklearn.metrics import log_loss, roc_auc_score

import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个

from tensorflow.keras.models import Model
from dssm_models import DSSM
from config import *
from utils import *
import faiss



if __name__ == '__main__':
    # 读取配置信息
    args = parse_args()

    # === 1. 读取数据并进行特征处理 ===
    data = pd.read_csv(args.data_dir, sep='\t',names=column_names)
    data = process_dense_feats(data, dense_feats)
    data = process_sparse_feats(data, sparse_feats)
    pad_sequences_dict, tokenizers, pad_len_dict = process_sequence_feats(data, sequence_feats) if sequence_feats else ({}, {}, {})


    # === 2. 构建DeepCTR特征列 ===
    fixlen_feature_columns = [
        SparseFeat(feat, data[feat].nunique() + 1, embedding_dim=args.embedding_dim) for feat in sparse_feats
    ] + [
        DenseFeat(feat, 1) for feat in dense_feats
    ]
    # maxlen:每个序列特征被 padding 到的最大长度，确保输入模型的 shape 一致
    # combiner='mean':指定如何将序列嵌入表示“压缩”为一个固定长度向量
    # 这段代码是告诉 DeepCTR：这些特征是变长的 token 序列，每个 token 有 embedding，用 mean 聚合序列表示，最终作为模型的输入
    if sequence_feats:
        for feat in sequence_feats:
            fixlen_feature_columns.append(
                VarLenSparseFeat(
                    SparseFeat(feat, vocabulary_size=len(tokenizers[feat].word_index) + 1, embedding_dim=args.embedding_dim),
                    maxlen=pad_len_dict[feat], combiner='mean')
            )
    # fixlen_feature_names就是sparse_feats、dense_feats、sequence_feats特征列表的总和
    fixlen_feature_names  = get_feature_names(fixlen_feature_columns)


    # === 3. 划分训练集和测试集 ===
    train, test = train_test_split(data, test_size=args.test_split, random_state=2018)
    # 构建模型输入
    train_model_input = {name: train[name] for name in fixlen_feature_names}
    test_model_input  = {name: test[name] for name in fixlen_feature_names}
    # 添加序列特征到输入
    # 把提前 pad 好的序列特征（如 genres、actors），按照 train/test 划分后的索引，分成训练和测试输入字典
    if sequence_feats:
        for feat in sequence_feats:
            train_model_input[feat] = pad_sequences_dict[feat][train.index]
            test_model_input[feat]  = pad_sequences_dict[feat][test.index]


    # === 4. DSSM模型构建和训练 ===
    # 4.1 初始化用于存储用户侧和物品侧的特征列-DeepCTR需要
    user_feature_columns = []
    item_feature_columns = []

    # 遍历所有构建好的特征列（包括 SparseFeat, DenseFeat 和 VarLenSparseFeat）
    for feat in fixlen_feature_columns:
        # 如果是变长稀疏特征（如序列类：actors、genres），特征名嵌套在 sparsefeat.name 中
        if isinstance(feat, VarLenSparseFeat):
            feat_name = feat.sparsefeat.name
        else:
            # 否则是普通的 SparseFeat 或 DenseFeat，直接使用 feat.name
            feat_name = feat.name
        # 根据字段名判断该特征属于用户侧还是物品侧
        if feat_name in user_tower_feature_columns:
            user_feature_columns.append(feat)  # 加入用户特征列表
        elif feat_name in item_tower_feature_columns:
            item_feature_columns.append(feat)  # 加入物品特征列表


    # 4.2 初始化用于存储用户侧和物品侧的输入数据-DeepCTR需要
    # 从训练输入中提取DSSM输入字段
    train_dssm_model_input = {
        name: train_model_input[name] for name in train_model_input if name in user_tower_feature_columns + item_tower_feature_columns
    }
    # 从测试输入中提取DSSM输入字段
    test_dssm_model_input = {
        name: test_model_input[name] for name in test_model_input if name in user_tower_feature_columns + item_tower_feature_columns
    }

    # 4.3 定义模型并进行训练
    model = DSSM(user_feature_columns, item_feature_columns)
    model.compile(optimizer='adagrad', loss="binary_crossentropy",metrics=['accuracy', keras.metrics.AUC(name='auc')])
    history = model.fit(train_dssm_model_input, train[target],batch_size=args.batch_size, epochs=args.epochs, verbose=1, validation_split= args.validation_split)
    print(pd.DataFrame(history.history))
    """
           loss  accuracy       auc  val_loss  val_accuracy   val_auc
    0  0.711999  0.478723  0.451893  0.704628      0.531915  0.507663
    1  0.664077  0.627660  0.577217  0.701667      0.510638  0.519157
    2  0.654718  0.627660  0.628161  0.699707      0.510638  0.526820
    3  0.647655  0.622340  0.662205  0.698252      0.510638  0.541188
    4  0.641807  0.622340  0.685457  0.696981      0.510638  0.543103
    5  0.636258  0.627660  0.704145  0.695832      0.510638  0.542146
    6  0.630556  0.643617  0.720735  0.694837      0.531915  0.538314
    7  0.624379  0.664894  0.735722  0.694041      0.531915  0.532567
    8  0.617334  0.670213  0.754471  0.693513      0.531915  0.534483
    9  0.608849  0.680851  0.779512  0.693366      0.531915  0.533525
    """

    # 4.4 可视化训练过程
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['loss'], label='Train Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['auc'], label='Train AUC')
    plt.plot(history.epoch, history.history['val_auc'], label='Val AUC - Like')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


    # 4.5 在测试集test上面看一下效果
    preds = model.predict(test_dssm_model_input, batch_size=args.batch_size)
    print("预测结果样例:", preds[:5])  # 打印前5个样例
    print("Test LogLoss:", round(log_loss(test[target], preds), 4))
    print("Test AUC:", round(roc_auc_score(test[target], preds), 4))
    """
    预测结果样例: [[0.40200618]
    [0.44934118]
    [0.36049336]
    [0.42412487]
    [0.60881853]]
    Test LogLoss: 0.6811
    Test AUC: 0.5452
    """

    # === 5. 构建DSSM当中用户塔和item塔的子模型 ===
    """
    这行代码的含义是：
    用训练好的 model 中的某些中间层（在这里是用户输入和用户 DNN 向量输出）来创建一个新的子模型；
    inputs=model.user_input：指定新的子模型接收哪些输入（即用户特征）；
    outputs=model.user_embedding：指定新的子模型输出什么（即用户 DNN 向量）；
    相当于：构建一个从用户原始输入 → 用户 embedding 向量 的模型，用于推理。
    同理：
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)
    构建一个从物品原始输入 → 物品 embedding 向量的模型。
    """
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    # 这次直接抽取全部的数据，输出用户 embedding 向量 和  物品 embedding 向量
    # 模型训练完后，提取用户塔和物品塔的 embedding 向量，用于召回阶段（如 Faiss 向量检索、相似度计算、离线计算 TopN 推荐等）
    data_all_input = {name: data[name] for name in fixlen_feature_names}
    if sequence_feats:
        for feat in sequence_feats:
            data_all_input[feat] = pad_sequences_dict[feat]
    data_user_input = {
        name: data_all_input[name] for name in data_all_input if name in user_tower_feature_columns
    }
    data_item_input = {
        name: data_all_input[name] for name in data_all_input if name in item_tower_feature_columns
    }
    user_embeddings = user_embedding_model.predict(data_user_input, batch_size=2 ** 12)
    item_embeddings = item_embedding_model.predict(data_item_input, batch_size=2 ** 12)
    print("User Embedding:\n", user_embeddings.shape)  # (294, 32)
    print("Item Embedding:\n", item_embeddings.shape)  # (294, 32)


    # -------------------------- 6. 基于余弦相似度获取user的相似item(所有用户查询） --------------------------
    # Step 1: 向量归一化
    normalized_item_embeddings = l2_normalize(item_embeddings.astype(np.float32))
    normalized_user_embeddings = l2_normalize(user_embeddings.astype(np.float32))

    # Step 2: 创建 Faiss 内积索引
    dimension = normalized_item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 基于内积的索引（需配合归一化使用）

    # Step 3: 添加商品向量（已归一化）
    index.add(normalized_item_embeddings)

    # Step 4: 查询每个用户的 Top-K 相似商品
    queries = normalized_user_embeddings  # 形状为 [num_users, embedding_dim]
    k = args.recall_topk
    scores, item_indices = index.search(queries, k)  # 批量查询，返回 shape: [num_users, k]

    # Step 5: 输出结果:保存最终结果，用于记录每个用户对应的推荐商品
    item_ids = data_all_input['item_id']  # 获取真实 item_id 列表
    user_ids = data_all_input['uid']      # 获取真实 user_id 列表
    num_users = queries.shape[0]

    user2item_dict = {}
    for user_index in range(num_users):
        real_user_id = user_ids[user_index]  # 对应的真实用户ID
        print(f"\n用户索引 {user_index + 1}（真实用户ID: {real_user_id}）最相似的商品：")

        # 初始化当前用户的推荐列表
        top_items = []

        for i in range(k):
            index_in_item_ids = item_indices[user_index][i]
            similarity_score = scores[user_index][i]
            if index_in_item_ids == -1:
                print(f"Top {i + 1}: 无有效相似商品")
            else:
                real_item_id = item_ids[index_in_item_ids]
                print(f"Top {i + 1}: 向量编号 {index_in_item_ids} -> 真实商品ID {real_item_id}（余弦相似度: {similarity_score:.4f}）")
                # 将推荐的 item_id 和分数加入用户的推荐列表
                top_items.append({
                    "item_id": real_item_id,
                    "score": round(float(similarity_score), 4)  # 记得转为 float，否则写入 redis/json 时可能有问题
                })
        # 保存当前用户的推荐结果到字典
        user2item_dict[real_user_id] = top_items
    """
    用户索引 1（真实用户ID: 220）最相似的商品：
    Top 1: 向量编号 181 -> 真实商品ID 104（余弦相似度: 0.7554）
    Top 2: 向量编号 48 -> 真实商品ID 36（余弦相似度: 0.7553）
    Top 3: 向量编号 86 -> 真实商品ID 56（余弦相似度: 0.7333）
    Top 4: 向量编号 167 -> 真实商品ID 171（余弦相似度: 0.7246）
    Top 5: 向量编号 217 -> 真实商品ID 264（余弦相似度: 0.7240）

    用户索引 2（真实用户ID: 106）最相似的商品：
    Top 1: 向量编号 52 -> 真实商品ID 38（余弦相似度: 0.8144）
    Top 2: 向量编号 252 -> 真实商品ID 147（余弦相似度: 0.8025）
    Top 3: 向量编号 242 -> 真实商品ID 139（余弦相似度: 0.7950）
    Top 4: 向量编号 138 -> 真实商品ID 81（余弦相似度: 0.7877）
    Top 5: 向量编号 28 -> 真实商品ID 224（余弦相似度: 0.7790）
    """

    for user, items in user2item_dict.items():
        print(user, items)
    """
    220 [{'item_id': 104, 'score': 0.7554}, {'item_id': 36, 'score': 0.7553}, {'item_id': 56, 'score': 0.7333}, {'item_id': 171, 'score': 0.7246}, {'item_id': 264, 'score': 0.724}]
    106 [{'item_id': 38, 'score': 0.8144}, {'item_id': 147, 'score': 0.8025}, {'item_id': 139, 'score': 0.795}, {'item_id': 81, 'score': 0.7877}, {'item_id': 224, 'score': 0.779}]
    """
