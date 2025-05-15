# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ------------------------------------------------------------------------------
# -- author：张明阳
# -- create：2025年5月8日00:46:53
# -- function：DSSM-练习版本V3-faiss查询用户相似的item+DeepCTR
# -- document:
# ------------------------------------------------------------------------------

import argparse
import random
import faiss
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from deepctr.feature_column import build_input_features, input_from_feature_columns, SparseFeat, VarLenSparseFeat
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.utils import combined_dnn_input

import pandas as pd
import numpy as np

# 设置显示选项：不省略列、不省略行、不截断内容
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_colwidth', None)  # 显示每列完整内容
pd.set_option('display.expand_frame_repr', False)  # 不自动换行显示DataFrame

import matplotlib.pyplot as plt

from DSSMModel_FuncApi import DSSM

def cosine_similarity(a, b):
    """
    计算两个向量之间的余弦相似度
    a, b: shape=(batch_size, embedding_dim)
    返回 shape=(batch_size, 1)
    """
    a = tf.nn.l2_normalize(a, axis=1)  # 对每个向量按行归一化
    b = tf.nn.l2_normalize(b, axis=1)
    return tf.reduce_sum(tf.multiply(a, b), axis=1, keepdims=True)





def parse_args():
    parser = argparse.ArgumentParser(description="DSSM Recommendation System Parameters")

    parser.add_argument("--data_dir", type=str, default=r"D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\dssm_data_2.csv",help="原始输入数据路径")
    parser.add_argument("--data_final_dir", type=str,default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_finash.csv", help="最终处理后的数据路径")

    parser.add_argument("--seq_len", type=int, default=15, help="用户历史序列的最大长度")
    parser.add_argument("--min_count", type=int, default=5, help="商品被点击的最小次数（过滤低频）")
    parser.add_argument("--negsample", type=int, default=3, help="负采样的数量")
    parser.add_argument("--embedding_dim", type=int, default=30, help="Embedding 向量维度")
    parser.add_argument("--batch_size", type=int, default=256, help="训练的批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--validation_split", type=float, default=0.0, help="验证集划分比例")
    parser.add_argument("--layer_embeding", type=int, default=32, help="模型中间层嵌入维度")
    parser.add_argument("--pred_topk", type=int, default=200, help="召回预测时选取的 Top-K 数量")
    parser.add_argument("--recall_topk", type=int, default=3, help="评估时的 Top-K 召回覆盖率")

    parser.add_argument("--save_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_u2i.txt",help="主召回结果保存路径")
    parser.add_argument("--save_dir_new", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_new.txt",help="新召回结果保存路径")
    parser.add_argument("--save_sdm_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/sdm_data_u2i.txt",help="SDM 模型召回结果保存路径")
    parser.add_argument("--save_mind_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/mind_data_u2i.txt",help="MIND 模型召回结果保存路径")
    parser.add_argument("--save_final_dir", type=str,default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_final_u2i.txt", help="最终合并召回结果保存路径")

    # ✅ 关键：让 argparse 忽略 Jupyter 注入的无关参数
    # Jupyter 会自动向 argparse 传入 notebook 的内部参数（比如 -f kernel-xxx.json），而你没有设置接受这些参数，所以报错
    args, _ = parser.parse_known_args()
    return args


def gen_data_set(data, min_count, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data[['sm_id']]
    # 统计sm_id 出现的次数, 转为dataframe
    item_ids = item_ids['sm_id'].value_counts().rename_axis('sm_id').reset_index(name='num')
    print(item_ids)
    item_ids = item_ids.loc[item_ids['num'] > min_count]
    print("-----------min_count-----------")
    print(min_count)

    # 最终，这一行得到的是：点击量比“平均热度阈值”还要高的物品行，即“热门物品”。
    # hot_ids = item_ids.loc[item_ids['num'] > int(item_ids["num"].sum() / (int(item_ids.shape[0] * 0.1)))]
    # hot_ids = set(hot_ids['sm_id'].unique())

    hot_ids = []

    item_nums = item_ids['num'].to_list()  # .div(item_ids['num'].sum())
    item_ids = item_ids['sm_id'].unique()

    train_set = []
    test_set = []
    count = 0
    user_items = {}
    print(data.groupby('user_id'))
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        # print(reviewerID, hist)
        pos_list = hist['sm_id'].tolist()  # 用户浏览过的item_id
        pos_list = pos_list[:min(100, len(pos_list))]
        user_items[reviewerID] = pos_list
        neg_list = set()
        if negsample > 0:  # 全局负采样
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)

        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                # 用户ID + 浏览记录倒排 + 当前itemID + 正负样本label（1，0） + 浏览记录的长度 + item评分（正样本有，负样本没有）
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), 1))
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + negi], 0, len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), 1))
        count += 1
    # 打散
    # random.shuffle(train_set)
    # random.shuffle(test_set)

    return train_set, test_set, user_items, hot_ids


def gen_model_input(train_set, user_profile, item_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    # padding
    train_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=seq_max_len, padding='post',
                                                                  truncating='post', value=0)
    # print(train_seq_pad)
    train_model_input = {"user_id": train_uid, "sm_id": train_iid, "hist_sm_id": train_seq_pad,
                         "hist_len": train_hist_len}

    for key in user_fearther_noid:
        try:
            train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values
        except:
            continue

    for key in item_fearther_noid:
        try:
            train_model_input[key] = item_profile.loc[train_model_input['sm_id']][key].values
        except:
            continue

    return train_model_input, train_label




if __name__ == "__main__":
    args = parse_args()
    data_file = args.data_dir
    SEQ_LEN = args.seq_len
    min_count = args.min_count
    negsample = args.negsample
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    epoch = args.epochs
    validation_split = args.validation_split

    user_hidden_unit = (64, 32)
    item_hidden_unit = (64, 32)

    user_dnn_hidden_units = user_hidden_unit
    item_dnn_hidden_units = item_hidden_unit

    # publish 训练特征
    data_cloums = ["user_id", "sm_id", "timestamp", "level", "user_lang", "platform", "country",
                   "gender", "age", "song_lang_id", "artist_gender", "song_quality", "song_recording_count",
                   "song_genres", "song_create_time", "artist_country_id", "region", "is_new"]

    features = ["user_id", "sm_id", "level", "user_lang", "platform",
                "country", "gender", "artist_gender", "age", "is_new",
                "song_quality", "song_recording_count", "song_genres"]

    num_header = ["level", "gender", "age", "artist_gender", "is_new", "song_quality", "song_recording_count"]
    string_header = ["user_id", "sm_id", "user_lang", "platform", "country", "song_genres"]
    time_header = ["timestamp"]
    user_fearther = ["user_id", "gender", "age", "level", "user_lang", "country", "platform", "is_new"]
    item_fearther = ["sm_id", "artist_gender", "song_quality", "song_recording_count","song_genres"]  # "sm_language" 90% 为空，去掉
    user_fearther_noid = ["gender", "age", "level", "user_lang", "country", "platform", "is_new"]
    item_fearther_noid = ["artist_gender", "song_quality", "song_recording_count", "song_genres"]

    # 录制训练特征
    final_col = ["user_id", "sm_id", "timestamp", "level", "user_lang", "platform", "country",
                 "gender", "age", "song_lang_id", "artist_gender", "song_quality", "song_recording_count",
                 "song_genres", "song_create_time", "artist_country_id"]

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    data = pd.read_csv(data_file, names=data_cloums, sep=',').head(2000)
    print(data.head(5))
    # print(data["song_recording_count"])

    for header in num_header:
        temp = data[header].fillna(-1)
        data[header] = temp
    for header in string_header:
        temp = data[header].fillna("unknow")
        data[header] = temp
    for header in time_header:
        temp = data[header].fillna(data[header].min())
        data[header] = temp
    print(data.head())
    # print(data["song_recording_count"])

    feature_max_idx = {}
    encoder = []
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1  # 特征预处理
        feature_max_idx[feature] = data[feature].max() + 1  # 处理后特征的最大值，用来记录特征的数量
        encoder.append(lbe)
    # print(encoder)
    # print(data[features].head())

    user_profile = data[user_fearther].drop_duplicates('user_id')  # 用户特征，drop_duplicates 去重
    user_profile.set_index("user_id", inplace=True)  # 将user_id列转为索引，inplace表示在原数据上修改
    print(user_profile.head())

    item_profile = data[item_fearther].drop_duplicates('sm_id')  # 物品特征，去重
    item_profile.set_index("sm_id", inplace=True)
    print(item_profile.head())

    item_ids = data[["sm_id"]].drop_duplicates('sm_id')
    print(item_ids.head())

    train_set, test_set, history_items, hot_ids = gen_data_set(data, min_count, negsample)  # 获取训练集和测试集，训练集和测试集采样方法可修改
    print(train_set)
    print(test_set)
    print(history_items)
    print(hot_ids)
    # [(11, [44], 97, 1, 1, 1), (11, [97, 44], 97, 1, 2, 1), (11, [97, 97, 44], 59, 1, 3, 1), (11, [59, 97, 97, 44], 63, 1, 4, 1), (175, [82], 91, 1, 1, 1), (175, [91, 82], 120, 1, 2, 1), (175, [120, 91, 82], 98, 1, 3, 1), (175, [98, 120, 91, 82], 45, 1, 4, 1)]
    # [(11, [63, 59, 97, 97, 44], 90, 1, 5, 1), (36, [85], 85, 1, 1, 1), (37, [72], 72, 1, 1, 1), (74, [103], 103, 1, 1, 1), (148, [18], 18, 1, 1, 1), (159, [30], 30, 1, 1, 1), (165, [48], 48, 1, 1, 1), (172, [66], 66, 1, 1, 1), (175, [45, 98, 120, 91, 82], 45, 1, 5, 1), (182, [26], 26, 1, 1, 1), (186, [61], 61, 1, 1, 1)]
    # {1: [120], 2: [106], 3: [99], 4: [84], 5: [122], 6: [123], 7: [80], 8: [117], 9: [72], 10: [101], 11: [44, 97, 97, 59, 63, 90], 12: [121], 13: [116], 14: [109], 15: [101], 16: [118], 17: [119], 18: [83], 19: [102], 20: [107], 21: [106], 22: [84], 23: [113], 24: [108], 25: [99], 26: [105], 27: [87], 28: [103], 29: [75], 30: [93], 31: [91], 32: [108], 33: [74], 34: [76], 35: [44], 36: [85, 85], 37: [72, 72], 38: [113], 39: [86], 40: [71], 41: [90], 42: [54], 43: [98], 44: [109], 45: [89], 46: [112], 47: [115], 48: [100], 49: [43], 50: [51], 51: [88], 52: [114], 53: [93], 54: [105], 55: [21], 56: [107], 57: [34], 58: [104], 59: [106], 60: [113], 61: [96], 62: [101], 63: [63], 64: [79], 65: [109], 66: [113], 67: [78], 68: [113], 69: [95], 70: [82], 71: [2], 72: [77], 73: [113], 74: [103, 103], 75: [73], 76: [110], 77: [98], 78: [99], 79: [113], 80: [70], 81: [92], 82: [101], 83: [87], 84: [80], 85: [113], 86: [77], 87: [111], 88: [84], 89: [109], 90: [90], 91: [94], 92: [99], 93: [81], 94: [6], 95: [7], 96: [37], 97: [63], 98: [69], 99: [44], 100: [48], 101: [68], 102: [49], 103: [17], 104: [45], 105: [34], 106: [67], 107: [52], 108: [24], 109: [4], 110: [42], 111: [20], 112: [8], 113: [45], 114: [59], 115: [53], 116: [45], 117: [29], 118: [56], 119: [58], 120: [65], 121: [59], 122: [43], 123: [39], 124: [34], 125: [33], 126: [44], 127: [49], 128: [5], 129: [47], 130: [11], 131: [44], 132: [39], 133: [27], 134: [17], 135: [11], 136: [22], 137: [62], 138: [55], 139: [44], 140: [32], 141: [57], 142: [59], 143: [36], 144: [41], 145: [1], 146: [45], 147: [50], 148: [18, 18], 149: [46], 150: [16], 151: [3], 152: [7], 153: [59], 154: [32], 155: [59], 156: [60], 157: [59], 158: [27], 159: [30, 30], 160: [44], 161: [34], 162: [35], 163: [6], 164: [48], 165: [48, 48], 166: [25], 167: [14], 168: [43], 169: [47], 170: [40], 171: [38], 172: [66, 66], 173: [23], 174: [28], 175: [82, 91, 120, 98, 45, 45], 176: [11], 177: [9], 178: [32], 179: [7], 180: [63], 181: [31], 182: [26, 26], 183: [63], 184: [11], 185: [12], 186: [61, 61], 187: [44], 188: [43], 189: [10], 190: [38], 191: [19], 192: [13], 193: [15], 194: [64], 195: [44], 196: [59]}
    # []




    # 处理后的特征以字典key: array的结构存储
    train_model_input, train_label = gen_model_input(train_set, user_profile, item_profile, SEQ_LEN)
    print("------------数据处理之后的结果:------------------")
    print(train_model_input)
    print(train_label)

    import pandas as pd
    import numpy as np

    # 设置显示选项：不省略列、不省略行、不截断内容
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_colwidth', None)  # 显示每列完整内容
    pd.set_option('display.expand_frame_repr', False)  # 不自动换行显示DataFrame



    import pandas as pd
    import numpy as np

    import pandas as pd
    import numpy as np

    # 拷贝一份输入字典，避免原地修改
    flat_input = {}

    for k, v in train_model_input.items():
        # 如果是二维数组（如hist_sm_id），将每一行转成列表或字符串
        if isinstance(v[0], (np.ndarray, list)):
            flat_input[k] = [list(x) for x in v]  # 或者: [str(list(x)) for x in v]
        else:
            flat_input[k] = v

    # 添加标签
    flat_input['label'] = train_label

    # 转为 DataFrame
    df_input = pd.DataFrame(flat_input)

    # 如果标签是 ndarray，转成 DataFrame 列
    df_input['label'] = train_label

    # 显示结果
    print(df_input.head())

    # train_model_input, train_label               = gen_model_input(train_set, user_profile, item_profile, SEQ_LEN)
    test_user_input, test_item_input, test_label = gen_test_inpout(test_set, user_profile, item_profile, item_ids,SEQ_LEN)

    # # 拷贝一份输入字典，避免原地修改
    # flat_input = {}
    #
    # for k, v in test_user_input.items():
    #     # 如果是二维数组（如hist_sm_id），将每一行转成列表或字符串
    #     if isinstance(v[0], (np.ndarray, list)):
    #         flat_input[k] = [list(x) for x in v]  # 或者: [str(list(x)) for x in v]
    #     else:
    #         flat_input[k] = v
    #
    # # 添加标签
    # flat_input['label'] = train_label
    #
    # # 转为 DataFrame
    # df_input = pd.DataFrame(flat_input)
    # #
    # # # 如果标签是 ndarray，转成 DataFrame 列
    # # df_input['label'] = train_label
    # print(df_input)



    # 如果某个用户只浏览过 1 个 item，那：pos_list = [item1]  range(1, len(pos_list)) 相当于 range(1, 1)，是空的
    # 所以既不会生成训练样本，也不会生成测试样本
    # 这个用户会被完全跳过，也不会出现在 train_set、test_set 或 user_items 中


    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    # 特征转化为SparseFeat
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
                            SparseFeat("age", feature_max_idx['age'], embedding_dim),
                            SparseFeat("level", feature_max_idx['level'], embedding_dim),
                            SparseFeat("user_lang", feature_max_idx['user_lang'], embedding_dim),
                            SparseFeat("country", feature_max_idx['country'], embedding_dim),
                            SparseFeat("platform", feature_max_idx['platform'], embedding_dim),
                            SparseFeat("is_new", feature_max_idx['is_new'], embedding_dim),
                            VarLenSparseFeat(SparseFeat('hist_sm_id', feature_max_idx['sm_id'], embedding_dim,
                                                        embedding_name="sm_id"), SEQ_LEN, 'mean', 'hist_len'),
                            ]

    item_feature_columns = [SparseFeat('sm_id', feature_max_idx['sm_id'], embedding_dim),
                            # SparseFeat("sm_language", feature_max_idx['sm_language'], embedding_dim),
                            SparseFeat("artist_gender", feature_max_idx['artist_gender'], embedding_dim),
                            SparseFeat("song_quality", feature_max_idx['song_quality'], embedding_dim),
                            SparseFeat("song_recording_count", feature_max_idx['song_recording_count'], embedding_dim),
                            SparseFeat("song_genres", feature_max_idx['song_genres'], embedding_dim),
                            ]

    # 3.Define Model and train
    model = DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=user_dnn_hidden_units,
                 item_dnn_hidden_units=item_dnn_hidden_units)

    model.compile(optimizer='adagrad', loss="binary_crossentropy")

    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=batch_size, epochs=epoch, verbose=1, validation_split=validation_split, )


    # 4. Generate user features for testing and full item features for retrieval
    test_user_model_input = test_user_input
    test_item_model_input  = test_item_input
    print("------------->test_user_input-------------->")
    print(test_user_input)
    print("=============test_item_input============")
    print(test_item_input)


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
    # print(user_embedding_model.summary())

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(test_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)  # (11, 32)
    print(item_embs.shape)  # (123, 32)



    # 1. Faiss 向量召回
    index = faiss.IndexFlatIP(item_embs.shape[1])
    index.add(item_embs)
    # 用每一个用户的 embedding 向量，去检索向量数据库中最相似的 top-k 个 item，返回它们的索引（I）和相似度（D）
    """
    IndexFlatIP 表示使用 Inner Product（内积） 来衡量相似度，适用于你模型输出 embedding 已经过 normalize（或你使用的是点积打分）时。
    D 是每个用户的 top-k 相似 item 的相似度分值（内积值）。
    I 是这些 item 的索引（对应 item_embs 中的下标）。
    """
    D, I = index.search(user_embs, args.recall_topk)
    print("6666666666666666666666")
    print(D.shape)  # (11, 3)
    print(I.shape)  # (11, 3)
    print(D)
    print(I)
    """
    [[-7.26867802e-05 -8.13217339e-05 -8.24873350e-05]
     [-4.06468607e-05 -4.47547391e-05 -4.61710661e-05]
     [-4.46781341e-05 -5.22108894e-05 -5.32776830e-05]
     [-3.01159944e-05 -3.51995986e-05 -3.61024395e-05]
     [-3.22540400e-05 -3.68425317e-05 -3.80667188e-05]
     [-7.35251670e-05 -1.00781705e-04 -1.02940350e-04]
     [-3.19964274e-05 -3.53781688e-05 -3.64529587e-05]
     [-3.41866835e-05 -3.55332704e-05 -3.55515112e-05]
     [-1.29403328e-04 -1.35532464e-04 -1.91821629e-04]
     [-5.93553741e-05 -8.00634443e-05 -8.03681542e-05]
     [-3.20936233e-05 -3.44605469e-05 -3.62912760e-05]]
    [[88  4 68]
     [88 68  4]
     [88 68  4]
     [88 68 93]
     [88 68 93]
     [88 68  4]
     [88 68 93]
     [68 88 93]
     [ 9 88 32]
     [88 68 32]
     [88 68 93]]
    """




    # 2. 构造真实标签
    for item in test_set:
        print(item)
    test_true_label = {line[0]: [line[2]] for line in test_set}
    print(test_true_label)
    """
    (11, [63, 59, 97, 97, 44], 90, 1, 5, 1)
    (36, [85], 85, 1, 1, 1)
    (37, [72], 72, 1, 1, 1)
    (74, [103], 103, 1, 1, 1)
    (148, [18], 18, 1, 1, 1)
    (159, [30], 30, 1, 1, 1)
    (165, [48], 48, 1, 1, 1)
    (172, [66], 66, 1, 1, 1)
    (175, [45, 98, 120, 91, 82], 45, 1, 5, 1)
    (182, [26], 26, 1, 1, 1)
    (186, [61], 61, 1, 1, 1)
    
    # 获取每个用户实际点击的item列表
    {11: [90], 36: [85], 37: [72], 74: [103], 148: [18], 159: [30], 165: [48], 172: [66], 175: [45], 182: [26], 186: [61]
    """


    # # 3. 计算 recall
    recall_10, recall_50 = [], []
    for i, uid in enumerate(test_user_model_input['user_id']):
        # print(i,uid)
        # print(item_ids['sm_id'])
        """
        这行代码的作用是：
        对于每个用户 i 的推荐物品索引 I[i]，将这些索引映射为物品的 ID（从 item_ids 中提取），形成最终的推荐列表 pred。
        
        举个例子，如果 I[i] = [2, 5, 8]，那么 pred 就是 [1003, 1006, 1009]，即推荐给用户 i 的物品 ID。
        
        希望这个解释清楚了！如果还有不明白的地方，随时告诉我。
        """
        # pred = [item_ids['sm_id'].values[x] for x in I[i]]
        pred = item_ids['sm_id'].iloc[I[i]].tolist()

        """
        recall_N 是一个计算 召回率 的函数。召回率的定义是：
        Recall=推荐结果中包含的实际点击物品数实际点击的物品总数
        Recall=实际点击的物品总数推荐结果中包含的实际点击物品数​
        
        具体来说，recall_N(test_true_label[uid], pred, N=10) 计算的是用户 uid 在推荐列表 pred 中的前 N=10 个物品中，实际点击物品 test_true_label[uid] 的 召回率。
        """
        rec_10 = recall_N(test_true_label[uid], pred, N=10)
        rec_50 = recall_N(test_true_label[uid], pred, N=50)
        recall_10.append(rec_10)
        recall_50.append(rec_50)

    print("recall@10:", np.mean(recall_10))
    print("recall@50:", np.mean(recall_50))

