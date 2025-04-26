# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from tensorflow.python.keras.models import Model
from model.dssm import DSSM
from dssm_preprocess import gen_data_set, gen_model_input, gen_test_inpout
import numpy as np
from tqdm import tqdm
from utils import recall_N
from dssm_config_argparse import *

'''
    StructField("user_id", StringType()),
    StructField("song_id", StringType()),
    StructField("server_timestamp", IntegerType()),
    StructField("user_level", IntegerType()),
    StructField("user_language", StringType()),
    StructField("platform", StringType()),
    # StructField("province", StringType()),
    StructField("country", StringType()),
    StructField("gender", IntegerType()),
    # StructField("song_language", StringType()),
    StructField("artist_gender", IntegerType()),
    StructField("age", IntegerType()),
    StructField("is_new", IntegerType()),
    StructField("song_quality", FloatType()),
    StructField("song_recording_count", IntegerType()),
    StructField("sgenres", StringType())
'''

if __name__ == "__main__":
    data_file = FLAGS.data_dir
    SEQ_LEN = FLAGS.seq_len
    min_count = FLAGS.min_count
    negsample = FLAGS.negsample
    embedding_dim = FLAGS.embedding_dim
    batch_size = FLAGS.batch_size
    epoch = FLAGS.epochs
    validation_split = FLAGS.validation_split
    user_dnn_hidden_units = user_hidden_unit
    item_dnn_hidden_units = item_hidden_unit

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    data = pd.read_csv(data_file, names=data_cloums, sep=',').head(2000)
    # print(data.head(5))
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
    # print(data.head())
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

    item_profile = data[item_fearther].drop_duplicates('sm_id')  # 物品特征，去重
    item_profile.set_index("sm_id", inplace=True)
    # print(item_profile.head())

    item_ids = data[["sm_id"]].drop_duplicates('sm_id')
    # print(item_ids.head())

    train_set, test_set, history_items, hot_ids = gen_data_set(data, min_count, negsample)  # 获取训练集和测试集，训练集和测试集采样方法可修改

    # 处理后的特征以字典key: array的结构存储
    train_model_input, train_label = gen_model_input(train_set, user_profile, item_profile, SEQ_LEN)
    test_user_input, test_item_input, test_label = gen_test_inpout(test_set, user_profile, item_profile, item_ids,
                                                                   SEQ_LEN)

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
    all_item_model_input = test_item_input

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(item_embs.shape[1])
    user_embs_size = user_embs.shape[1]

    # 5. [Optional] ANN search by faiss  and evaluate the result

    test_true_label = {line[0]: [line[2]] for line in test_set}
    #


    import faiss
    index = faiss.IndexFlatIP(user_embs_size)
    # faiss.normalize_L2(item_embs)
    index.add(item_embs)
    # faiss.normalize_L2(user_embs)
    D, I = index.search(user_embs, FLAGS.pred_topk)

    hot_10 = []
    hot_50 = []
    hot_hit = 0
    hot_count = 0
    recall_50 = []
    recall_10 = []
    hit = 0
    print(len(test_user_model_input['user_id']))
    print(item_ids['sm_id'].values[I[0][0]])
    # f = open(FLAGS.save_dir, 'w')

    c_top100 = 0
    for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
        try:
            pred = [item_ids['sm_id'].values[x] for x in I[i]]
            #print(1)
            filter_item = None
            rec_10 = recall_N(test_true_label[uid], pred, N=10)
            rec_50 = recall_N(test_true_label[uid], pred, N=FLAGS.recall_topk)
            # print(2)

            # recall_10.append(rec_10)
            # recall_50.append(rec_50)
            # if test_true_label[uid] in pred:
            #    hit += 1

            for lab in test_true_label[uid]:
                if lab in hot_ids:
                    hot_10.append(rec_10)
                    hot_50.append(rec_50)
                    hot_count += 1
                else:
                    recall_10.append(rec_10)
                    recall_50.append(rec_50)
            # print(1)
            for lab in test_true_label[uid]:
                if lab in pred:
                    if lab in hot_ids:
                        hot_hit += 1
                    else:
                        hit += 1
            '''
            his_item = set(history_items[uid])
            pred = [item_ids['sm_id'].values[x] - 1 for x in I[i] if item_ids['sm_id'].values[x] not in his_item]
            if len(pred) < 100:
                c_top100 += 1
            user_orgin = encoder[0].inverse_transform([uid - 1])   # 特征处理之前的uid
            item_orgin = [str(x) for x in encoder[1].inverse_transform(pred)]    # 特征处理之前的item_id
            f.write(str(user_orgin[0]) + "\t" + ','.join(item_orgin) + '\n')
            # break
            '''
        except Exception as e:
            print(i, e)
            continue
            # break
    # f.close()

    print("hot_10", np.mean(hot_10))
    print("hot_50", np.mean(hot_50))
    print("hr", hit / max(1, hot_count))
    print(hot_count)

    print("recall_10", np.mean(recall_10))
    print("recall_50", np.mean(recall_50))
    print("hr", hit / len(test_user_model_input['user_id']))
    print("pret less 100: ", c_top100)






