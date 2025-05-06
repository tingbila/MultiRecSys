import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from config.dssm_config import *


def gen_data_set_sdm(data, seq_short_len=5, seq_prefer_len=50):
    data.sort_values("timestamp", inplace=True)
    train_set = []
    test_set = []
    user_items = {}
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        # print(reviewerID)
        pos_list = hist['sm_id'].tolist()
        pos_list = pos_list[:min(100, len(pos_list))]
        user_items[reviewerID] = pos_list
        genres_list = hist['song_genres'].tolist()
        # rating_list = hist['rating'].tolist()
        # print(hist)
        # print(pos_list)
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            genres_hist = genres_list[:i]
            seq_short_len = min(i, seq_short_len)
            seq_prefer_len = min(max(i - seq_short_len, 0), seq_prefer_len)
            if i != len(pos_list) - 1:
                train_set.append(
                    (reviewerID, pos_list[i], 1, hist[::-1][:seq_short_len][::-1],
                     hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], seq_short_len,
                     seq_prefer_len, genres_hist[::-1][:seq_short_len][::-1],
                     genres_hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], 1))
            else:
                test_set.append(
                    (reviewerID, pos_list[i], 1, hist[::-1][:seq_short_len][::-1],
                     hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], seq_short_len,
                     seq_prefer_len, genres_hist[::-1][:seq_short_len][::-1],
                     genres_hist[::-1][seq_short_len:seq_short_len + seq_prefer_len], 1))
    # uid, 当前id , label, 短期点击item， 长期点击item, 短期点击序列长度， 长期点击序列长度， 短期点击类型， 长期点击类型， 分数
    random.shuffle(train_set)
    random.shuffle(test_set)
    # print(len(train_set[0]), len(test_set[0]))
    return train_set, test_set, user_items


def gen_model_input_sdm(train_set, user_profile, seq_short_max_len, seq_prefer_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_iid = np.array([line[1] for line in train_set])
    train_label = np.array([line[2] for line in train_set])
    short_train_seq = [line[3] for line in train_set]
    prefer_train_seq = [line[4] for line in train_set]
    train_short_len = np.array([line[5] for line in train_set])
    train_prefer_len = np.array([line[6] for line in train_set])
    short_train_seq_genres = np.array([line[7] for line in train_set])
    prefer_train_seq_genres = np.array([line[8] for line in train_set])

    train_short_item_pad = tf.keras.preprocessing.sequence.pad_sequences(short_train_seq, maxlen=seq_short_max_len, padding='post', truncating='post',
                                         value=0)
    train_prefer_item_pad = tf.keras.preprocessing.sequence.pad_sequences(prefer_train_seq, maxlen=seq_prefer_max_len, padding='post',
                                          truncating='post',
                                          value=0)
    train_short_genres_pad = tf.keras.preprocessing.sequence.pad_sequences(short_train_seq_genres, maxlen=seq_short_max_len, padding='post',
                                           truncating='post',
                                           value=0)
    train_prefer_genres_pad = tf.keras.preprocessing.sequence.pad_sequences(prefer_train_seq_genres, maxlen=seq_prefer_max_len, padding='post',
                                            truncating='post',
                                            value=0)

    train_model_input = {"user_id": train_uid, "sm_id": train_iid, "short_sm_id": train_short_item_pad,
                         "prefer_sm_id": train_prefer_item_pad,
                         "prefer_sess_length": train_prefer_len,
                         "short_sess_length": train_short_len, 'short_genres': train_short_genres_pad,
                         'prefer_genres': train_prefer_genres_pad}

    for key in user_fearther_noid:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


def gen_data_set_mind(data):
    data.sort_values("timestamp", inplace=True)
    train_set = []
    test_set = []
    user_items = {}
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        # print(reviewerID)
        pos_list = hist['sm_id'].tolist()
        pos_list = pos_list[:min(100, len(pos_list))]
        # print(pos_list)
        user_items[reviewerID] = pos_list

        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            if i != len(pos_list) - 1:
                # 用户ID + 浏览记录倒排 + 当前itemID + 正负样本label（1，0） + 浏览记录的长度 + item评分（正样本有，负样本没有）
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), 1))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1]), 1))
    # uid, 当前id , label, 短期点击item， 长期点击item, 短期点击序列长度， 长期点击序列长度， 短期点击类型， 长期点击类型， 分数
    random.shuffle(train_set)
    random.shuffle(test_set)
    # print(len(train_set[0]), len(test_set[0]))
    return train_set, test_set, user_items


def gen_model_input_mind(train_set, user_profile, seq_max_len):
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
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values

    return train_model_input, train_label


