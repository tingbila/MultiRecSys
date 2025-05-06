import random
import numpy as np
from tqdm import tqdm
from config.dssm_config import *
import tensorflow as tf


def gen_data_set(data, min_count, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data[['sm_id']]
    # 统计sm_id 出现的次数, 转为dataframe
    item_ids = item_ids['sm_id'].value_counts().rename_axis('sm_id').reset_index(name='num')
    item_ids = item_ids.loc[item_ids['num'] > min_count]

    # hot_ids = item_ids.loc[item_ids['num'] > int(item_ids["num"].sum() / (int(item_ids.shape[0] * 0.1)))]
    # hot_ids = set(hot_ids['sm_id'].unique())

    hot_ids = []

    item_nums = item_ids['num'].to_list()  # .div(item_ids['num'].sum())
    item_ids = item_ids['sm_id'].unique()

    train_set = []
    test_set = []
    count = 0
    user_items = {}
    print(1)
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
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set, user_items, hot_ids


def gen_data_set_new(data, min_count, negsample=0):
    data.sort_values("timestamp", inplace=True)
    item_ids = data[['sm_id']]
    # 统计sm_id 出现的次数, 转为dataframe
    item_ids = item_ids['sm_id'].value_counts().rename_axis('sm_id').reset_index(name='num')
    item_ids = item_ids.loc[item_ids['num'] > min_count]

    hot_ids = item_ids.loc[item_ids['num'] > int(item_ids["num"].sum() / (int(item_ids.shape[0] * 0.1)))]
    hot_ids = set(hot_ids['sm_id'].unique())

    item_nums = item_ids['num'].to_list()  # .div(item_ids['num'].sum())
    item_ids = item_ids['sm_id'].unique()

    train_set = []
    test_set = []
    count = 0
    user_items = {}
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['sm_id'].tolist()  # 用户浏览过的item_id
        pos_list = pos_list[:min(100, len(pos_list))]
        user_items[reviewerID] = pos_list
        neg_list = set()
        if negsample > 0:  # 全局负采样
            # candidate_set = list(set(item_ids) - set(pos_list))
            # neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
            while len(neg_list) < len(pos_list) * negsample:
                tmp = random.choices(item_ids, weights=item_nums, k=len(pos_list) * negsample)
                for t in tmp:
                    if t in set(pos_list):
                        continue
                    neg_list.add(t)
                    if len(neg_list) >= len(pos_list) * negsample:
                        break
            # print(neg_list)
        neg_list = list(neg_list)

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
    random.shuffle(train_set)
    random.shuffle(test_set)

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


def gen_test_inpout(train_set, user_profile, item_profile, item_ids, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = item_ids['sm_id'].values
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    # padding
    train_seq_pad = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=seq_max_len, padding='post',
                                                                  truncating='post', value=0)
    # print(train_seq_pad)
    train_user_input = {"user_id": train_uid,  "hist_sm_id": train_seq_pad, "hist_len": train_hist_len}

    for key in user_fearther_noid:
        try:
            train_user_input[key] = user_profile.loc[train_user_input['user_id']][key].values
        except:
            continue

    train_item_input = {"sm_id": train_iid}
    for key in item_fearther_noid:
        try:
            train_item_input[key] = item_profile.loc[train_item_input['sm_id']][key].values
        except:
            continue

    return train_user_input, train_item_input, train_label

