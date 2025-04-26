# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email :  mingyang.zhang@ushow.media

# coding = utf-8
import argparse
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="DSSM Recommendation System Parameters")

    parser.add_argument("--data_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_data.csv",help="原始输入数据路径")
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
    parser.add_argument("--recall_topk", type=int, default=50, help="评估时的 Top-K 召回覆盖率")

    parser.add_argument("--save_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_u2i.txt",help="主召回结果保存路径")
    parser.add_argument("--save_dir_new", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_new.txt",help="新召回结果保存路径")
    parser.add_argument("--save_sdm_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/sdm_data_u2i.txt",help="SDM 模型召回结果保存路径")
    parser.add_argument("--save_mind_dir", type=str, default="/data1/guifang.ji/DSSM_SongRecall/data/mind_data_u2i.txt",help="MIND 模型召回结果保存路径")
    parser.add_argument("--save_final_dir", type=str,default="/data1/guifang.ji/DSSM_SongRecall/data/dssm_final_u2i.txt", help="最终合并召回结果保存路径")

    return parser.parse_args()


# 定义用户和物品塔的 DNN 隐藏层结构（每层神经元个数）
user_hidden_unit = (64, 32)
item_hidden_unit = (64, 32)

# 训练数据字段名称列表（对应 CSV 文件的列）
data_cloums = ["user_id", "sm_id", "timestamp", "level", "user_lang", "platform", "country",
               "gender", "age", "song_lang_id", "artist_gender", "song_quality", "song_recording_count",
               "song_genres", "song_create_time", "artist_country_id", "region", "is_new"]

# 模型使用的特征字段
features = ["user_id", "sm_id", "level", "user_lang", "platform",
            "country", "gender", "artist_gender", "age", "is_new",
            "song_quality", "song_recording_count", "song_genres"]

# 数值型字段（可直接归一化/填充处理）
num_header = ["level", "gender", "age", "artist_gender", "is_new", "song_quality", "song_recording_count"]

# 字符串型字段（需要 LabelEncoder 编码）
string_header = ["user_id", "sm_id", "user_lang", "platform", "country", "song_genres"]

# 时间字段（如需用于建模可做时间差/周期处理）
time_header = ["timestamp"]

# 用户侧特征字段（含 user_id）
user_fearther = ["user_id", "gender", "age", "level", "user_lang", "country", "platform", "is_new"]

# 物品侧特征字段（含 sm_id）
item_fearther = ["sm_id", "artist_gender", "song_quality", "song_recording_count", "song_genres"]  # 注：去掉了缺失率高的字段

# 用户侧特征字段（去除 user_id，用于塔结构建模）
user_fearther_noid = ["gender", "age", "level", "user_lang", "country", "platform", "is_new"]

# 物品侧特征字段（去除 sm_id，用于塔结构建模）
item_fearther_noid = ["artist_gender", "song_quality", "song_recording_count", "song_genres"]

# 模型训练最终需要的字段（供写出/转存）
final_col = ["user_id", "sm_id", "timestamp", "level", "user_lang", "platform", "country",
             "gender", "age", "song_lang_id", "artist_gender", "song_quality", "song_recording_count",
             "song_genres", "song_create_time", "artist_country_id"]


if __name__ == "__main__":
    args = parse_args()

    # 你可以在后续逻辑中这样使用参数：
    print("训练数据路径：", args.data_dir)
    print("batch_size:", args.batch_size)
    print("embedding_dim:", args.embedding_dim)


    print(final_col)
