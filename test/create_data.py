# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import json
import csv
import random

# 构造一些可能的风格和演员供随机选择
possible_genres = ["动作", "喜剧", "爱情", "科幻", "悬疑", "惊悚", "剧情", "战争"]
possible_actors = ["刘德华", "成龙", "李连杰", "张学友", "周星驰", "马特·达蒙", "小李子", "汤姆·哈迪", "巩俐", "舒淇"]

# 加载字典（如果没有就创建空的）
try:
    with open("movie_id_to_genres.json", "r", encoding="utf-8") as f:
        movie_id_to_genres = json.load(f)
except FileNotFoundError:
    movie_id_to_genres = {}

try:
    with open("movie_id_to_actors.json", "r", encoding="utf-8") as f:
        movie_id_to_actors = json.load(f)
except FileNotFoundError:
    movie_id_to_actors = {}

# 打开原始文件和输出文件
with open(r"D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\train_2.csv", "r", encoding="utf-8") as infile, \
     open("data_with_seq.tsv", "w", encoding="utf-8", newline='') as outfile:

    reader = csv.reader(infile, delimiter='\t')
    writer = csv.writer(outfile, delimiter='\t')

    for row in reader:
        # 取出 movie_id，假设在第3列（即下标为2）
        movie_id = row[2]

        # 转换为字符串 key
        movie_id_str = str(movie_id)

        # 如果不存在，则随机生成
        if movie_id_str not in movie_id_to_genres:
            genres = random.sample(possible_genres, k=random.randint(1, 2))
            movie_id_to_genres[movie_id_str] = genres
        else:
            genres = movie_id_to_genres[movie_id_str]

        if movie_id_str not in movie_id_to_actors:
            actors = random.sample(possible_actors, k=random.randint(1, 2))
            movie_id_to_actors[movie_id_str] = actors
        else:
            actors = movie_id_to_actors[movie_id_str]

        # 转为字符串
        genres_str = ",".join(genres)
        actors_str = ",".join(actors)

        # 添加到行尾
        row.extend([genres_str, actors_str])
        writer.writerow(row)

# 保存更新后的字典
with open("movie_id_to_genres.json", "w", encoding="utf-8") as f:
    json.dump(movie_id_to_genres, f, ensure_ascii=False, indent=2)

with open("movie_id_to_actors.json", "w", encoding="utf-8") as f:
    json.dump(movie_id_to_actors, f, ensure_ascii=False, indent=2)


