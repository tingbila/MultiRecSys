# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media
# 构建一个自动化系统，用于检测文本之间的相似性，从而识别潜在的文本抄袭行为，并进行分析与可视化。
# 原始 CSV 文件
#     ↓
# 文本清洗与去重
#     ↓
# 中文分词（jieba） + 停用词处理
#     ↓
# TF-IDF 向量化
#     ↓
# 相似度计算（余弦 / 编辑距离）
#     ↓
# 筛选高相似度文本对
#     ↓
# 输出分析结果 / 可视化



import re
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
from pprint import pprint
import os

# -----------------------------
# 加载停用词列表
# -----------------------------
with open(r'D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\chinese_stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = [i.strip() for i in file.readlines()]

# -----------------------------
# 加载新闻数据
# -----------------------------
news = pd.read_csv(r'D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\sqlResult.csv', encoding='utf-8')
print(news.shape)
print(news.head(5))

# -----------------------------
# 删除内容缺失的数据
# -----------------------------
print(news[news.content.isna()].head(5))
news = news.dropna(subset=['content'])
print(news.shape)

# -----------------------------
# 文本分词函数
# -----------------------------
def split_text(text):
    text = text.replace(' ', '').replace('\n', '')
    text2 = jieba.cut(text.strip())
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result

print(news.iloc[0].content)
print(split_text(news.iloc[0].content))

# -----------------------------
# 构建语料库（若已有缓存则直接加载）
# -----------------------------
if not os.path.exists("corpus.pkl"):
    corpus = list(map(split_text, map(str, news.content)))
    with open('corpus.pkl', 'wb') as file:
        pickle.dump(corpus, file)
else:
    with open('corpus.pkl', 'rb') as file:
        corpus = pickle.load(file)

# -----------------------------
# 计算TF-IDF矩阵
# -----------------------------
countvectorizer = CountVectorizer(min_df=0.015)
tfidftransformer = TfidfTransformer()

countvector = countvectorizer.fit_transform(corpus)
tfidf = tfidftransformer.fit_transform(countvector)
print(tfidf.shape)

# -----------------------------
# 构建标签：标记是否为新华社新闻
# -----------------------------
label = [1 if '新华' in str(source) else 0 for source in news.source]

# -----------------------------
# 划分训练集与测试集，训练模型
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), label, test_size=0.3, random_state=42)
clf = MultinomialNB()
clf.fit(X=X_train, y=y_train)

y_predict = clf.predict(X_test)

# -----------------------------
# 输出评估指标
# -----------------------------
def show_test_reslt(y_true, y_pred):
    print('accuracy:', accuracy_score(y_true, y_pred))
    print('precision:', precision_score(y_true, y_pred))
    print('recall:', recall_score(y_true, y_pred))
    print('f1_score:', f1_score(y_true, y_pred))

show_test_reslt(y_test, y_predict)

# -----------------------------
# 检测潜在抄袭文章
# -----------------------------
prediction = clf.predict(tfidf.toarray())
labels = np.array(label)
compare_news_index = pd.DataFrame({'prediction': prediction, 'labels': labels})

copy_news_index = compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['labels'] == 0)].index
xinhuashe_news_index = compare_news_index[(compare_news_index['labels'] == 1)].index

print('可能为Copy的新闻条数：', len(copy_news_index))

# -----------------------------
# 聚类：KMeans 预处理
# -----------------------------
if not os.path.exists("label.pkl"):
    from sklearn.preprocessing import Normalizer
    from sklearn.cluster import KMeans

    normalizer = Normalizer()
    scaled_array = normalizer.fit_transform(tfidf.toarray())

    kmeans = KMeans(n_clusters=25, random_state=42, n_init='auto')
    k_labels = kmeans.fit_predict(scaled_array)

    with open('label.pkl', 'wb') as file:
        pickle.dump(k_labels, file)
else:
    with open('label.pkl', 'rb') as file:
        k_labels = pickle.load(file)

# -----------------------------
# 构建 id_class 映射（index -> cluster）
# -----------------------------
if not os.path.exists("id_class.pkl"):
    id_class = {index: class_ for index, class_ in enumerate(k_labels)}
    with open('id_class.pkl', 'wb') as file:
        pickle.dump(id_class, file)
else:
    with open('id_class.pkl', 'rb') as file:
        id_class = pickle.load(file)

# -----------------------------
# 构建 class_id 映射（cluster -> indices）仅保留新华社的文章
# -----------------------------
if not os.path.exists("class_id.pkl"):
    from collections import defaultdict
    class_id = defaultdict(set)
    for index, class_ in id_class.items():
        if index in xinhuashe_news_index.tolist():
            class_id[class_].add(index)
    with open('class_id.pkl', 'wb') as file:
        pickle.dump(class_id, file)
else:
    with open('class_id.pkl', 'rb') as file:
        class_id = pickle.load(file)

# 输出每个聚类下新华社文章数量
count = 0
for k in class_id:
    print(count, len(class_id[k]))
    count += 1

# -----------------------------
# 在指定类别中查找相似文章（基于余弦相似度）
# -----------------------------
def find_similar_text(cpindex, top=10):
    dist_dict = {i: cosine_similarity(tfidf[cpindex], tfidf[i]) for i in class_id[id_class[cpindex]]}
    return sorted(dist_dict.items(), key=lambda x: x[1][0], reverse=True)[:top]

# -----------------------------
# 编辑距离计算并找出最相似句子对
# -----------------------------
import editdistance

def find_similar_sentence(candidate, raw):
    similist = []
    cl = candidate.strip().split('。')
    ra = raw.strip().split('。')
    for c in cl:
        for r in ra:
            similist.append([c, r, editdistance.eval(c, r)])
    sort = sorted(similist, key=lambda x: x[2])[:5]
    for c, r, ed in sort:
        if c != '' and r != '':
            print(f'怀疑抄袭句: {c}\n相似原句: {r}\n编辑距离: {ed}\n')

# -----------------------------
# 示例：找出某篇被怀疑抄袭的文章及其原文
# -----------------------------
cpindex = 89567  # 示例索引
similar_list = find_similar_text(cpindex)
similar2 = similar_list[0][0]

print('怀疑抄袭:\n', news.iloc[cpindex].content)
print('相似原文:\n', news.iloc[similar2].content)
print('编辑距离:', editdistance.eval(corpus[cpindex], corpus[similar2]))

find_similar_sentence(news.iloc[cpindex].content, news.iloc[similar2].content)
