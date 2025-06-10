# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media

import numpy as np
import pandas as pd
import pickle
import os
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import faiss


# 🔍 工作机制说明：
#     文本表示：
#         每篇文章通过分词 → CountVectorizer → TF-IDF 转换成向量 tfidf。
#     索引建立：
#         使用 faiss.IndexFlatL2 构建索引，计算的是 L2 欧几里得距离。
#     查询检索：
#         选取待检测文章 query_self = tfidf[cpindex:cpindex+1]。
#          index.search(query_self, k) 返回最相似的 k 篇文章的索引与距离。


# ----------------------
# 1. 中文文本分词函数定义
# ----------------------
def split_text(text):
    """
    对输入文本进行中文分词，清除非中文字符。
    :param text: 原始字符串文本
    :return: 分词后的文本字符串，以空格连接
    """
    # 清除非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba进行分词
    words = jieba.cut(text)
    return ' '.join(words)

# ----------------------
# 2. 数据读取与预处理
# ----------------------
news = pd.read_csv(r'D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\sqlResult.csv', encoding='utf-8')


# 删除 content 为空的行
print(news[news.content.isna()].head(5))
news = news.dropna(subset=['content'])

# ----------------------
# 3. 分词结果缓存与加载
# ----------------------
corpus_path = "corpus.pkl"
if not os.path.exists(corpus_path):
    corpus = list(map(split_text, [str(i) for i in news.content]))
    print(corpus[0])
    print(len(corpus))
    print(corpus[1])
    with open(corpus_path, 'wb') as file:
        pickle.dump(corpus, file)
else:
    with open(corpus_path, 'rb') as file:
        corpus = pickle.load(file)

# ----------------------
# 4. TF-IDF 特征提取
# ----------------------
tfidf_path = "tfidf.pkl"
if not os.path.exists(tfidf_path):
    countvectorizer = CountVectorizer(min_df=0.015)  # 去除低频词
    tfidftransformer = TfidfTransformer()
    countvector = countvectorizer.fit_transform(corpus)  # 词频统计矩阵
    tfidf = tfidftransformer.fit_transform(countvector)  # TF-IDF 矩阵
    print(countvector.shape)
    print(tfidf.shape)
    with open(tfidf_path, 'wb') as file:
        pickle.dump(tfidf, file)
else:
    with open(tfidf_path, 'rb') as file:
        tfidf = pickle.load(file)

# 转换为 NumPy 格式并降低精度（节省内存）
tfidf = tfidf.toarray().astype(np.float32)
d = tfidf.shape[1]  # 向量维度
print(d)
print(tfidf.shape)
print(type(tfidf))
print(type(tfidf[1][1]))

# ----------------------
# 5. FAISS 向量搜索构建索引
# ----------------------
index = faiss.IndexFlatL2(d)  # 使用欧氏距离构建索引（无需训练）
print(index.is_trained)  # 验证是否训练（对于 IndexFlatL2 总为 True）
index.add(tfidf)  # 添加全部文本向量到索引中
print(index.ntotal)  # 索引中向量个数

# ----------------------
# 6. 查询相似文本（模拟抄袭检测）
# ----------------------
k = 10  # 返回最相似的前k条记录
cpindex = 3352  # 指定待检测文本的索引
query_self = tfidf[cpindex:cpindex + 1]  # 取出查询向量

# 执行相似度搜索
dis, ind = index.search(query_self, k)

print(dis.shape)
print(ind.shape)
print(dis)  # 相似度（欧氏距离）
print(ind)  # 相似文本的索引

# 输出相似文本对内容
print('怀疑抄袭:\n', news.iloc[cpindex].content)
similar2 = ind[0][1]  # 第二个最相似的是相似文本（第一个是自己）
print(similar2)
print('相似原文:\n', news.iloc[similar2].content)
