# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media

import faiss
import numpy as np

# 1. 生成随机向量数据（模拟向量库）
# 共生成 10000 个向量，每个向量维度为 128
# 数据类型必须为 float32（Faiss 要求）
vectors = np.random.rand(10000, 128).astype('float32')

# 2. 构建 IVF（倒排文件）索引结构
# 设置向量维度为 128
dimension = 128

# 设置聚类中心的数量（nlist），即 coarse quantizer 的簇数量
# nlist 越大，索引越稀疏，查询速度越快但可能降低精度
nlist = 100

# 构建量化器（quantizer）：
# 使用 IndexFlatL2 作为 coarse quantizer，即先用精确 L2 距离做聚类
quantizer = faiss.IndexFlatL2(dimension)

# 构建 IVF 索引：
# - quantizer：聚类器
# - dimension：向量维度
# - nlist：聚类中心数量
# - METRIC_L2：使用欧氏距离（L2 距离）作为度量方式
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
print(index.is_trained) # False

# 3. 训练索引（必须）
# IVF 索引在添加向量前必须先用训练数据进行聚类，以确定簇中心
index.train(vectors)
print(index.is_trained) # True


# 检查索引是否训练成功
assert index.is_trained, "索引未训练成功！"

# 4. 将数据添加到训练好的索引中
index.add(vectors)  # 此处数据会被分配到对应的聚类簇中

# 5. 构造查询向量（形状必须与索引向量一致）
query = np.random.rand(5, 128).astype('float32')
print(query)

# 6. 设置 nprobe 参数：
# 控制查询时要搜索多少个聚类簇（nprobe 越大，查全率越高但速度下降）
# 默认值是 1，通常设置为 8~32 会更平衡
# nprobe较小时，查询可能会出错，但时间开销很小
# nprobe较大时，精度逐渐增大，但时间开销也增加
# nprobe=nlist时，等效于IndexFlatL2索引类型。
index.nprobe = 10

# 7. 执行查询，返回最相似的 Top-5 向量
# D: 距离数组，表示每个匹配向量与查询向量的距离
# I: 索引数组，表示匹配向量在向量库中的位置
D, I = index.search(query, k=5)

# 8. 打印查询结果（但是这里竟然没有输出，感觉很奇怪！）
print("Top-5 相似向量的索引位置：", I)
print("对应的 L2 距离：", D)
