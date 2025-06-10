# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import faiss
import numpy as np

# 1. 生成随机向量数据（模拟向量库）
# 向量维度为 128，生成 10000 个样本，float32 类型
vectors = np.random.rand(10000, 128).astype('float32')

# 2. 定义参数
dimension = 128     # 向量维度
nlist = 100         # 聚类中心个数（倒排表的桶数量）
m = 8               # PQ 的子空间个数（128维将被拆成 m=8 个子向量）
nbits = 8           # 每个子向量编码使用的 bit 数，8bit 表示 256 个质心

# 3. 构建 coarse quantizer（粗量化器）
# 用于对数据做初始的聚类划分（和 IVF 一样）
quantizer = faiss.IndexFlatL2(dimension)

# 4. 构建 IVF + PQ 索引
# - quantizer：粗聚类器
# - dimension：输入向量维度
# - nlist：聚类簇数
# - m：PQ 子向量数量（需整除 dimension）
# - nbits：每个子向量编码使用的比特位数
index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, nbits)

# 5. 训练索引（必要步骤）
# 训练包括聚类（IVF）和每个簇中子向量的 PQ 量化中心学习
index.train(vectors)
assert index.is_trained, "IndexIVFPQ 索引未训练成功！"

# 6. 添加向量到训练好的索引中
index.add(vectors)

# 7. 构造查询向量（维度一致）
query = np.random.rand(5, 128).astype('float32')

# 8. 设置 nprobe，控制查询时扫描的簇数（越大召回越全）
index.nprobe = 10

# 9. 执行搜索，返回前 k 个最相似向量
k = 5
D, I = index.search(query, k)

# 10. 打印查询结果
print("Top-5 相似向量的索引位置：", I)
print("对应的 L2 距离：", D)
