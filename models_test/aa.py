import faiss
import numpy as np

# 1. 生成随机向量数据，模拟一个向量库
# 这里生成 10000 个向量，每个向量维度为 128，数据类型为 float32（Faiss要求）
vectors = np.random.rand(10000, 128).astype('float32')

# 2. 构建精确搜索索引：IndexFlatL2，基于欧氏距离的全量扫描索引
# 该索引不做压缩或聚类，查询时会遍历所有向量，计算精确距离
index = faiss.IndexFlatL2(128)

# 3. 将向量数据添加到索引中，完成索引构建
index.add(vectors)

# 4. 构建多个查询向量（这里是5个），维度需与索引一致
query = np.random.rand(5, 128).astype('float32')

# 5. 执行查询，寻找每个查询向量最近的 Top-5 向量
# 返回两个矩阵：
# D: 距离矩阵，形状为 (5, 5)，表示5个查询向量分别与对应Top-5向量的L2距离
# I: 索引矩阵，形状为 (5, 5)，表示5个查询向量对应Top-5向量在数据库中的索引位置
D, I = index.search(query, k=5)

# 6. 打印查询结果
print("Top-5 相似向量的索引位置（每行对应一个查询向量）：\n", I)
print("对应的 L2 距离（每行对应一个查询向量）：\n", D)
