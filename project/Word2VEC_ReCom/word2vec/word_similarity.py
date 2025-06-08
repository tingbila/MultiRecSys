# -*- coding: utf-8 -*-
"""
中文文本 Word2Vec 相似度计算脚本：
1. 请确保先执行分词脚本（word_seg.py）对原始文本进行中文分词。
2. 本脚本将使用 gensim 的 Word2Vec 模型训练词向量，并计算词语相似度。
"""

from gensim.models import word2vec
import multiprocessing
import os

# 指定中文分词后的语料文件目录
segment_folder = './journey_to_the_west/segment'

# 加载分词后的语料，每行为一个句子，多个文件自动遍历
sentences = word2vec.PathLineSentences(segment_folder)
print(f"[INFO] 成功加载分词语料路径: {os.path.abspath(segment_folder)}")

# 第一个 Word2Vec 模型训练示例（较小参数，适用于调试）
print("[INFO] 开始训练 Word2Vec 模型（vector_size=100, window=3, min_count=1）...")
model = word2vec.Word2Vec(sentences, vector_size=100, window=3, min_count=1)
print("[INFO] 模型训练完成。")

# 相似度测试示例
print("[INFO] 计算词语相似度（模型1）:")
print(f"  孙悟空 vs 猪八戒 相似度: {model.wv.similarity('孙悟空', '猪八戒'):.4f}")
print(f"  孙悟空 vs 孙行者 相似度: {model.wv.similarity('孙悟空', '孙行者'):.4f}")

# 词向量运算示例：孙悟空 + 唐僧 - 孙行者 的相似词
print("[INFO] 词向量关系示例（孙悟空 + 唐僧 - 孙行者）:")
# model.wv.most_similar() 方法默认返回 前 10 个最相似词项
similar_words = model.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者'])
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# 第二个 Word2Vec 模型训练（更大规模，用于保存）
print("[INFO] 开始训练 Word2Vec 模型（vector_size=128, window=5, min_count=5）...")
model2 = word2vec.Word2Vec(
    sentences,                # 输入语料：一个可迭代对象，每个元素是一句已分词的列表（这里是分好词的文本行）
    vector_size=128,          # 词向量的维度（每个词将被表示为一个128维的向量）
    window=5,                 # 上下文窗口大小（目标词左右各考虑5个词作为上下文）
    min_count=5,              # 忽略词频小于5的词（只训练语料中至少出现5次的词，过滤噪声）
    workers=multiprocessing.cpu_count()  # 使用 CPU 的核心数量来并行训练，加快模型构建速度
)
print("[INFO] 模型训练完成。")

# 保存模型到本地
model_path = './models/word2Vec.model'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model2.save(model_path)
print(f"[INFO] 模型已保存至: {model_path}")

# 第二个模型的相似度计算与结果
print("[INFO] 计算词语相似度（模型2）:")
print(f"  孙悟空 vs 猪八戒 相似度: {model2.wv.similarity('孙悟空', '猪八戒'):.4f}")
print(f"  孙悟空 vs 孙行者 相似度: {model2.wv.similarity('孙悟空', '孙行者'):.4f}")

print("[INFO] 词向量关系示例（孙悟空 + 唐僧 - 孙行者）:")
similar_words = model2.wv.most_similar(positive=['孙悟空', '唐僧'], negative=['孙行者'])
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")



"""
[INFO] 成功加载分词语料路径: D:\software\pycharm_repository\StarMaker\MultiRecSys\project\Word2VEC_6\word2vec\journey_to_the_west\segment
[INFO] 开始训练 Word2Vec 模型（vector_size=100, window=3, min_count=1）...
[INFO] 模型训练完成。
[INFO] 计算词语相似度（模型1）:
  孙悟空 vs 猪八戒 相似度: 0.9769
  孙悟空 vs 孙行者 相似度: 0.9805
[INFO] 词向量关系示例（孙悟空 + 唐僧 - 孙行者）:
  大王: 0.9771
  老: 0.9751
  陛下: 0.9697
  菩萨: 0.9694
  长老: 0.9692
  先生: 0.9691
  们: 0.9679
  滴泪: 0.9677
  师兄: 0.9676
  贫僧: 0.9669
[INFO] 开始训练 Word2Vec 模型（vector_size=128, window=5, min_count=5）...
[INFO] 模型训练完成。
[INFO] 模型已保存至: ./models/word2Vec.model
[INFO] 计算词语相似度（模型2）:
  孙悟空 vs 猪八戒 相似度: 0.9571
  孙悟空 vs 孙行者 相似度: 0.9696
[INFO] 词向量关系示例（孙悟空 + 唐僧 - 孙行者）:
  弟子: 0.9321
  菩萨: 0.9218
  何往: 0.9183
  众: 0.9155
  惊讶: 0.9121
  玉帝: 0.9117
  天王: 0.9081
  太子: 0.9078
  太宗: 0.9069
  银角: 0.9005
"""