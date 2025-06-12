# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import multiprocessing
import gensim
from gensim.models import word2vec
import pickle

"""
word2vec介绍:
Word2Vec 的作用是 “Embedding 级别的协同过滤召回”:"用户听过的歌曲序列" → 找出相似歌曲（embedding） → 推荐（召回）
 
说起来word2vec，其实就是把词映射成一定维度的稠密向量，同时保持住词和词之间的关联性，主要体现在(欧式)距离的远近上。
那么问题来了，word2vec为什么能够学习到这样的结果？
因为我们相信“物以类聚，人以群分” “一个人的层次与他身边最近的一些人是差不多的”
同样的考虑，我们是不是可以认为，一个歌单里的歌曲，相互之间都有一定的关联性呢？就像句子中的词一样。答案是，是的！

这段代码的整体目的是：训练一个基于 Word2Vec 的“歌曲向量表示模型（Song2Vec）”，从而可以根据某首歌推荐与其相似的歌曲，本质上是一种“协同过滤” + “序列建模”的方法，常用于音乐推荐系统中。
163_music_playlist.txt:
每一行表示一个歌单，第一个字段代表是歌单 ID，后面是多首歌的信息（用 \t 分隔，单首歌用 ::: 分隔字段）。每首歌格式为：song_id:::song_name:::author_id:::popularity
"""


def parse_playlist_get_sequence(in_line, playlist_sequence):
    """
    解析一行歌单内容，提取出歌曲 ID 序列，追加到 playlist_sequence 中。
    歌单格式：playlist_id \t song1 ::: song_name ::: author_id ::: popularity \t song2 ::: ...
    """
    song_sequence = []
    contents = in_line.strip().split("\t")
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split(":::")
            song_sequence.append(song_id)
        except ValueError:
            print("[解析错误] 歌曲格式有误:", song)
    playlist_sequence.append(song_sequence)


def train_song2vec(in_file):
    """
    从输入文件读取歌单序列，训练 Word2Vec 模型，并保存模型。
    """
    playlist_sequence = []
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            parse_playlist_get_sequence(line, playlist_sequence)
    # playlist_sequence = [
    #     ['songA', 'songB', 'songC'],
    #     ['songB', 'songD', 'songE'],
    #     ...
    # ]
    cores = multiprocessing.cpu_count()
    print(f"使用 {cores} 个核心训练 Word2Vec 模型...") # 使用 12 个核心训练 Word2Vec 模型...

    import gensim
    from gensim.models import word2vec
    # playlist_sequence = [
    #     ['songA', 'songB', 'songC'],
    #     ['songB', 'songD', 'songE'],
    #     ...
    # ]
    """
    参数名	说明
    sentences	输入语料，类型为列表的列表（例如 [['song1', 'song2', 'song3'], ...]），每个子列表是一条“句子”，在推荐场景中可以是一个播放序列、行为序列等。
    vector_size	词向量的维度大小，原参数名为 size，在 gensim>=4.0 中改为 vector_size。这里设为 150，表示每个“词”（如歌曲、用户行为等）将被编码为 150 维的向量。
    min_count	最小词频阈值，只有出现次数大于或等于这个值的词才会被纳入训练。设为 3 表示只考虑在 playlist_sequence 中出现了至少 3 次的词，过滤低频项。
    window	上下文窗口大小，即模型训练时一个词与其上下文词的最大距离。设为 7 表示每个词的前后各看最多 7 个词作为上下文。
    workers	并行训练的线程数，通常设为你的 CPU 核心数。提高 workers 数可以加快训练速度。
    """
    model = word2vec.Word2Vec(
        sentences=playlist_sequence,
        vector_size=150,
        min_count=3,
        window=7,
        workers=cores
    )

    return model


if __name__ == '__main__':
    song_sequence_file = r"D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\163_music_playlist.txt"
    """
    import cPickle as pickle
    song_dic = pickle.load(open("song.pkl","rb"))
    model_str = "./song2vec.model"
    model = gensim.models.Word2Vec.load(model_str)
    """
    model = train_song2vec(song_sequence_file)

    # 选择部分歌曲 ID 测试相似度
    # 选择部分歌曲 ID 测试相似度
    song_id_list = ['4513', '1237', '5938', '5843', '5969', '3369']
    for song_id in song_id_list:
        if song_id in model.wv:
            result_song_list = model.wv.most_similar(song_id)
            print(f"\n🎵 歌曲 ID: {song_id}")
            print("相似歌曲ID\t\t相似度")
            for similar_song_id, similarity in result_song_list:
                print(f"\t{similar_song_id}\t\t{similarity:.4f}")
        else:
            print(f"\n⚠️ 歌曲 ID {song_id} 不在词向量模型中，可能未满足 min_count 要求")


"""
使用 12 个核心训练 Word2Vec 模型...

🎵 歌曲 ID: 4513
相似歌曲ID		相似度
	4622		0.3119
	3630		0.2643
	2947		0.2556
	2695		0.2519
	1143		0.2506
	2171		0.2479
	3065		0.2429
	1378		0.2411
	5110		0.2409
	3400		0.2397

🎵 歌曲 ID: 1237
相似歌曲ID		相似度
	2997		0.2767
	1621		0.2679
	4790		0.2647
	4126		0.2504
	5600		0.2469
	2902		0.2450
	1525		0.2346
	1217		0.2316
	1676		0.2285
	5178		0.2265

⚠️ 歌曲 ID 5938 不在词向量模型中，可能未满足 min_count 要求

⚠️ 歌曲 ID 5843 不在词向量模型中，可能未满足 min_count 要求

🎵 歌曲 ID: 5969
相似歌曲ID		相似度
	1226		0.3422
	4136		0.2853
	2320		0.2833
	1319		0.2651
	3926		0.2649
	3390		0.2615
	4281		0.2515
	2430		0.2481
	5965		0.2463
	1938		0.2414

🎵 歌曲 ID: 3369
相似歌曲ID		相似度
	2452		0.2713
	2585		0.2535
	4252		0.2515
	4392		0.2512
	1815		0.2505
	4294		0.2448
	4440		0.2435
	3162		0.2412
	3472		0.2401
	1954		0.2340
"""