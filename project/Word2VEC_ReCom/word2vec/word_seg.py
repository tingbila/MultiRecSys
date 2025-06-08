# -*-coding: utf-8 -*-
# 对txt文件进行中文分词
import jieba
import os
from utils import files_processing

# 源文件所在目录
source_folder = './journey_to_the_west/source'
segment_folder = './journey_to_the_west/segment'

# 字词分割，对整个文件内容进行字词分割
"""
✅ 示例输入输出：
输入文件内容（原始未分词）：
孙悟空大闹天宫，玉皇大帝命令李靖带兵擒拿。
输出文件内容（处理后结果）：
孙悟空 大闹 天宫 ， 玉皇大帝 李靖 带兵 擒拿 。
你也可以加入自定义停用词，比如把“命令”从分词结果中去掉。
"""
# 定义一个函数：对一组文本文件进行分词，并将结果保存到指定目录
# 这段代码的作用是：对一批中文文本文件进行分词处理，并将分词后的内容保存到指定目录中。适用于如《西游记》这类文本的预处理工作，
# 为后续的自然语言处理（如TF-IDF、文本分类、文本生成）做准备。
def segment_lines(file_list, segment_out_dir, stopwords=[]):
    # 遍历文件列表，每个文件进行分词处理
    for i, file in enumerate(file_list):
        # 构造输出文件名，例如 segment_0.txt、segment_1.txt ...
        segment_out_name = os.path.join(segment_out_dir, 'segment_{}.txt'.format(i))
        print(segment_out_name)   # ./journey_to_the_west/segment\segment_0.txt

        # 以二进制方式读取原始文件内容（适用于中文文件避免编码错误）
        with open(file, 'rb') as f:
            document = f.read()

            # 使用 jieba 对文档内容进行分词（得到生成器）
            document_cut = jieba.cut(document)

            # 存储分词结果的列表
            sentence_segment = []

            # 遍历每个分词结果
            for word in document_cut:
                # 如果该词不是停用词，就保留
                if word not in stopwords:
                    sentence_segment.append(word)

            # 用空格将所有保留的词拼接成一个字符串
            result = ' '.join(sentence_segment)
            print(result)
            """
            南无 接引 归真 佛 。 南无 金刚 不坏 佛 。 南无宝 光佛 。 南无龙尊 王佛 。 南无 精进 善佛 。 南无宝 月光 佛 。 南无现 无 愚佛 。 南无婆 留 那佛 。 南无 那罗延佛 。 南 无功 德华 佛 。 南无才 功德 佛 。 南无善 游步 佛 。 南无 旃檀 光佛 。 南无摩尼幢 佛 。 南无慧炬照 佛 。 
            南无 海德 光明 佛 。 南无大慈 光佛 。 南无 慈力王 佛 。 南无贤善 首佛 。 南无广主 严佛 。 南无 金华 光佛 。 南无才 光明 佛 。 南无 智慧 胜佛 。 南无世静 光佛 。 南无 日月 光佛 。 南无 日月 珠光 佛 。 
            南无慧 幢 胜 王佛 。 南无 妙音 声佛 。 南 无常 光幢 佛 。 南无观 世灯 佛 。 南 无法 胜王佛 。 南 无须 弥 光佛 。 南无大慧力 王佛 。 南无 金海 光佛 。 南无 大通 光佛 。 南无才 光佛 。 南无 旃檀 功德 佛 。
            """
            # 将字符串编码为 utf-8 格式的二进制内容
            result = result.encode('utf-8')

            # 以二进制写入方式保存分词结果到新文件中
            with open(segment_out_name, 'wb') as f2:
                f2.write(result)


# 对source中的txt文件进行分词，输出到segment目录中
file_list=files_processing.get_files_list(source_folder, postfix='*.txt')
print(file_list)   # ['./journey_to_the_west/source\\journey_to_the_west.txt']
segment_lines(file_list, segment_folder)
