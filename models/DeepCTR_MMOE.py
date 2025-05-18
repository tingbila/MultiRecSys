import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm
from tensorflow import keras
from deepctr.models import DeepFM, MMOE

import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个

# ========== 设置选项 & 警告过滤 ==========
# 设置 pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)




# === 数值特征处理 ===
def process_dense_feats(data, feats):
    """
    使用 StandardScaler 对数值特征进行标准化处理。
    """
    data[feats] = data[feats].fillna(0)
    scaler = StandardScaler()
    data[feats] = scaler.fit_transform(data[feats])
    return data


# === 稀疏特征处理 ===
def process_sparse_feats(data, feats):
    """
    对稀疏特征进行填充缺失并编码为整数索引。
    """
    for f in tqdm(feats, desc='Processing Sparse Features'):
        label_encoder = LabelEncoder()
        data[f] = label_encoder.fit_transform(data[f].astype(str))
    return data


# === 变长序列特征处理 ===
def process_sequence_feats(data, sequence_features):
    """
    对变长特征使用自定义 Tokenizer 分词，并进行 padding 处理。
    """
    # 用于存储每个变长特征处理后的 padding 序列
    pad_sequences_dict = {}
    # 每个变长特征对应一个独立的 Tokenizer，用于后续文本转索引
    tokenizers = {}
    # 用于记录每个变长特征的 padding 长度（即序列被填充后的最大长度）
    pad_len_dict = {}

    for feature in sequence_features:
        texts = data[feature].fillna("").apply(lambda x: x.replace('|', ' ')).tolist()
        tokenizer = Tokenizer(oov_token='OOV')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, padding='post')  # pad_sequences(sequences, padding='post') 会自动按照最长的序列长度进行补齐
        pad_sequences_dict[feature] = padded
        tokenizers[feature] = tokenizer
        pad_len_dict[feature] = padded.shape[1]

    return pad_sequences_dict, tokenizers, pad_len_dict


def main():
    # === 1. 读取数据 ===
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "finish", "like", "music_id", "device", "time", "duration_time", "actors", "genres"]
    data = pd.read_csv(r"D:\\software\\pycharm_repository\\StarMaker\\MultiRecSys\\data_files\\train_2.csv", sep='\t', names=column_names)

    # 特征定义
    sparse_feats = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
    dense_feats = ["time", "duration_time"]
    sequence_feats = ['actors', 'genres']
    target = ['finish']  # 推荐任务目标，MMOE这里没有用到这个字段，在后面写的。

    # === 2. 特征处理 ===
    data = process_dense_feats(data, dense_feats)
    data = process_sparse_feats(data, sparse_feats)
    pad_sequences_dict, tokenizers, pad_len_dict = process_sequence_feats(data, sequence_feats) if sequence_feats else ({}, {}, {})

    # === 3. 构建特征列 ===
    fixlen_feature_columns = [
        SparseFeat(feat, data[feat].nunique() + 1, embedding_dim=4) for feat in sparse_feats
    ] + [
        DenseFeat(feat, 1) for feat in dense_feats
    ]

    # maxlen:每个序列特征被 padding 到的最大长度，确保输入模型的 shape 一致
    # combiner='mean':指定如何将序列嵌入表示“压缩”为一个固定长度向量
    # 这段代码是告诉 DeepCTR：这些特征是变长的 token 序列，每个 token 有 embedding，用 mean 聚合序列表示，最终作为模型的输入
    if sequence_feats:
        for feat in sequence_feats:
            fixlen_feature_columns.append(
                VarLenSparseFeat(
                    SparseFeat(feat, vocabulary_size=len(tokenizers[feat].word_index) + 1, embedding_dim=4),
                    maxlen=pad_len_dict[feat], combiner='mean')
            )

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns    = fixlen_feature_columns
    fixlen_feature_names   = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # === 4. 划分训练集和测试集 ===
    train, test = train_test_split(data, test_size=0.5, random_state=2018)

    # 构建模型输入
    train_model_input = {name: train[name] for name in fixlen_feature_names}
    test_model_input  = {name: test[name] for name in fixlen_feature_names}

    # 添加序列特征到输入
    # 把提前 pad 好的序列特征（如 genres、actors），按照 train/test 划分后的索引，分成训练和测试输入字典
    if sequence_feats:
        for feat in sequence_feats:
            train_model_input[feat] = pad_sequences_dict[feat][train.index]
            test_model_input[feat]  = pad_sequences_dict[feat][test.index]

    # === 5. 构建和训练 MMOE 模型 ===
    model = MMOE(dnn_feature_columns, tower_dnn_hidden_units=[],task_types=['binary', 'binary'],task_names=['finish', 'like'])

    # 编译模型
    model.compile("adam",loss=["binary_crossentropy", "binary_crossentropy"],metrics=['accuracy', keras.metrics.AUC(name='auc')])

    # 模型训练
    history = model.fit(train_model_input,[train['finish'].values, train['like'].values],batch_size=256, epochs=10, verbose=2, validation_split=0.2)

    # 模型预测
    pred_ans = model.predict(test_model_input, batch_size=256)

    # AUC 评估
    print("test finish AUC",  round(roc_auc_score(test['finish'], pred_ans[0]), 4))
    print("test like AUC", round(roc_auc_score(test['like'], pred_ans[1]), 4))

    # test income AUC 0.2696
    # test marital AUC 0.8363

    # === 6. 可视化数据 ===
    print(pd.DataFrame(history.history))
    print(history.epoch)
    plt.figure(figsize=(12, 5))

    plt.plot(history.history['val_finish_auc'], label='Val Finish AUC')
    plt.plot(history.history['val_like_auc'], label='Val Like AUC')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC per Task')
    plt.grid()
    plt.show()



if __name__ == "__main__":
    main()


#        loss  finish_loss  like_loss  finish_accuracy  finish_auc  like_accuracy  like_auc  val_loss  val_finish_loss  val_like_loss  val_finish_accuracy  val_finish_auc  val_like_accuracy  val_like_auc
# 0  1.396883     0.695688   0.701195         0.376068    0.543492       0.606838  0.602865  1.348868         0.689644       0.659224             0.600000        0.561993                1.0      0.615394
# 1  1.352113     0.691124   0.660989         0.512821    0.574405       0.991453  0.681725  1.303363         0.686795       0.616568             0.533333        0.659496                1.0      0.680633
# 2  1.311687     0.687430   0.624257         0.615385    0.643916       1.000000  0.704211  1.263147         0.684410       0.578737             0.600000        0.689533                1.0      0.702315
# 3  1.274607     0.684053   0.590554         0.658120    0.673471       1.000000  0.714310  1.227076         0.682471       0.544604             0.600000        0.701707                1.0      0.710904
# 4  1.240578     0.681113   0.559464         0.641026    0.684322       1.000000  0.715690  1.194510         0.681060       0.513449             0.600000        0.705989                1.0      0.713250
# 5  1.208751     0.678393   0.530357         0.641026    0.693497       1.000000  0.718756  1.164623         0.679760       0.484862             0.600000        0.710542                1.0      0.716504
# 6  1.178705     0.675675   0.503029         0.641026    0.698631       1.000000  0.719851  1.136378         0.678349       0.458029             0.600000        0.712688                1.0      0.717762
# 7  1.149845     0.672940   0.476904         0.641026    0.701869       1.000000  0.720214  1.109345         0.677127       0.432217             0.600000        0.714091                1.0      0.718511
# 8  1.121689     0.670259   0.451429         0.641026    0.704392       1.000000  0.720524  1.083535         0.676193       0.407341             0.600000        0.715063                1.0      0.718982
# 9  1.094038     0.667728   0.426308         0.641026    0.706203       1.000000  0.720618  1.059045         0.675866       0.383177             0.600000        0.715597                1.0      0.719118
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]