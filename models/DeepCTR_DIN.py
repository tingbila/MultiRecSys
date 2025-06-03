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
import os
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个

# Pandas 显示选项（非必须，但开发调试时很有用）
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# 定义稀疏特征的函数，返回特征名和特征可能的取值数量
def sparse_feat(feat, feat_num):
    """
    构建稀疏特征的字典。

    参数:
    feat (str): 特征名
    feat_num (int): 特征的唯一取值数量

    返回:
    dict: 包含特征名和特征值数量的字典
    """
    return {'feat': feat, 'feat_num': feat_num}


# 定义数值特征的函数，返回特征名
def dense_feat(feat):
    """
    构建数值特征的字典。

    参数:
    feat (str): 特征名

    返回:
    dict: 包含特征名的字典
    """
    return {'feat': feat}


# 可以用下面的函数替代上面的这个
def process_dense_feats(data, feats):
    """
    使用 StandardScaler 对数值特征进行标准化处理。

    参数:
    data (DataFrame): 包含特征的数据框
    feats (list): 数值特征列名列表

    返回:
    DataFrame: 处理后的数据
    scaler: 训练好的标准化器（可保存用于线上预测）
    """
    data[feats] = data[feats].fillna(0)

    scaler = StandardScaler()
    data[feats] = scaler.fit_transform(data[feats])

    return data


# 处理稀疏特征，填充缺失值并编码
def process_sparse_feats(data, feats):
    """
    对稀疏特征进行处理，填充缺失值并使用 Label Encoding 编码。
    使用 LabelEncoder 的核心目的之一就是：将类别型特征转换为从 0 开始的整数索引，以便构建 embedding 矩阵时可以作为下标使用。
    参数:
    data (DataFrame): 包含特征的数据框
    feats (list): 稀疏特征的列名列表

    返回:
    DataFrame: 处理后的数据框
    """
    # data[feats] = data[feats].fillna('-1')  # 填充缺失值为字符串 '-1'
    # 只读取指定的列
    # columns_to_read = ['label','I1','I2', 'C1', 'C2','C3']  # 替换为实际的列名
    # data = pd.read_csv(file, usecols=columns_to_read)

    # 对每个稀疏特征进行 Label Encoding 编码
    label_encoders_dict = {} # 后面可能会用到
    for f in tqdm(feats, desc='process_sparse_feats'):
        label_encoder = LabelEncoder()  # 创建 LabelEncoder 实例
        data[f] = label_encoder.fit_transform(data[f])  # 编码特征
        label_encoders_dict[f] = label_encoder

    # for feat, encoder in label_encoders.items():
    #     # print(f"Feature: {feat}")
    #     mapping = {cls: idx for idx, cls in enumerate(encoder.classes_)}
    #     # print("原始值 -> 编码值 映射前5项:", list(mapping.items())[:5])
    """
    Feature: item_id
    原始值 -> 编码值 映射前5项: [(224, 0), (426, 1), (1565, 2), (4273, 3), (4297, 4)]
    """
    return data,label_encoders_dict


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
        padded = pad_sequences(sequences, padding='post')
        pad_sequences_dict[feature] = padded
        tokenizers[feature] = tokenizer
        pad_len_dict[feature] = padded.shape[1]

    return pad_sequences_dict, tokenizers, pad_len_dict



# === 变长序列特征处理 ===
def process_history_sequence_feats(data, history_sequence_feats, history_sequence_label_encoder_map_config, label_encoders_dict, maxlen=5):
    """
    对已经是整数 ID 表示的历史序列特征进行处理：分割 -> 编码 -> padding。

    参数说明：
    :param data: pd.DataFrame，原始数据集，包含若干历史序列特征列（如 'history_item_ids'）
    :param history_sequence_feats: List[str]，历史序列特征名列表，例如 ['history_item_ids', 'history_citys']
    :param history_sequence_label_encoder_map_config: Dict[str, str]，每个历史序列特征对应其主特征的编码器名，例如：
           {
               "history_item_ids": "item_id",
               "history_citys": "item_city"
           }
    :param label_encoders: Dict[str, LabelEncoder]，主特征名 -> 已训练好的 LabelEncoder 对象
    :param maxlen: int，序列最大长度，多余部分会被截断，不足会用 0 补齐（post-padding）

    :return: pd.DataFrame，处理后的数据，历史序列列中的值为长度固定的整数列表（padding 后结果）
    """
    pad_hist_sequences_dict = {}
    for feature in history_sequence_feats:
        # 1. 分割原始字符串为 token 序列（例如 '12,45,7' -> ['12', '45', '7']）
        sequences = data[feature].fillna('').apply(lambda x: x.split(','))

        # 2. 获取当前序列特征所对应的主特征编码器（如 history_item_ids -> item_id 的 LabelEncoder）
        encoder = label_encoders_dict[history_sequence_label_encoder_map_config[feature]]

        # 3. 构建 token -> index 映射表（使用 str(cls) 是因为有的 LabelEncoder 中的类为字符串）
        token2index = {str(cls): idx for idx, cls in enumerate(encoder.classes_)}

        # 4. 对每个 token 编码，若未登录则映射为 0
        encoded_sequences = sequences.apply(lambda seq: [token2index.get(token, 0) for token in seq])

        # 5. 进行 padding（后补零）统一长度
        padded = pad_sequences(encoded_sequences.tolist(), padding='post', maxlen=maxlen)

        # 6. 用填充后的整数序列更新原始列
        data[feature] = list(padded)
        pad_hist_sequences_dict[feature] = padded    # 这是后来新增加的代码

    return data,pad_hist_sequences_dict



def main():
    # === 1. 读取数据 ===
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel","finish", "like", "music_id", "device", "time", "duration_time", "actors", "genres", "hist_item_id", "hist_item_city"]
    data = pd.read_csv(r"D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\train_2_with_history.csv", sep='\t', names=column_names)
    print(data.head(5))

    # 特征定义
    dense_feats = ["time", "duration_time"]
    sparse_feats = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
    sequence_feats = ['actors', 'genres']   # 有的时候可能为空列表 []
    # Notice: History behavior sequence feature name must start with "hist_". !!!!!
    history_sequence_feats = ["hist_item_id", "hist_item_city"]   # 增加变长历史序列数据

    # 下面2个配置其实可以合并为一个，历史原因:
    # 历史序列特征 -> 主特征编码器名称的映射配置（用于 LabelEncoder 查找）
    history_sequence_label_encoder_map_config = {"hist_item_id":'item_id',"hist_item_city":'item_city'}
    # 历史序列特征 -> 主特征 embedding 源的映射配置（用于模型构建时查找共享 embedding）
    history_sequence_emb_map_config = [{'feat': 'hist_item_id', 'target_emb_column': 'item_id', 'target_item_index': 2}, {'feat': 'hist_item_city','target_emb_column': 'item_city','target_item_index': 4}]

    target = ['finish']  # 推荐任务目标

    # === 2. 特征处理 ===
    # 对数值特征、稀疏特征、序列特征进行处理
    data = process_dense_feats(data,  dense_feats)
    # 你必须对历史序列字段和当前字段使用相同的 LabelEncoder 实例进行 transform，不能重新 fit
    data, label_encoders_dict  = process_sparse_feats(data, sparse_feats)

    pad_sequences_dict, tokenizers, pad_len_dict = process_sequence_feats(data, sequence_feats) if sequence_feats else ({}, {}, {})

    if history_sequence_feats:
        data,pad_hist_sequences_dict = process_history_sequence_feats(data,history_sequence_feats,history_sequence_label_encoder_map_config,label_encoders_dict)
    for hist_feat in history_sequence_feats: # 统计每个序列特征的有效长度（非零元素数量）
        data[f"{hist_feat}_length"] = data[hist_feat].apply(lambda x: np.count_nonzero(x))
    print(data.head(5))

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

    if history_sequence_feats:
        for feat in history_sequence_feats:
            target_feat = history_sequence_label_encoder_map_config[feat]
            fixlen_feature_columns.append(
                VarLenSparseFeat(
                    SparseFeat(feat, vocabulary_size=data[target_feat].nunique() + 1, embedding_dim=4,embedding_name=target_feat),
                    maxlen=5,combiner='mean',
                    length_name=f"{feat}_length"
                )
            )


    fixlen_feature_names   = get_feature_names(fixlen_feature_columns)
    """
    !!!!变长序列的 length 字段 'history_item_id_length', 'history_item_city_length',DeepCTR 自动识别并加入模型输入fixlen_feature_names中。
    """

    # === 4. 划分训练集和测试集 ===
    train, test = train_test_split(data, test_size=0.2, random_state=2018)

    # 构建模型输入
    train_model_input = {name: train[name] for name in fixlen_feature_names}
    test_model_input  = {name: test[name]  for name in fixlen_feature_names}

    # 添加序列特征到输入
    # 把提前 pad 好的序列特征（如 genres、actors），按照 train/test 划分后的索引，分成训练和测试输入字典
    if sequence_feats:
        for feat in sequence_feats:
            train_model_input[feat] = pad_sequences_dict[feat][train.index]
            test_model_input[feat]  = pad_sequences_dict[feat][test.index]

    # 历史序列特征也这么处理一下，否则报错:Failed to convert a NumPy array to a Tensor
    if history_sequence_feats:
        for feat in history_sequence_feats:
            train_model_input[feat] = pad_hist_sequences_dict[feat][train.index]
            test_model_input[feat]  = pad_hist_sequences_dict[feat][test.index]

    # === 5. 构建和训练 DIN 模型 ===
    from deepctr.models import DIN
    model = DIN(
        dnn_feature_columns=fixlen_feature_columns,
        history_feature_list=["item_id", "item_city"],
        task='binary'
    )
    model.compile("adagrad", "binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name='auc')])


    # 输出目录结构
    base_dir = r'D:\software\pycharm_repository\StarMaker\MultiRecSys\outputs'
    log_dir = os.path.join(base_dir, 'logs')
    callbacks_dir = os.path.join(base_dir, 'callbacks')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(callbacks_dir, exist_ok=True)
    output_model_file = os.path.join(callbacks_dir, 'best_model.h5')

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_model_file,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1,
            save_format='tf'  # 显式指定格式
        )
    ]

    history = model.fit(
        train_model_input,
        train[target].values,
        batch_size=256,
        epochs=10,
        verbose=2,
        validation_split=0.2,
        callbacks=callbacks
    )



    # === 6. 模型评估(加载训练好的模型进行测试) ===
    from deepctr.models import DIN
    model = DIN(
        dnn_feature_columns=fixlen_feature_columns,
        history_feature_list=["item_id", "item_city"],
        task='binary'
    )
    model.load_weights(output_model_file)  # 只加载权重，不加载结构
    # 虽然在预测时用了batch_size=256，但是模型只是分批地处理数据，但最终会拼接每个 batch 的结果，返回一个完整的预测结果数组**
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("Test LogLoss:", round(log_loss(test[target].values, pred_ans, labels=[0, 1]), 4))
    print("Test AUC:", round(roc_auc_score(test[target].values, pred_ans), 4))

    # === 7. 可视化训练过程 ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['loss'], label='Train Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['auc'], label='Train AUC')
    plt.plot(history.epoch, history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.epoch, history.history['val_auc'], label='Val AUC')
    plt.plot(history.epoch, history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('AUC and Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
