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
        padded = pad_sequences(sequences, padding='post')
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
    target = ['finish']  # 推荐任务目标

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
    train, test = train_test_split(data, test_size=0.2, random_state=2018)

    # 构建模型输入
    train_model_input = {name: train[name] for name in fixlen_feature_names}
    test_model_input  = {name: test[name] for name in fixlen_feature_names}

    # 添加序列特征到输入
    # 把提前 pad 好的序列特征（如 genres、actors），按照 train/test 划分后的索引，分成训练和测试输入字典
    if sequence_feats:
        for feat in sequence_feats:
            train_model_input[feat] = pad_sequences_dict[feat][train.index]
            test_model_input[feat]  = pad_sequences_dict[feat][test.index]

    # === 5. 构建和训练 DeepFM 模型 ===
    from deepctr.models import DeepFM
    model = DeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
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
    from deepctr.models import DeepFM
    model = DeepFM(
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
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
