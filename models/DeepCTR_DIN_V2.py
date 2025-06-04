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



def process_history_sequence_feats(data, hist_behavior_feats, max_len=5):
    """
    对历史行为序列特征进行统一长度的 padding 操作，方便后续输入模型。

    参数:
    data (DataFrame): 包含历史行为序列特征的数据框
    hist_behavior_feats (list): 历史行为序列特征名称列表（例如用户点击过的文章ID序列）
    max_len (int): 每个序列统一补齐的最大长度，超过则截断，不足则后补 0（post-padding）

    返回:
    dict: 每个序列特征对应的 padding 后的 numpy 数组，形状为 (样本数, max_len)
    """
    pad_hist_sequences_dict = {}

    for feature in hist_behavior_feats:
        # 提取每个样本对应的历史行为序列
        his_list = [l for l in data[feature]]

        # 将每个序列统一补齐或截断为固定长度（后补零）
        padded = pad_sequences(his_list, maxlen=max_len, padding='post')

        # 存入字典，键为特征名，值为 padding 后的二维数组
        pad_hist_sequences_dict[feature] = padded

    return pad_hist_sequences_dict




def main():
    # === 1. 读取原始数据 ===
    data = pd.read_csv(r"D:\\software\\pycharm_repository\\StarMaker\\MultiRecSys\\data_files\\news_click_log.csv")
    data["is_click"] = np.random.randint(0, 2, size=len(data))  # 随机生成点击标签作为二分类目标

    # === 2. 构造用户历史点击序列（可提前通过 Hive SQL 离线聚合） ===
    hist_click = data[['user_id', 'click_article_id']].groupby('user_id').agg({list}).reset_index()
    his_behavior_df = pd.DataFrame()
    his_behavior_df['user_id'] = hist_click['user_id']
    his_behavior_df['hist_click_article_id'] = hist_click['click_article_id']

    # 合并历史点击序列到原始数据中
    data = data.merge(his_behavior_df, on='user_id')
    print(data.head(5))

    # === 3. 特征定义 ===
    dense_feats = ["click_timestamp"]
    sparse_feats = ['user_id', 'click_article_id', 'click_environment', 'click_deviceGroup','click_os', 'click_country', 'click_region', 'click_referrer_type']
    hist_behavior_feats = ["hist_click_article_id"]  # 注意：历史行为特征必须以 "hist_" 开头
    history_sequence_label_encoder_map_config = {"hist_click_article_id": 'click_article_id'}
    target = ['is_click']

    # === 4. 特征预处理 ===
    data = process_dense_feats(data, dense_feats)  # 标准化数值特征

    if hist_behavior_feats:
        pad_hist_sequences_dict = process_history_sequence_feats(data, hist_behavior_feats)  # 对序列特征进行 padding

    # 统计每个变长历史序列的实际长度
    for hist_feat in hist_behavior_feats:
        data[f"{hist_feat}_length"] = np.count_nonzero(pad_hist_sequences_dict[hist_feat], axis=1)
    print(data.head(5))

    # === 5. 构建特征列定义（FixLen 和 VarLen） ===
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4) for feat in sparse_feats
    ] + [
        DenseFeat(feat, 1) for feat in dense_feats
    ]

    # 构造变长序列特征列
    if hist_behavior_feats:
        for hist_feat in hist_behavior_feats:
            target_feat = history_sequence_label_encoder_map_config[hist_feat]
            fixlen_feature_columns.append(
                VarLenSparseFeat(
                    SparseFeat(hist_feat, vocabulary_size=data[target_feat].max() + 1, embedding_dim=4,embedding_name=target_feat),
                    maxlen=5,
                    combiner='mean',
                    length_name=f"{hist_feat}_length"
                )
            )

    fixlen_feature_names = get_feature_names(fixlen_feature_columns)
    print(fixlen_feature_names)
    # DeepCTR 会自动将变长序列的主特征与 length 特征加入输入中

    # === 6. 划分训练和测试数据 ===
    train, test = train_test_split(data, test_size=0.2, random_state=2018)

    # 构建模型输入字典（除序列特征外）
    train_model_input = {name: train[name] for name in fixlen_feature_names if name not in hist_behavior_feats}
    test_model_input  = {name: test[name] for name in fixlen_feature_names  if name not in hist_behavior_feats}

    # 添加序列特征的 padding 输入
    if hist_behavior_feats:
        for feat in hist_behavior_feats:
            train_model_input[feat] = pad_hist_sequences_dict[feat][train.index]
            test_model_input[feat] = pad_hist_sequences_dict[feat][test.index]

    # === 7. 构建和训练 DIN 模型 ===
    from deepctr.models import DIN
    model = DIN(
        dnn_feature_columns=fixlen_feature_columns,
        history_feature_list=["click_article_id"],  # DIN 内部自动处理与 hist_click_article_id 匹配
        task='binary'
    )
    model.compile("adagrad", "binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name='auc')])

    # === 8. 设置训练日志与回调 ===
    base_dir = r'D:\\software\\pycharm_repository\\StarMaker\\MultiRecSys\\outputs'
    log_dir = os.path.join(base_dir, 'logs')
    callbacks_dir = os.path.join(base_dir, 'callbacks')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(callbacks_dir, exist_ok=True)
    output_model_file = os.path.join(callbacks_dir, 'best_model.h5')

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=output_model_file,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1,
            save_format='tf'
        )
    ]

    # === 9. 模型训练 ===
    history = model.fit(
        train_model_input,
        train[target].values,
        batch_size=256,
        epochs=10,
        verbose=2,
        validation_split=0.2,
        callbacks=callbacks
    )

    # === 10. 模型评估 ===
    model.load_weights(output_model_file)  # 重新加载最优权重
    pred_ans = model.predict(test_model_input, batch_size=256)
    print(pred_ans[:5])  # 展示前5个预测结果
    print("Test LogLoss:", round(log_loss(test[target].values, pred_ans, labels=[0, 1]), 4))
    print("Test AUC:", round(roc_auc_score(test[target].values, pred_ans), 4))

    # === 11. 可视化训练过程 ===
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
