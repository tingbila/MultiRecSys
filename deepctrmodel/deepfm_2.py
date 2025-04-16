import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM
from tensorflow import keras
import tensorflow.python.keras.engine.data_adapter as data_adapter
import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个

# --- 修复 DeepCTR 与新版 TensorFlow 的数据适配器兼容性 ---
def _is_distributed_dataset_fixed(ds):
    return False

data_adapter._is_distributed_dataset = _is_distributed_dataset_fixed
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")


# --- 1. 数据加载 ---
def load_data(file_path, column_names):
    data = pd.read_csv(file_path, sep='\t', names=column_names)
    return data


# --- 2. 特征预处理 ---
def preprocess_features(data, sparse_feats, dense_feats):
    # 填补缺失值
    data[sparse_feats] = data[sparse_feats].fillna("missing")
    data[dense_feats]  = data[dense_feats].fillna(0)

    # 编码稀疏特征
    for feat in sparse_feats:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])

    # 标准化密集特征
    scaler = StandardScaler()
    data[dense_feats] = scaler.fit_transform(data[dense_feats])

    return data



# --- 3. 构建特征列 ---
def build_feature_columns(data, sparse_feats, dense_feats):
    """
    构建 DeepCTR 模型所需的特征列（Feature Columns）。
    参数说明：
    ----------
    data : pd.DataFrame         包含所有样本的原始特征数据（稀疏+密集）。
    sparse_feats : list of str  稀疏特征名列表，例如 ["uid", "item_id", ...]。
    dense_feats : list of str   数值（密集）特征名列表，例如 ["time", "duration_time"]。

    返回值：
    --------
    feature_columns : list
        DeepCTR 所需的特征列对象，包括：
        - SparseFeat（稀疏特征，每个特征将映射为稀疏 Embedding 向量）
        - DenseFeat（密集特征，保留原始数值）
        示例：[SparseFeat(name='uid', vocabulary_size=10001, embedding_dim=4),DenseFeat(name='duration_time', dimension=1),...]

    feature_names : list of str
        特征名称列表，用于模型输入字典的构建。
        示例：['uid', 'user_city', 'item_id', ..., 'duration_time']
    """
    feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique() + 1, embedding_dim=4) for feat in sparse_feats] \
                      + [DenseFeat(feat, 1) for feat in dense_feats]

    return feature_columns, get_feature_names(feature_columns)


# --- 4. 模型构建与训练 ---
def build_and_train_model(train, feature_names, feature_columns, target):
    train_input = {name: train[name] for name in feature_names}   # 传入的是字段名称

    # 传入的是各种字段元数据信息-Embedding信息等
    model = DeepFM(
        linear_feature_columns=feature_columns,
        dnn_feature_columns=feature_columns,
        task='binary'
    )
    model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy', keras.metrics.AUC(name='auc')])
    history = model.fit(train_input, train[target].values, batch_size=256, epochs=10, verbose=2, validation_split=0.2)
    return model, history


# --- 5. 模型评估 ---
def evaluate_model(model, test, fixlen_feature_names, target):
    test_input = {name: test[name] for name in fixlen_feature_names}
    pred_ans = model.predict(test_input, batch_size=256)
    logloss = round(log_loss(test[target].values, pred_ans, labels=[0, 1]), 4)
    auc = round(roc_auc_score(test[target].values, pred_ans), 4)
    print("test LogLoss:", logloss)
    print("test AUC:", auc)
    return pred_ans


# --- 6. 训练过程可视化 ---
def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['loss'], label='Train Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # AUC 和 Accuracy 曲线
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



# --- 主函数入口 ---
def main():
    column_names = ["uid", "user_city", "item_id", "author_id", "item_city", "channel",
                    "finish", "like", "music_id", "device", "time", "duration_time"]
    sparse_feats = ["uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"]
    dense_feats = ["time", "duration_time"]
    target = ['finish']

    data = load_data(r'D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\train_2.csv', column_names)
    data = preprocess_features(data, sparse_feats, dense_feats)
    feature_columns, feature_names = build_feature_columns(data, sparse_feats, dense_feats)
    train, test = train_test_split(data, test_size=0.2, random_state=2018)

    model, history = build_and_train_model(train, feature_names, feature_columns, target)
    evaluate_model(model, test, feature_names, target)
    plot_metrics(history)



if __name__ == '__main__':
    main()
