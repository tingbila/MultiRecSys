# config/data_config.py

# 原始字段名
COLUMN_NAMES = [
    "uid", "user_city", "item_id", "author_id", "item_city", "channel",
    "finish", "like", "music_id", "device", "time", "duration_time"
]

# 离散特征
CATEGORICAL_COLS = [
    "uid", "user_city", "item_id", "author_id", "item_city", "channel", "music_id", "device"
]

# 数值特征
NUMERIC_COLS = ["time", "duration_time"]

# 目标列
TARGET_COLS = ["finish", "like"]

test_size = 0.2  # 测试集比例
batch_size = 2   # 批量大小
embed_dim = 5    # 嵌入维度
epochs = 20      # 训练轮数
lr = 0.002       # 学习率
file = r"D:\software\jupyter_code\criteo_sample.txt"  # 数据文件路径
















