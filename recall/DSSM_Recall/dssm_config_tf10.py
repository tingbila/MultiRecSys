# coding = utf-8
# 指定文件编码格式为 UTF-8，确保读取中文时不会乱码

import tensorflow as tf  # 导入 TensorFlow 库

# 使用 TensorFlow 1.x 的兼容模块定义命令行参数（flags 是 TF1 常用的参数方式）
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS  # 参数容器

# 定义各类输入路径参数（字符串类型）
flags.DEFINE_string("data_dir", "/data1/guifang.ji/DSSM_SongRecall/data/dssm_data.csv", "原始输入数据文件路径")
flags.DEFINE_string("data_final_dir", "/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_finash.csv", "处理后的数据文件路径")

# 模型训练相关的超参数（整数型）
flags.DEFINE_integer("seq_len", 15, "用户行为序列的最大长度")
flags.DEFINE_integer("min_count", 5, "最小曝光次数阈值（用于过滤低频 item）")
flags.DEFINE_integer("negsample", 3, "每个正样本对应的负采样个数")
flags.DEFINE_integer("embedding_dim", 30, "Embedding 向量的维度")
flags.DEFINE_integer("batch_size", 256, "训练批次大小")
flags.DEFINE_integer("epochs", 3, "训练轮数")
flags.DEFINE_float("validation_split", 0.0, "验证集划分比例（0 表示不使用验证集）")
flags.DEFINE_integer("layer_embeding", 32, "隐藏层中间 embedding 层维度")
flags.DEFINE_integer("pred_topk", 200, "预测时选取 Top-K 商品用于召回")
flags.DEFINE_integer("recall_topk", 50, "评估时使用的召回 Top-K 范围")

# 中间及最终结果保存路径（字符串类型）
flags.DEFINE_string("save_dir", "/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_u2i.txt", "训练后结果保存路径")
flags.DEFINE_string("save_dir_new", "/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_new.txt", "新的结果保存路径")
flags.DEFINE_string("save_sdm_dir", "/data1/guifang.ji/DSSM_SongRecall/data/sdm_data_u2i.txt", "SDM 模型结果保存路径")
flags.DEFINE_string("save_mind_dir", "/data1/guifang.ji/DSSM_SongRecall/data/mind_data_u2i.txt", "MIND 模型结果保存路径")
flags.DEFINE_string("save_final_dir", "/data1/guifang.ji/DSSM_SongRecall/data/dssm_final_u2i.txt", "最终召回结果保存路径")

# 设置日志等级为 INFO（用于打印训练过程信息）
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# 定义用户和物品塔的 DNN 隐藏层结构（每层神经元个数）
user_hidden_unit = (64, 32)
item_hidden_unit = (64, 32)

# 训练数据字段名称列表（对应 CSV 文件的列）
data_cloums = ["user_id", "sm_id", "timestamp", "level", "user_lang", "platform", "country",
               "gender", "age", "song_lang_id", "artist_gender", "song_quality", "song_recording_count",
               "song_genres", "song_create_time", "artist_country_id", "region", "is_new"]

# 模型使用的特征字段
features = ["user_id", "sm_id", "level", "user_lang", "platform",
            "country", "gender", "artist_gender", "age", "is_new",
            "song_quality", "song_recording_count", "song_genres"]

# 数值型字段（可直接归一化/填充处理）
num_header = ["level", "gender", "age", "artist_gender", "is_new", "song_quality", "song_recording_count"]

# 字符串型字段（需要 LabelEncoder 编码）
string_header = ["user_id", "sm_id", "user_lang", "platform", "country", "song_genres"]

# 时间字段（如需用于建模可做时间差/周期处理）
time_header = ["timestamp"]

# 用户侧特征字段（含 user_id）
user_fearther = ["user_id", "gender", "age", "level", "user_lang", "country", "platform", "is_new"]

# 物品侧特征字段（含 sm_id）
item_fearther = ["sm_id", "artist_gender", "song_quality", "song_recording_count", "song_genres"]  # 注：去掉了缺失率高的字段

# 用户侧特征字段（去除 user_id，用于塔结构建模）
user_fearther_noid = ["gender", "age", "level", "user_lang", "country", "platform", "is_new"]

# 物品侧特征字段（去除 sm_id，用于塔结构建模）
item_fearther_noid = ["artist_gender", "song_quality", "song_recording_count", "song_genres"]

# 模型训练最终需要的字段（供写出/转存）
final_col = ["user_id", "sm_id", "timestamp", "level", "user_lang", "platform", "country",
             "gender", "age", "song_lang_id", "artist_gender", "song_quality", "song_recording_count",
             "song_genres", "song_create_time", "artist_country_id"]
