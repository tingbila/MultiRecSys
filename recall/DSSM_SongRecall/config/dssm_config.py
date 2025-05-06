# coding = utf-8
import tensorflow as tf

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\dssm_data_2.csv", "The input data dir")
flags.DEFINE_string("data_final_dir", "/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_finash.csv", "The input data dir")
flags.DEFINE_integer("seq_len", 15, "seq len")
flags.DEFINE_integer("min_count", 5, "min count item click")
flags.DEFINE_integer("negsample", 3, "negsample num")
flags.DEFINE_integer("embedding_dim", 30, "embedding_dim")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("epochs", 3, "train epoches")
flags.DEFINE_float("validation_split", 0.0, "validation split")
flags.DEFINE_integer("layer_embeding", 32, "layer embeding")
flags.DEFINE_integer("pred_topk", 200, "pred top")     # 选取多少召回
flags.DEFINE_integer("recall_topk", 50, "recall topk")   # 召回覆盖率
flags.DEFINE_string("save_dir", "/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_u2i.txt", "The input data dir")
flags.DEFINE_string("save_dir_new", "/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_new.txt", "The input data dir")
flags.DEFINE_string("save_sdm_dir", "/data1/guifang.ji/DSSM_SongRecall/data/sdm_data_u2i.txt", "The input data dir")
flags.DEFINE_string("save_mind_dir", "/data1/guifang.ji/DSSM_SongRecall/data/mind_data_u2i.txt", "The output data dir")
flags.DEFINE_string("save_final_dir", "/data1/guifang.ji/DSSM_SongRecall/data/dssm_final_u2i.txt", "The input data dir")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

user_hidden_unit = (64, 32)
item_hidden_unit = (64, 32)

# publish 训练特征
data_cloums = ["user_id", "sm_id", "timestamp", "level", "user_lang", "platform", "country",
            "gender", "age", "song_lang_id", "artist_gender", "song_quality", "song_recording_count",
            "song_genres", "song_create_time", "artist_country_id", "region", "is_new"]

features = ["user_id", "sm_id", "level", "user_lang", "platform",
            "country", "gender", "artist_gender", "age", "is_new",
            "song_quality", "song_recording_count", "song_genres"]

num_header = ["level", "gender", "age", "artist_gender", "is_new", "song_quality", "song_recording_count"]
string_header = ["user_id", "sm_id", "user_lang", "platform", "country", "song_genres"]
time_header = ["timestamp"]
user_fearther = ["user_id", "gender", "age", "level", "user_lang", "country", "platform", "is_new"]
item_fearther = ["sm_id", "artist_gender", "song_quality", "song_recording_count", "song_genres"]  # "sm_language" 90% 为空，去掉
user_fearther_noid = ["gender", "age", "level", "user_lang", "country", "platform", "is_new"]
item_fearther_noid = ["artist_gender", "song_quality", "song_recording_count", "song_genres"]

# 录制训练特征
final_col = ["user_id", "sm_id", "timestamp", "level", "user_lang", "platform", "country",
            "gender", "age", "song_lang_id", "artist_gender", "song_quality", "song_recording_count",
            "song_genres", "song_create_time", "artist_country_id"]


#0807