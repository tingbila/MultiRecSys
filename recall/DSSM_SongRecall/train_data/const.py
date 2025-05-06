# coding=utf-8
#import tensorflow as tf
from pyspark.sql.types import *

# 242
COUNTRY_LIST = ["AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", "AU", "AW", "AX", "AZ", "BA", "BB",
                "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BM", "BN", "BO", "BQ", "BR", "BS", "BT", "BW", "BY", "BZ",
                "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN", "CO", "CR", "CU", "CV", "CW", "CX",
                "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI", "FJ",
                "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF", "GG", "GH", "GI", "GJ", "GL", "GM", "GN", "GP",
                "GQ", "GR", "GS", "GT", "GU", "GY", "HK", "HM", "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN",
                "IO", "IQ", "IR", "IS", "IT", "JB", "JE", "JM", "JO", "JP", "JT", "KA", "KE", "KG", "KH", "KI", "KM",
                "KN", "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY",
                "MA", "MC", "MD", "ME", "MG", "MH", "MK", "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU",
                "MV", "MW", "MX", "MY", "MZ", "NA", "NC", "NE", "NG", "NI", "NL", "NO", "NP", "NR", "NZ", "OM", "PA",
                "PE", "PF", "PG", "PH", "PK", "PL", "PM", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU",
                "RW", "SA", "SB", "SC", "SD", "SE", "SG", "SI", "SK", "SL", "SM", "SN", "SO", "SR", "ST", "SV", "SX",
                "SY", "SZ", "TC", "TD", "TE", "TF", "TG", "TH", "TJ", "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW",
                "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI", "VN", "WF", "WS", "XK", "YE",
                "YT", "ZA", "ZM", "ZW"]

# 5
APP_NAME_LIST = ["sm", "sm_id", "sm_in", "stm", "ftp"]

#6
AGE_LIST = ["0", "1", "2", "3", "4", "5"]

# 30
LANGUAGE_LIST = ["ar", "as", "bho", "bn", "da", "de", "en", "es", "fr", "gu", "hi", "hry", "id", "in", "it", "ja",
                 "kn", "ko", "ml", "mr", "ms", "or", "pa", "pt", "raj", "ta", "te", "th", "vi", "zh"]

# 29
PROVINCE_LIST = ['AP', 'AS', 'BR', 'CH', 'CT', 'DL', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'MH', 'ML', 'MN', 'MP',
                 'NL', 'OR', 'PB', 'PY', 'RJ', 'SK', 'TG', 'TN', 'TR', 'UP', 'UT', 'WB']

# 2
PLATFORM_LIST = ["android", "ios"]

# 曲风
GENRES_LIST = ["Religious","Blues","Rock","Kids","Stage&Screen","&Country","Classical",
    "HipHop","Soul","Childrens","Brass&Military","Dance","Reggae","Holiday","Alternative","Non-Music",
    "Pop","Electronic","Jazz","TryYourLuck","Latin","Soundtrack","Traditional","Funk/Soul","Folk",
    "Opera","R&B","ContemporaryJazz","Country","World","Hip-Hop","unknown"]   #32

# 7
RECALL_REGION_LIST = ["IN", "ID", "ME", "PK", "US", "JP", "KR"]

GENDER_LIST = ['0', '1', '2', '3', '5']

R_PLATFORM_LIST = ['ios', 'android']

fields = [
    StructField("is_click", FloatType()),
    StructField("is_positive", FloatType()),
    StructField("scene", StringType()),
    StructField("user_id", StringType()),
    StructField("r_user_id", StringType()),
    StructField("sm_id", StringType()),
    StructField("index_feature", StringType()),
    StructField("recall_score", FloatType()),
    StructField("media_type_orgin", StringType()),
    StructField("media_type", FloatType()),
    StructField("r_duration", FloatType()),
    StructField("a_song_num", FloatType()),
    StructField("a_song_num_sm", FloatType()),
    StructField("a_recording_count", FloatType()),
    StructField("s_recording_count", FloatType()),
    StructField("r_performance_score", FloatType()),
    StructField("s_song_quality", StringType()),
    StructField("country", StringType()),
    StructField("province", StringType()),
    StructField("r_location_province", StringType()),
    StructField("r_local", StringType()),
    StructField("r_location_country", StringType()),
    StructField("sm_stat_origin", ArrayType(FloatType(), False)),
    StructField("sm_stat", ArrayType(FloatType(), False)),
    StructField("user_persona_data_origin", ArrayType(FloatType(), False)),
    StructField("user_persona_data", ArrayType(FloatType(), False)),
    StructField("numerical_feature", ArrayType(FloatType(), False)),
    StructField("time_feature", ArrayType(FloatType(), False)),
    StructField("onehot_feature", ArrayType(FloatType(), False)),
    StructField("multihot_feature", ArrayType(FloatType(), False)),
]

feature_float = ['is_click', 'is_positive', 'recall_score', 'media_type', 'r_duration', 'a_song_num', 'a_song_num_sm', \
                 'a_recording_count', 's_recording_count', 'r_performance_score']

# feature_float=['is_click','is_positive','media_type']

id_list = ['user_id', 'r_user_id', 'sm_id']

mul_feature = ['sm_stat_origin', 'sm_stat', 'user_persona_data_origin', 'user_persona_data', 'numerical_feature', \
               'time_feature', 'onehot_feature', 'multihot_feature']

# df.user_id, df.song_id, df.user_level, df.user_language, df.platform, df.province, df.country, df.gender, \
#             df.song_language, df.artist_gender, df.age

fields = [
    StructField("user_id", StringType()),
    StructField("song_id", StringType()),
    StructField("server_timestamp", IntegerType()),
    StructField("user_level", IntegerType()),
    StructField("user_language", StringType()),
    StructField("platform", StringType()),
    StructField("country", StringType()),
    StructField("gender", IntegerType()),
    StructField("age", IntegerType()),
    StructField("song_lang_id", IntegerType()),
    StructField("artist_gender", IntegerType()),
    StructField("song_quality", FloatType()),
    StructField("song_recording_count", IntegerType()),
    StructField("sgenres", StringType()),
    # StructField("sing_lang", StringType()),
    StructField("song_create_time", IntegerType()),
    StructField("artist_country_id", IntegerType()),
    StructField("region", StringType())
]

new_fields = [
    StructField("user_id", StringType()),
    StructField("song_id", StringType()),
    StructField("server_timestamp", IntegerType()),
    StructField("user_level", IntegerType()),
    StructField("user_language", StringType()),
    StructField("platform", StringType()),
    # StructField("province", StringType()),
    StructField("country", StringType()),
    StructField("gender", IntegerType()),
    StructField("song_language", StringType()),
    StructField("artist_gender", IntegerType()),
    StructField("age", IntegerType()),
    StructField("is_new", IntegerType()),
    StructField("song_quality", FloatType()),
    StructField("song_recording_count", IntegerType()),
    StructField("sgenres", StringType())
]

# "age", "is_new", "song_quality", "song_recording_count", "sgenres"
#

# df.user_id, df.song_id, df.user_level, df.user_language, df.platform, df.province, df.country, df.gender, \
#             df.song_language, df.artist_gender, df.age
'''
feature_des = {
    'user_id': tf.FixedLenFeature(shape=[1], dtype=tf.string),
    'song_id': tf.FixedLenFeature(shape=[1], dtype=tf.string),
    'user_level': tf.FixedLenFeature(shape=[1], dtype=tf.int32),
    'user_language': tf.FixedLenFeature(shape=[1], dtype=tf.string),
    'platform': tf.FixedLenFeature(shape=[1], dtype=tf.string),
    'province': tf.FixedLenFeature(shape=[1], dtype=tf.string),
    'country': tf.FixedLenFeature(shape=[1], dtype=tf.string),
    'gender': tf.FixedLenFeature(shape=[1], dtype=tf.int32),
    'song_language': tf.FixedLenFeature(shape=[1], dtype=tf.string),
    'artist_gender': tf.FixedLenFeature(shape=[1], dtype=tf.int32),
    'age': tf.FixedLenFeature(shape=[1], dtype=tf.int32),
}
'''
