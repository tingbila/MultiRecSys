# coding=utf-8
import time
import sys
from const import *
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
from datetime import datetime
from pyspark.sql import SparkSession
import json
import requests

schema = StructType(fields)


def convert_type(raw_feature, param):
    if param == "string":
        return str(raw_feature)
    if param == "float":
        return float(raw_feature)
    return raw_feature


def feature_transform(df):
    user_id = convert_type(df.user_id, "string")
    song_id = convert_type(df.song_id, "string")
    return user_id, song_id, df.server_timestamp, df.user_level, df.user_language, df.platform, df.country, df.gender, \
           df.age, df.song_lang_id, df.artist_gender, df.song_quality, df.song_recording_count, df.sgenres, \
           df.song_create_time, df.artist_country_id, df.region, df.is_new


def get_region():
    urls = "http://sg-internal-goapi.starmaker.co/internal/all-regions"
    ret = requests.get(url=urls).text
    m = []
    try:
        m = json.loads(ret)
    except Exception as e:
        print(e)
    return m


def map_region(reg_map):
    def process(x, reg_map):
        for k, v in reg_map.items():
            for c in v:
                if c == x:
                    return k.split("_")[1]
        return "other"

    return F.udf(lambda x: process(x, reg_map))


def get_data_frame(dt, cols):
    res = spark.read.option("mergeSchema", "true").parquet(
        "cosn://starmaker-research-sg-1256122840/statistics/sing_feature_table_new/dt=%s/*" % dt).select(cols).where("is_publish=1")
    return res


if __name__ == '__main__':
    days = int(sys.argv[1] if len(sys.argv) > 1 else 14)
    start_dt = time.strftime("%Y%m%d", time.localtime(time.time() - 24 * 60 * 60))
    #start_dt = "20220807"
    spark = SparkSession.builder \
        .appName("song_train_data_%s_%d" % (start_dt, days)) \
        .enableHiveSupport() \
        .config("spark.port.maxRetries", "100") \
        .getOrCreate()

    col_used = ["is_publish", "user_id", "song_id", "server_timestamp", "user_level", "user_language", "platform", "country",
                "gender", "age", "song_lang_id", "artist_gender", "song_quality", "song_recording_count", "song_genres",
                "sing_lang", "song_create_time", "artist_country_id", "is_new"]

    data_frame = get_data_frame(start_dt, col_used)
    '''
    for i in range(1, days):
        try:
            tmp_dt = time.strftime("%Y%m%d", time.localtime(time.time() - 24 * (i + 1) * 60 * 60))
            data_frame_tmp = get_data_frame(tmp_dt, col_used)
            data_frame = data_frame.union(data_frame_tmp)
        except:
            continue
    '''
    # data_frame.show(5)
    print "start......"
    data_path = 'cosn://starmaker-research-sg/multi_task_data/song_feature_publish_%s' % (start_dt)

    # data_frame = data_frame.where("is_new=0")
    firstelement = F.udf(lambda v: v[0] if len(v) > 0 else "", StringType())
    data_frame = data_frame.withColumn("sgenres", firstelement(data_frame["song_genres"]))

    df = data_frame
    df2 = df.select("user_id").groupBy('user_id').count().where("count>=2")  # 23万 14天 #3 20200125
    df2 = df2.where("count<=150")
    # print(df2.select(F.count('user_id')).collect())
    data_final = df2.join(df, df.user_id == df2.user_id, "left").drop(df2.user_id)  # 小表join大表

    reg_dict = get_region()
    df3 = df.select("song_id").groupBy('song_id').count().where("count>=2")
    data_final = df3.join(data_final, data_final.song_id == df3.song_id, "left").drop(df3.song_id)
    data_final = data_final.withColumn("region", map_region(reg_dict)(data_final["country"]))
    # print(data_final.count())

    features = data_final.rdd.map(feature_transform).filter(lambda x: x is not None)
    df_handle = spark.createDataFrame(features)
    # df_handle.show(5)
    print("start write files")
    df_handle.repartition(3).write.format("csv").option("header", "false").mode("overwrite").save(data_path)
    print("end!")

