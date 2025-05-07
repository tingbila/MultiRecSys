#!/usr/bin/env bash
source /etc/profile
source ~/.bashrc

day=$(date -d "0 day ago" "+%Y-%m-%d_%H:%M:%S")
echo $day

/usr/local/service/spark/bin/spark-submit --deploy-mode client --master yarn \
  --executor-memory 33G \
  --num-executors 6 \
  --executor-cores 11 \
  --conf spark.default.parallelism=150  \
  --conf spark.sql.shuffle.partitions=150 \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer  \
  --driver-memory 4g \
  dssm_data.py 1 #21 20210322

if [ $? != 0 ]; then
    echo "youtubeDnn data faild"
    #curl "http://devops-callback.ushow.media:9910/v1/api/message?title=youtubeDnn_data_faild&toUser=ZaiLuShang&text=$day"
    curl -H "Content-Type:application/json" -X POST -d '{"type": "Card", "touser": ["guifang.ji", "limei.du"], "id": 9, "data": {"title": "songdssm_data_faild", "msg": "songdssm_data_faild"}}' https://devops.ushow.media/devops-goserver-v1/notify/message/
    exit 1
fi


