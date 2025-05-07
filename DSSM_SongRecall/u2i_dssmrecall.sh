cd /data1/guifang.ji/DSSM_SongRecall
day=$(date -d "0 day ago" "+%Y-%m-%d_%H:%M:%S")
echo $day

echo "get_youtube_data start"
/data/anaconda2/bin/python /data1/guifang.ji/DSSM_SongRecall/download.py
if [ $? != 0 ]; then
    echo "get_dssm_data faild"
#    curl "http://devops-callback.ushow.media:9910/v1/api/message?title=get_youtube_data_faild&toUser=ZaiLuShang&text=$day"
    curl -H "Content-Type:application/json" -X POST -d '{"type": "Card", "touser": ["guifang.ji"], "id": 5, "data": {"title": "get_dssm_data_faild", "msg": "get_dssm_data_faild"}}' https://devops.ushow.media/devops-goserver-v1/notify/message/
    exit 1
fi

echo "u2i_dssm training start"
/home/worker/anaconda3/bin/python3.7 /data1/guifang.ji/DSSM_SongRecall/u2i_dssm.py
if [ $? != 0 ]; then
    echo "u2i_dssm faild"
#    curl "http://devops-callback.ushow.media:9910/v1/api/message?title=u2i_youtube_faild&toUser=ZaiLuShang&text=$day"
    curl -H "Content-Type:application/json" -X POST -d '{"type": "Card", "touser": ["guifang.ji"], "id": 5, "data": {"title": "u2i_dssm_faild", "msg": "u2i_dssm_faild"}}' https://devops.ushow.media/devops-goserver-v1/notify/message/
    exit 1
fi

echo "write pika start"
/data/anaconda2/bin/python /data1/guifang.ji/DSSM_SongRecall/write_pika_pipeline.py "song:dssmU2i:%s" "/data1/guifang.ji/DSSM_SongRecall/data/dssm_data_u2i.txt"
if [ $? != 0 ]; then
    echo "write pika faild"
    curl "http://devops-callback.ushow.media:9910/v1/api/message?title=write_pika_faild&toUser=ZaiLuShang&text=$day"
    curl -H "Content-Type:application/json" -X POST -d '{"type": "Card", "touser": ["guifang.ji"], "id": 5, "data": {"title": "write_pika_faild", "msg": "write_pika_faild"}}' https://devops.ushow.media/devops-goserver-v1/notify/message/
    exit 1
fi




