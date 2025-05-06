# coding = utf-8
import os
import sys
import logging
import time
import requests
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client

BUCKET_NAME = 'starmaker-research'
BUCKET_NAME_WITH_ID = 'starmaker-research-sg-1256122840'
# SECRET_ID = 'AKIDDLL9lBBRHvMbhb5n6FCFCxZzBEZmkiNu'
SECRET_ID = 'AKIDoQmshFWXGitnQmrfCTYNwEExPaU6RVHm'
# SECRET_KEY = 'UmvuqIbjeN83qVyHrUwCIF4iATZYNMFw'
SECRET_KEY = 'F9n9E2ZonWy93f04qMaYFfogHadPt62h'
REGION = 'ap-singapore'
REMOTE_YOUTUBEDNN_SESSION_PATH = 'multi_task_data/%s/'
LOCAL_YOUTUBEDNN_SESSION_PATH = '/data1/guifang.ji/DSSM_SongRecall/data'
FILE_NAME = '/dssm_data_finash.csv'

logging.basicConfig(level=logging.FATAL, stream=sys.stdout)
COS_CONFIG = CosConfig(Secret_id=SECRET_ID, Secret_key=SECRET_KEY, Region=REGION)
COS_CLIENT = CosS3Client(COS_CONFIG)


def download_file(FILE_DATES):
    file_path = LOCAL_YOUTUBEDNN_SESSION_PATH + FILE_NAME
    if os.path.exists(file_path):
        os.system("rm -f %s" % file_path)

    for file_date in FILE_DATES:
        print(REMOTE_YOUTUBEDNN_SESSION_PATH % file_date)
        if COS_CLIENT.object_exists(BUCKET_NAME_WITH_ID, (REMOTE_YOUTUBEDNN_SESSION_PATH + "_SUCCESS") % file_date):
            os.system("coscmd download -r %s %s" % (REMOTE_YOUTUBEDNN_SESSION_PATH % file_date, LOCAL_YOUTUBEDNN_SESSION_PATH))
        else:
            send_warnings()
        os.chdir(LOCAL_YOUTUBEDNN_SESSION_PATH)

        print(os.getcwd())
        os.system("cat part-* >> %s" % file_path)
        os.system("rm -rf part-* _*")


def send_warnings():
    data = '{"type": "Card", "touser": ["guifang.ji"], "id": 9, "data": {"title": "song_eges", "msg": "get data error..."}}'
    requests.post(url="https://devops.ushow.media/devops-goserver-v1/notify/message/", data=data)


if __name__ == '__main__':
    TODAY = time.strftime("%Y%m%d", time.localtime(time.time() - 24 * 60 * 60 * 2))
    print(TODAY)
    FILE_DATE = "song_finish_feature_%s"
    days = 7
    FILE_DATES = []
    for i in range(2, days + 2):
        day = time.strftime("%Y%m%d", time.localtime(time.time() - 24 * 60 * 60 * i))
        FILE_DATES.append(FILE_DATE % (day))

    print(FILE_DATES)
    download_file(FILE_DATES)
    # cosn://starmaker-research-sg/multi_task_data/dssm_songfeature_20220726_21


