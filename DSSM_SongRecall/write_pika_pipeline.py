# coding = utf-8
import redis
import time
import sys

REDIS_HOST = "sg-proxy01.starmaker.co"
REDIS_PORT = 22122
REDIS_SAVETIME = 86400 * 7

write_redis_knn_num = 0.0
pipe = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT).pipeline(transaction=False)


def write_data_to_redis(data_dict, key):
    print(key)
    REDIS_KEY = key
    global write_redis_knn_num
    for sm_id, id_with_l2_list in data_dict.items():
        key = REDIS_KEY % int(sm_id)
        try:
            pipe.delete(key)
            pipe.rpush(key, *id_with_l2_list)
            write_redis_knn_num += 1
            pipe.expire(key, REDIS_SAVETIME)  # 14,20210125,test #7 20210226 test //21 normal
            if write_redis_knn_num % 1000 == 0:
                pipe.execute()
                if write_redis_knn_num % 10000 == 0:
                    print("Batch: %d" % write_redis_knn_num)
        except Exception as e:
            print('redis error!', e)
            sys.exit(1)
        # break
    print(key)
    pipe.execute()
    if write_redis_knn_num % 1000 == 0:
        print("redis rpush end", write_redis_knn_num)


if __name__ == '__main__':
    key = sys.argv[1]
    data_path = sys.argv[2]
    start = time.time()
    img_dist_dict = {}
    #print(key)
    with open(data_path, 'r') as fp:
        count = 0.0
        for data in fp:
            count = count + 1
            data = data.strip().split('\t')
            if len(data) < 2:
                continue
            dist = data[1].strip().split(',')
            img_dist_dict[data[0]] = dist

    print(len(img_dist_dict))
    write_data_to_redis(img_dist_dict, key)
    print('write_num:', write_redis_knn_num)
    print('write redis time:', time.time() - start, 's')

