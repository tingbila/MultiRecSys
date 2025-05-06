部署在:
worker@sg-prod-research-worker-6
/data1/guifang.ji/DSSM_Recall

配置文件是 ./config/dssm_config.py
如果用到的user_id和sm_id、rating 特征名称不同，
需要修改u2i_dssm.py 和 dssm_preprocess.py 中对应的特征名

u2i_dssmrecall.sh 为任务入口
写redis的配置在write_pika_pipeline.py中修改
data_path 和 config/dssm_config.py  中的 save_dir 对应

上游数据生成任务在：
hadoop@sg-prod-algorithmv2-emrmaster-1
/data/guifang.ji/pyspark_code/popular_rec

data_cloums 特征名和生成数据的要一一对应
