import pandas as pd
import numpy as np
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from model.sdm import SDM
# from .utils import sampledsoftmaxloss, NegativeSampler
from sdm_preprocess import gen_data_set_sdm, gen_model_input_sdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
import sys
sys.path.append("../recall")
from layers.utils import sampledsoftmaxloss, NegativeSampler
from config.dssm_config import *


if __name__ == "__main__":
    data_file = FLAGS.data_dir
    SEQ_LEN = FLAGS.seq_len
    min_count = FLAGS.min_count
    negsample = FLAGS.negsample
    batch_size = FLAGS.batch_size
    epoch = 4
    validation_split = FLAGS.validation_split
    user_dnn_hidden_units = user_hidden_unit
    item_dnn_hidden_units = item_hidden_unit

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    # data = pd.read_csv(data_file, names=data_cloums, sep=',').head(2000)
    data = pd.read_csv("./tmp_datas.csv", names=data_cloums, sep=',')
    print(data.shape[0])
    data['user_id'] = data['user_id'].replace('None', np.nan)
    data.dropna(subset=['user_id'], inplace=True)
    data['sm_id'] = data['sm_id'].replace('None', np.nan)
    data.dropna(subset=['sm_id'], inplace=True)
    print(data.shape[0])
    # print(data.)
    SEQ_LEN_short = 3
    SEQ_LEN_prefer = 20
    embedding_dim = 32

    for header in num_header:
        temp = data[header].fillna(-1)
        data[header] = temp
    for header in string_header:
        temp = data[header].fillna("unknow")
        data[header] = temp
    for header in time_header:
        temp = data[header].fillna(data[header].min())
        data[header] = temp

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`
    encoder = []
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        encoder.append(lbe)

    user_profile = data[user_fearther].drop_duplicates('user_id')  # 用户特征，drop_duplicates 去重
    user_profile.set_index("user_id", inplace=True)  # 将user_id列转为索引，inplace表示在原数据上修改

    #item_profile = data[item_fearther].drop_duplicates('sm_id')  # 物品特征，去重
    item_profile = data[["sm_id"]].drop_duplicates('sm_id')
    #item_profile.set_index("sm_id", inplace=True)
    # print(item_profile.head())
    # user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set, history_items = gen_data_set_sdm(data, seq_short_len=SEQ_LEN_short, seq_prefer_len=SEQ_LEN_prefer)

    train_model_input, train_label = gen_model_input_sdm(train_set, user_profile, SEQ_LEN_short, SEQ_LEN_prefer)
    test_model_input, test_item_input = gen_model_input_sdm(test_set, user_profile, SEQ_LEN_short, SEQ_LEN_prefer)


    # 2.count #unique features for each sparse field and generate feature config for sequence feature


    # for sdm,we must provide `VarLenSparseFeat` with name "prefer_xxx" and "short_xxx" and their length
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
                            SparseFeat("age", feature_max_idx['age'], embedding_dim),
                            SparseFeat("level", feature_max_idx['level'], embedding_dim),
                            SparseFeat("user_lang", feature_max_idx['user_lang'], embedding_dim),
                            SparseFeat("country", feature_max_idx['country'], embedding_dim),
                            SparseFeat("platform", feature_max_idx['platform'], embedding_dim),
                            SparseFeat("is_new", feature_max_idx['is_new'], embedding_dim),
                            VarLenSparseFeat(SparseFeat('short_sm_id', feature_max_idx['sm_id'], embedding_dim,
                                                        embedding_name="sm_id"), SEQ_LEN_short, 'mean',
                                             'short_sess_length'),
                            VarLenSparseFeat(SparseFeat('prefer_sm_id', feature_max_idx['sm_id'], embedding_dim,
                                                        embedding_name="sm_id"), SEQ_LEN_prefer, 'mean',
                                             'prefer_sess_length'),
                            VarLenSparseFeat(SparseFeat('short_genres', feature_max_idx['song_genres'], embedding_dim,
                                                        embedding_name="song_genres"), SEQ_LEN_short, 'mean',
                                             'short_sess_length'),
                            VarLenSparseFeat(SparseFeat('prefer_genres', feature_max_idx['song_genres'], embedding_dim,
                                                        embedding_name="song_genres"), SEQ_LEN_prefer, 'mean',
                                             'prefer_sess_length'),
                            ]

    item_feature_columns = [SparseFeat('sm_id', feature_max_idx['sm_id'], embedding_dim),]

    from collections import Counter

    train_counter = Counter(train_model_input['sm_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name='sm_id', item_count=item_count)

    K.set_learning_phase(True)

    import tensorflow as tf

    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()
    else:
        K.set_learning_phase(True)

    # units must be equal to item embedding dim!
    model = SDM(user_feature_columns, item_feature_columns, history_feature_list=['sm_id', 'song_genres'],
                units=embedding_dim, sampler_config=sampler_config)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=sampledsoftmaxloss)
    # init = tf.compat.v1.global_variables_initializer()
    history = model.fit(train_model_input, train_label,  # train_label,
                        batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.0, )

    K.set_learning_phase(False)
    # 3.Define Model,train,predict and evaluate
    test_user_model_input = test_model_input
    all_item_model_input = {"sm_id": item_profile['sm_id'].values, }

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    # user_embs = user_embs[:, i, :]  # i in [0,k_max) if MIND
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    print(user_embs.shape)
    print(user_embs)
    print(item_embs.shape)

    test_true_label = {line[0]: [line[1]] for line in test_set}

    import numpy as np
    import faiss
    from tqdm import tqdm
    from utils import recall_N

    index = faiss.IndexFlatIP(embedding_dim)
    faiss.normalize_L2(item_embs)
    index.add(item_embs)
    faiss.normalize_L2(user_embs)
    D, I = index.search(np.ascontiguousarray(user_embs), 50)
    recall_50 = []
    recall_10 = []
    hit = 0

    f = open(FLAGS.save_sdm_dir, 'w')
    for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
        try:
            pred = [item_profile['sm_id'].values[x] for x in I[i]]
            # print(1)
            filter_item = None
            rec_10 = recall_N(test_true_label[uid], pred, N=10)
            rec_50 = recall_N(test_true_label[uid], pred, N=FLAGS.recall_topk)
            # print(2)

            recall_10.append(rec_10)
            recall_50.append(rec_50)
            if test_true_label[uid] in pred:
                hit += 1

            his_item = set(history_items[uid])
            pred = [item_profile['sm_id'].values[x] - 1 for x in I[i] if item_profile['sm_id'].values[x] not in his_item]

            user_orgin = encoder[0].inverse_transform([uid - 1])  # 特征处理之前的uid
            item_orgin = [str(x) for x in encoder[1].inverse_transform(pred)]  # 特征处理之前的item_id
            f.write(str(user_orgin[0]) + "\t" + ','.join(item_orgin) + '\n')

        except Exception as e:
            print(i, e)
    f.close()
    print("recall_10", np.mean(recall_10))
    print("recall_50", np.mean(recall_50))
    print("hr", hit / len(test_user_model_input['user_id']))

