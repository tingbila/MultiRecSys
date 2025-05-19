# feature_columns.py

from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
from config.data_config import *

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


class SparseFeat:
    def __init__(self, name, vocab_size, embedding_dim):
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

class DenseFeat:
    def __init__(self, name):
        self.name = name

class VarLenSparseFeat:
    def __init__(self, name, vocab_size, embedding_dim, maxlen):
        self.name = name
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen


# layers/embedding.py
def build_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns):
    embed_dict = {}
    for feat in sparse_feature_columns:
        embed_dict[feat.name] = tf.keras.layers.Embedding(feat.vocab_size, feat.embedding_dim)
    for feat in varlen_sparse_feature_columns:
        embed_dict[feat.name] = tf.keras.layers.Embedding(feat.vocab_size, feat.embedding_dim, mask_zero=True)
    return embed_dict


# inputs/input_fn.py
def build_input(features, sparse_feature_names, dense_feature_names, sequence_feature_names=None):
    input_dict = {
        'sparse': {name: features[name] for name in sparse_feature_names},
        'dense': tf.concat([features[name] for name in dense_feature_names], axis=-1)
    }
    if sequence_feature_names:
        input_dict['sequence'] = {
            name: features[name] for name in sequence_feature_names
        }
    return input_dict




class DeepFM(Model):
    def __init__(self, linear_feature_columns, dnn_feature_columns, varlen_sparse_feature_columns=[], dnn_hidden_units=[128, 128]):
        super().__init__()
        self.sparse_features = [f for f in dnn_feature_columns if isinstance(f, SparseFeat)]
        self.dense_features = [f for f in dnn_feature_columns if isinstance(f, DenseFeat)]
        self.seq_features = varlen_sparse_feature_columns

        self.embedding_dict = build_embedding_dict(self.sparse_features, self.seq_features)

        self.linear_dense = layers.Dense(1)
        self.linear_sparse_embed = {
            feat.name: tf.keras.layers.Embedding(feat.vocab_size, 1)
            for feat in self.sparse_features
        }

        self.dnn_input_dim = sum([feat.embedding_dim for feat in self.sparse_features + self.seq_features]) + len(self.dense_features)
        self.dnn = tf.keras.Sequential()
        for units in dnn_hidden_units:
            self.dnn.add(layers.Dense(units, activation='relu'))
            self.dnn.add(layers.Dropout(0.5))
        self.dnn.add(layers.Dense(1, activation=None))

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs, training=False):
        dense_inputs = inputs['dense']
        sparse_inputs = inputs['sparse']
        seq_inputs = inputs.get('sequence', {})

        linear_dense_out = self.linear_dense(dense_inputs)
        linear_sparse_out = tf.add_n([
            tf.squeeze(self.linear_sparse_embed[feat](sparse_inputs[feat]), axis=-1)
            for feat in sparse_inputs
        ])
        linear_output = linear_dense_out + tf.expand_dims(linear_sparse_out, axis=-1)

        sparse_embed_list = [
            self.embedding_dict[feat](sparse_inputs[feat]) for feat in sparse_inputs
        ]
        seq_embed_list = [
            tf.reduce_mean(self.embedding_dict[feat](seq_inputs[feat]), axis=1) for feat in seq_inputs
        ]
        fm_input = tf.concat(sparse_embed_list + seq_embed_list, axis=1)
        square_sum = tf.square(tf.reduce_sum(fm_input, axis=1))
        sum_square = tf.reduce_sum(tf.square(fm_input), axis=1)
        fm_output = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=1, keepdims=True)

        dnn_input = tf.concat([dense_inputs] + sparse_embed_list + seq_embed_list, axis=1)
        dnn_output = self.dnn(dnn_input)

        logit = linear_output + fm_output + dnn_output
        output = self.output_layer(logit)
        return output




# 1. 定义特征列
sparse_features = [SparseFeat('user_id', 1000, 8), SparseFeat('item_id', 2000, 8)]
dense_features = [DenseFeat('time'), DenseFeat('duration')]
sequence_features = [VarLenSparseFeat('history_item', 2000, 8, maxlen=20)]

# 2. 模型实例化
model = DeepFM(
    linear_feature_columns=sparse_features + dense_features,
    dnn_feature_columns=sparse_features + dense_features,
    varlen_sparse_feature_columns=sequence_features
)

# 3. 构造输入
input_dict = build_input(batch_data, ['user_id', 'item_id'], ['time', 'duration'], ['history_item'])

# 4. 前向调用
pred = model(input_dict)

