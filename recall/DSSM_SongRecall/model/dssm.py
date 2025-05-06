"""
Author:
    Zhe Wang,734914022@qq.com
Reference:
Huang P S , He X , Gao J , et al. Learning deep structured semantic models for web search using clickthrough data[C]// Acm International Conference on Conference on Information & Knowledge Management. ACM, 2013.
"""

from tensorflow.python.keras.models import Model
import tensorflow.python.keras.backend  as K
from deepctr.feature_column import build_input_features, create_embedding_matrix, concat_func
import sys
sys.path.append("..")
from inputs import input_from_feature_columns
from layers.core import Similarity, PredictionLayer, DNN, PoolingLayer
from layers.utils import combined_dnn_input, reduce_mean, combined_dnn_input_new
from tensorflow.python.keras.layers import Layer, Flatten
import tensorflow as tf
import itertools


def DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32),
         item_dnn_hidden_units=(64, 32),
         dnn_activation='tanh', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, metric='cos'):
    """Instantiates the Deep Structured Semantic Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param item_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of item tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param metric: str, ``"cos"`` for  cosine  or  ``"ip"`` for inner product
    :return: A Keras model instance.

    """

    # key: SparseFeat 转为 key: Embedding
    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed,
                                                    seq_mask_zero=True)
    print(embedding_matrix_dict)

    # get user dnn_input
    # user_features: OrderedDict([('user_id', <KerasTensor: shape=(None, 1) dtype=int32 (created by layer 'user_id')>), ...]
    print(embedding_matrix_dict)
    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    # 获取稀疏特征向量列表和稠密特征向量列表
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    # 向量拼接得到dnn_input， KerasTensor
    user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)

    # get item dnn_input
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
    print(user_dnn_input)
    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(user_dnn_input)
    print(user_dnn_out)

    item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(item_dnn_input)

    score = Similarity(type=metric, gamma=10)([user_dnn_out, item_dnn_out])

    output = PredictionLayer("binary", False)(score)

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    # 按名称访问变量内容
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model


def SENet_DSSM(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32),
         item_dnn_hidden_units=(64, 32),
         dnn_activation='tanh', dnn_use_bn=False,
         l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024, metric='cos'):
    """Instantiates the Deep Structured Semantic Model architecture.

    :param user_feature_columns: An iterable containing user's features used by  the model.
    :param item_feature_columns: An iterable containing item's features used by  the model.
    :param user_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of user tower
    :param item_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of item tower
    :param dnn_activation: Activation function to use in deep net
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param seed: integer ,to use as random seed.
    :param metric: str, ``"cos"`` for  cosine  or  ``"ip"`` for inner product
    :return: A Keras model instance.

    """

    # key: SparseFeat 转为 key: Embedding
    embedding_matrix_dict = create_embedding_matrix(user_feature_columns + item_feature_columns, l2_reg_embedding,
                                                    seed=seed,
                                                    seq_mask_zero=True)
    # get user dnn_input
    # user_features: OrderedDict([('user_id', <KerasTensor: shape=(None, 1) dtype=int32 (created by layer 'user_id')>), ...]

    user_features = build_input_features(user_feature_columns)
    user_inputs_list = list(user_features.values())
    # 获取稀疏特征向量列表和稠密特征向量列表
    user_sparse_embedding_list, user_dense_value_list = input_from_feature_columns(user_features,
                                                                                   user_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    # 向量拼接得到dnn_input， KerasTensor
    # print("senet start")
    user_senet_embedding_list = SENETLayer()(user_sparse_embedding_list)
    # print("user senet", user_senet_embedding_list)
    user_dnn_input = combined_dnn_input_new(user_senet_embedding_list, user_dense_value_list)
    # print(user_dnn_input)
    # print("senet end")

    # get item dnn_input
    item_features = build_input_features(item_feature_columns)
    item_inputs_list = list(item_features.values())
    item_sparse_embedding_list, item_dense_value_list = input_from_feature_columns(item_features,
                                                                                   item_feature_columns,
                                                                                   l2_reg_embedding, seed=seed,
                                                                                   embedding_matrix_dict=embedding_matrix_dict)
    print('item', item_sparse_embedding_list)
    item_senet_embedding_list = SENETLayer()(item_sparse_embedding_list)
    print('item_senet', item_senet_embedding_list)
    item_dnn_input = combined_dnn_input_new(item_senet_embedding_list, item_dense_value_list)

    user_dnn_out = DNN(user_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(user_dnn_input)
    # print(user_dnn_out)

    item_dnn_out = DNN(item_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                       dnn_use_bn, seed=seed)(item_dnn_input)

    score = Similarity(type=metric, gamma=10)([user_dnn_out, item_dnn_out])

    output = PredictionLayer("binary", False)(score)

    model = Model(inputs=user_inputs_list + item_inputs_list, outputs=output)

    # 按名称访问变量内容
    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("item_input", item_inputs_list)
    model.__setattr__("user_embedding", user_dnn_out)
    model.__setattr__("item_embedding", item_dnn_out)

    return model


class SENETLayer(Layer):
    """SENETLayer used in FiBiNET.

      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Arguments
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.

        - **seed** : A Python integer to use as random seed.

      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)
    """

    def __init__(self, reduction_ratio=3, seed=1024, **kwargs):
        self.reduction_ratio = reduction_ratio

        self.seed = seed
        super(SENETLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')

        self.filed_size = len(input_shape)
        self.embedding_size = input_shape[0][-1]
        reduction_size = max(1, self.filed_size // self.reduction_ratio)

        self.W_1 = self.add_weight(shape=(
            self.filed_size, reduction_size), initializer=tf.compat.v1.keras.initializers.glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.filed_size), initializer=tf.compat.v1.keras.initializers.glorot_normal(seed=self.seed), name="W_2")

        self.tensordot = tf.keras.layers.Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(SENETLayer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        # inputs = concat_func(inputs, axis=1)
        inputs = tf.concat(inputs, 1)
        Z = reduce_mean(inputs, axis=-1, )
        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))
        # print(self.embedding_size)
        # print(tf.reshape(V, [-1, self.filed_size * self.embedding_size]))

        return tf.split(V, self.filed_size, axis=1)
        # return tf.reshape(V, [-1, self.filed_size * self.embedding_size])

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return [None] * self.filed_size

    def get_config(self, ):
        config = {'reduction_ratio': self.reduction_ratio, 'seed': self.seed}
        base_config = super(SENETLayer, self).get_config()
        base_config.update(config)
        return base_config


class BilinearInteraction(Layer):
    """BilinearInteraction Layer used in FiBiNET.

      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``. Its length is ``filed_size``.

      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size*(filed_size-1)/2,embedding_size)``.

      Arguments
        - **bilinear_type** : String, types of bilinear functions used in this layer.

        - **seed** : A Python integer to use as random seed.

      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)

    """

    def __init__(self, bilinear_type="interaction", seed=1024, **kwargs):
        self.bilinear_type = bilinear_type
        self.seed = seed

        super(BilinearInteraction, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')
        embedding_size = int(input_shape[0][-1])

        if self.bilinear_type == "all":
            self.W = self.add_weight(shape=(embedding_size, embedding_size), initializer=tf.compat.v1.keras.initializers.glorot_normal(
                seed=self.seed), name="bilinear_weight")
        elif self.bilinear_type == "each":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=tf.compat.v1.keras.initializers.glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i)) for i in range(len(input_shape) - 1)]
        elif self.bilinear_type == "interaction":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=tf.compat.v1.keras.initializers.glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i) + '_' + str(j)) for i, j in
                           itertools.combinations(range(len(input_shape)), 2)]
        else:
            raise NotImplementedError

        super(BilinearInteraction, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        n = len(inputs)
        if self.bilinear_type == "all":
            vidots = [tf.tensordot(inputs[i], self.W, axes=(-1, 0)) for i in range(n)]
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "each":
            vidots = [tf.tensordot(inputs[i], self.W_list[i], axes=(-1, 0)) for i in range(n - 1)]
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "interaction":
            p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                 for v, w in zip(itertools.combinations(inputs, 2), self.W_list)]
        else:
            raise NotImplementedError
        output = concat_func(p, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        filed_size = len(input_shape)
        embedding_size = input_shape[0][-1]
        return (None, filed_size * (filed_size - 1) // 2, embedding_size)

    def get_config(self, ):
        config = {'bilinear_type': self.bilinear_type, 'seed': self.seed}
        base_config = super(BilinearInteraction, self).get_config()
        base_config.update(config)
        return base_config


