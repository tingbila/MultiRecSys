# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media

from config.data_config import *

# trainers/trainer.py
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf

# 设置 Pandas 显示选项，防止省略
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个

def train_and_evaluate(model, train_dataset, valid_dataset, test_dataset):
    # 评估指标从 Accuracy 改为 AUC（同时保留 Accuracy）
    model.compile(
        optimizer='adam',
        loss={'finish': 'binary_crossentropy', 'like': 'binary_crossentropy'},
        metrics={
            'finish': [keras.metrics.AUC(name='auc'), 'accuracy'],
            'like': [keras.metrics.AUC(name='auc'), 'accuracy']
        }
    )

    # 输出目录结构
    base_dir = './outputs'
    log_dir = os.path.join(base_dir, 'logs')
    callbacks_dir = os.path.join(base_dir, 'callbacks')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(callbacks_dir, exist_ok=True)
    output_model_file = os.path.join(callbacks_dir, 'best_model.keras')

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=output_model_file,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=50,
        callbacks=callbacks
    )

    print("\nTest Evaluation:")
    test_loss = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss & Metrics: {test_loss}")

    # 你完全可以调用 model.predict(test_dataset)，即使 test_dataset 中包含了标签，Keras 会自动忽略标签部分，只用特征做前向预测。
    # y_pred = model.predict(test_dataset)
    # print(y_pred)
    # 模型有两个输出任务（finish 和 like），每个都是一个 [0, 1] 之间的概率值，表示每条样本被 finish / like 的概率
    # {'finish': array([[0.45237187],
    #                   [0.44068933],
    #                   [0.48365387],
    #                   [0.45916972],
    #                   [0.47017196],
    #                   [0.47122645],
    #                   [0.4356586],
    #                   [0.42037615]], dtype=float32), 'like': array([[0.51725936],
    #                                                                 [0.5225682],
    #                                                                 [0.5031291],
    #                                                                 [0.5141797],
    #                                                                 [0.5092073],
    #                                                                 [0.50873137],
    #                                                                 [0.52486163],
    #                                                                 [0.531861]], dtype=float32)}
    # 评估指标
    # 预测测试数据中每一条数据的点击（finish + like）概率。本次比赛使用AUC（ROC曲线下面积）作为评估指标。AUC的定义和计算方法可参考维基百科。AUC越高，代表结果越优，排名越靠前。
    # 在总分中，finish和like的分配比例是：0.7finish + 0.3like

    
    print(pd.DataFrame(history.history))
    print(history.epoch)

    # 可视化
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['loss'], label='Train Loss')
    plt.plot(history.epoch, history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['finish_auc'], label='Train AUC - Finish')
    plt.plot(history.epoch, history.history['val_finish_auc'], label='Val AUC - Finish')
    plt.plot(history.epoch, history.history['like_auc'], label='Train AUC - Like')
    plt.plot(history.epoch, history.history['val_like_auc'], label='Val AUC - Like')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Test Loss & Metrics: [1.2047441005706787, 0.6992161273956299, 0.5055279731750488, 0.375, 0.7999999523162842, 1.0, 0.0]
#    finish_accuracy  finish_auc  finish_loss  like_accuracy  like_auc  like_loss      loss  val_finish_accuracy  val_finish_auc  val_finish_loss  val_like_accuracy  val_like_auc  val_like_loss  val_loss
# 0             0.64       0.800     0.550040           0.48  0.458333   0.605947  1.183139             0.428571        0.166667         0.956011                1.0           0.0       0.398620  1.365861
# 1             0.80       0.690     0.432517           0.96  0.291667   0.387115  0.745555             0.428571        0.250000         1.544587                1.0           0.0       0.133579  1.844764
# 2             0.80       0.805     0.414861           0.96  0.416667   0.208397  0.647375             0.428571        0.250000         1.681163                1.0           0.0       0.133551  1.993070
# 3             0.80       0.830     0.390524           0.96  0.479167   0.208582  0.604369             0.428571        0.250000         1.659434                1.0           0.0       0.156601  1.976349
# 4             0.80       1.000     0.330603           0.96  0.791667   0.191864  0.524144             0.428571        0.291667         1.880151                1.0           0.0       0.127263  2.209401
# 5             0.80       0.950     0.358698           0.96  0.916667   0.203496  0.511783             0.428571        0.291667         2.171582                1.0           0.0       0.082203  2.520214
# [0, 1, 2, 3, 4, 5]