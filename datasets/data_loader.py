# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


# datasets/data_loader.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import os
from config.data_config import COLUMN_NAMES, CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COLS


def preprocess_dataframe(df, numeric_cols, categorical_cols, save_path=None):
    df_processed = df.copy()

    # 1️⃣ 连续特征-数值标准化处理
    scaler = StandardScaler()
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])

    # 2️⃣ 离散特征-类别特征先进行LabelEncoder再OneHot编码处理
    # LabelEncoder → OneHotEncoder 的组合，虽然LabelEncoder不是必须，但正如你说的，它能养成一个良好的编码习惯，为后续 Embedding 或 Index 对应提供一致性
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le


    encoder = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown="ignore")
    encoded_data = encoder.fit_transform(df_processed[categorical_cols])
    encoded_columns = encoder.get_feature_names_out(categorical_cols)

    df_encoded = pd.DataFrame(encoded_data, columns=encoded_columns, index=df.index)

    df_processed = df_processed.drop(columns=categorical_cols)
    df_processed = pd.concat([df_processed, df_encoded], axis=1)

    # 3️⃣ 保存
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(scaler, f"{save_path}/scaler.pkl")
        joblib.dump(encoder, f"{save_path}/encoder.pkl")
        joblib.dump(label_encoders, f"{save_path}/label_encoders.pkl")
        print(f"✅ 所有转换器已保存至 {save_path}")

    return df_processed



def load_dataset(file_path, batch_size=256):
    column_names = COLUMN_NAMES
    df = pd.read_csv(file_path, names=column_names)

    # 区分数值特征和稀疏特征
    categorical_cols  = CATEGORICAL_COLS
    numeric_cols      = NUMERIC_COLS

    df = preprocess_dataframe(df, numeric_cols, categorical_cols, save_path="./pkl_struct/")
    print(df)

    y = df[TARGET_COLS].values
    X = df.drop(columns=TARGET_COLS)
    # print(X.shape)  # (40, 234)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, (y_train[:, 0], y_train[:, 1])))
    train_dataset = train_dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, (y_valid[:, 0], y_valid[:, 1])))
    valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, (y_test[:, 0], y_test[:, 1])))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, valid_dataset, test_dataset, X.shape[1]