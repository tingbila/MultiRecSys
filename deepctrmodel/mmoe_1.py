# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media

# ========== 标准库 ==========
import warnings

# ========== 第三方库 ==========
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # 或者 'QtAgg'，看你电脑支持哪个
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score

from tensorflow import keras
import tensorflow.python.keras.engine.data_adapter as data_adapter

# ========== DeepCTR 模块 ==========
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM, MMOE

# ========== 设置选项 & 警告过滤 ==========
# 设置 pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# 忽略 matplotlib 的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.image")

# 修复 DeepCTR 与新版 TensorFlow 的兼容性问题
def _is_distributed_dataset_fixed(ds):
    return False
data_adapter._is_distributed_dataset = _is_distributed_dataset_fixed


# This example shows how to use MMOE to solve a multi task learning problem.
# You can get the demo data census-income.sample and run the following codes.
# https://deepctr-doc.readthedocs.io/en/latest/Examples.html

if __name__ == "__main__":
    # 1. 定义数据字段名称
    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # 2. 读取原始数据并命名列
    data = pd.read_csv(r'D:\software\pycharm_repository\StarMaker\MultiRecSys\data_files\census-income.sample',header=None, names=column_names)

    # 3. 构造多任务标签：收入是否超过 50k、是否未婚
    data['label_income'] = data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
    data['label_marital'] = data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
    data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)

    # 4. 区分稀疏特征和连续特征
    columns = data.columns.values.tolist()
    sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                       'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                       'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                       'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                       'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                       'vet_question']
    dense_features = [col for col in columns if col not in sparse_features and col not in ['label_income', 'label_marital']]

    # 5. 缺失值填充：稀疏特征用 "-1" 填充，连续特征用 0 填充
    data[sparse_features] = data[sparse_features].fillna('-1')
    data[dense_features] = data[dense_features].fillna(0)

    # 6. 连续特征归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 7. 稀疏特征编码（LabelEncoder）
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 8. 构建 DeepCTR 所需的特征列
    fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4) for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    # 9. 特征列用于 DNN 和 Linear 部分
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    # 10. 获取模型输入所需的特征名列表
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 11. 划分训练集和测试集
    train, test = train_test_split(data, test_size=0.2, random_state=2020)

    # 12. 构造模型输入字典
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 13. 定义 MMOE 模型（多任务输出：收入分类 + 婚姻状态分类）
    model = MMOE(dnn_feature_columns, tower_dnn_hidden_units=[],task_types=['binary', 'binary'],task_names=['label_income', 'label_marital'])

    # 14. 编译模型
    model.compile("adam",loss=["binary_crossentropy", "binary_crossentropy"],metrics=['binary_crossentropy'])

    # 15. 模型训练
    history = model.fit(train_model_input,[train['label_income'].values, train['label_marital'].values],batch_size=256, epochs=10, verbose=2, validation_split=0.2)

    # 16. 模型预测
    pred_ans = model.predict(test_model_input, batch_size=256)

    # 17. AUC 评估
    print("test income AUC",  round(roc_auc_score(test['label_income'], pred_ans[0]), 4))
    print("test marital AUC", round(roc_auc_score(test['label_marital'], pred_ans[1]), 4))

    # test income AUC 0.2696
    # test marital AUC 0.8363

    # 18. 可视化数据:可以自己选择数据进行可视化matplotlib绘画 ---
    print(pd.DataFrame(history.history))
    print(history.epoch)
    plt.figure(figsize=(12, 5))


#        loss  label_income_loss  label_marital_loss  label_income_binary_crossentropy  label_marital_binary_crossentropy  val_loss  val_label_income_loss  val_label_marital_loss  val_label_income_binary_crossentropy  val_label_marital_binary_crossentropy
# 0  1.311852           0.633247            0.678605                          0.633247                           0.678605  1.253394               0.587825                0.665569                              0.587825                               0.665569
# 1  1.262170           0.591361            0.670809                          0.591361                           0.670809  1.203463               0.546695                0.656768                              0.546695                               0.656768
# 2  1.215557           0.552399            0.663159                          0.552399                           0.663159  1.156016               0.508086                0.647931                              0.508086                               0.647931
# 3  1.170998           0.515558            0.655440                          0.515558                           0.655440  1.110823               0.471819                0.639003                              0.471819                               0.639003
# 4  1.128389           0.480636            0.647753                          0.480636                           0.647753  1.068299               0.437658                0.630640                              0.437658                               0.630640
# 5  1.088079           0.447723            0.640356                          0.447723                           0.640356  1.027738               0.405292                0.622446                              0.405292                               0.622446
# 6  1.049889           0.416405            0.633484                          0.416405                           0.633484  0.989416               0.374809                0.614607                              0.374809                               0.614607
# 7  1.013854           0.386796            0.627057                          0.386796                           0.627057  0.953882               0.346590                0.607292                              0.346590                               0.607292
# 8  0.980380           0.359145            0.621235                          0.359145                           0.621235  0.920665               0.320408                0.600256                              0.320408                               0.600256
# 9  0.949334           0.333360            0.615973                          0.333360                           0.615973  0.890479               0.296610                0.593868                              0.296610                               0.593868
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]