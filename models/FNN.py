# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


"""
FNN（Factorization-machine supported Neural Network）是推荐系统中一个 早期将 FM 和神经网络结合 的模型，
常常被作为很多复杂模型的 baseline（基线），因为它结构清晰、性能稳定，并且是后续 DeepFM、NFM、AFM 等模型的思想基础之一。
FNN 的全称是 Factorization-machine supported Neural Network，主要思想是：
利用 FM 初始化特征的嵌入（embedding），然后将其作为输入送入一个前馈神经网络进行预测。

不是端到端模型，已经被淘汰！！！！需要先通过FM预训练好向量。。。
"""

