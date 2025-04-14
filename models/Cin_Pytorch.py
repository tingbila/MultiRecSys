# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media


import torch
import torch.nn.functional as F


# 定义 XdeepFM中的CIN组件
class CIN(torch.nn.Module):
    def __init__(self, input_dim, layer_dims):
        super().__init__()
        self.layer_dims = layer_dims   # layer_dims 是一个列表，表示每层输出的特征数。例如，[7, 9] 表示每层输出 7,9 个交叉特征。
        self.field_nums = [input_dim]  # 初始特征数量（输入特征的数量）
        self.conv_layers = torch.nn.ModuleList()  # 存储每一层的卷积层

        # 构建每层的卷积层
        for i, dim in enumerate(layer_dims):
            # in_channels = F0 * Fn，表示卷积的输入通道数，这里是交叉特征的数量
            in_channels = self.field_nums[0] * self.field_nums[-1]
            self.conv_layers.append(
                torch.nn.Conv1d(in_channels=in_channels, out_channels=dim, kernel_size=1, bias=True)
            )
            self.field_nums.append(dim)  # 添加每层输出的特征数到 field_nums 中

        # print(self.conv_layers)

        # 最终的线性输出层，输入维度为所有卷积层输出特征的总和
        self.fc = torch.nn.Linear(sum(layer_dims), 1)

    def forward(self, x):
        # x 的形状为 [batch_size, field_num, embedding_dim]
        batch_size, field_num, embed_dim = x.shape

        # 扩展 x 维度，准备进行外积计算
        x0 = x.unsqueeze(2)  # [B, F, 1, D]，扩展维度，准备做外积
        h = x                # [B, F, D]，原始特征作为交叉操作的第二个输入
        xs = []              # 用于存储每一层卷积后的结果

        # 对每一层的卷积进行操作
        for i, conv in enumerate(self.conv_layers):
            h1 = h.unsqueeze(1)         #  [B, 1, Fn, D]，扩展维度，以便计算外积
            x_ = x0 * h1                #  第1步：X0 和 Xk 进行外积，考虑所有情况的特征交叉乘积  # 外积操作： [B, F0, Fn, D]
            x_ = x_.view(batch_size, self.field_nums[0] * self.field_nums[i], embed_dim)    # 第2步： [B, F0*Fn, D] reshape 数据格式变换，符合卷积输入格式要求
            # print(x_.shape)
            x_ = conv(x_)               #  第3步： 卷积操作： [B, D_out, D]，提取交叉特征中的模式 卷积操作对外积结果进行特征提取
            x_ = F.relu(x_)             # 使用 ReLU 激活函数，增强特征非线性表达能力
            h = x_                      # 将卷积后的输出作为下一层的输入
            xs.append(x_)               # 保存每一层的输出

        # for item in xs:    因为测试用例是layer_dims = [7, 9]，所以结果是下面的
        #     print(item.shape)
        #     torch.Size([2, 7, 5])
        #     torch.Size([2, 9, 5])

        # 第4步：拼接所有卷积层输出，在 field 维度上进行拼接，然后通过 sum pooling 聚合特征  [B, sum(layer_dims)] => [B,16] 将每一层的特征堆叠并求和
        result = torch.sum(torch.cat(xs, dim=1), dim=2)  # [2, 16, 5]  ==> [2,16]

        # 输出层，最终将特征映射到一个标量（分类或回归任务）
        return self.fc(result)   # 输出的形状为 [B, 1]，对应最终的预测结果


if __name__ == '__main__':
    import torch
    import numpy as np

    # 假设 input_dim=3 (输入特征个数) 和 embedding_dim=5 (每个特征的嵌入维度)
    input_dim = 3
    embedding_dim = 5
    layer_dims = [7, 9]    # 每一层的输出特征个数（相当于原来的 num_layers=2）

    # 生成模拟数据，假设 batch_size=2
    x = torch.randn(2, input_dim, embedding_dim)  # [batch_size, field_num, embedding_dim]

    print("Input shape:", x.shape)
    print("Input x:", x)

    # 导入并实例化你优化后的 CIN 模块
    cin = CIN(input_dim=input_dim, layer_dims=layer_dims)

    # 前向传播
    output = cin(x)

    # 打印输出的形状
    print("Output shape:", output.shape)

    # 查看输出值
    print("Output values:", output)


# Input shape: torch.Size([2, 3, 5])
# Input x: tensor([[[-1.0736, -0.0119, -0.6501,  0.6828,  0.1811],
#          [-1.5963, -1.2592, -0.6891,  1.5207,  0.1446],
#          [ 1.1991, -0.4725, -0.3187,  2.1846, -0.3761]],
#
#         [[ 1.6299,  0.9647,  0.5362, -1.2203, -0.7350],
#          [-1.0462, -1.8208, -0.5738,  0.0447, -1.2010],
#          [-0.6067,  0.8325,  1.4435,  0.8080,  0.1424]]])
# ModuleList(
#   (0): Conv1d(9, 7, kernel_size=(1,), stride=(1,))
#   (1): Conv1d(21, 9, kernel_size=(1,), stride=(1,))
# )
# torch.Size([2, 9, 5])
# torch.Size([2, 21, 5])
# Output shape: torch.Size([2, 1])
# Output values: tensor([[0.7096],
#         [0.6673]], grad_fn=<AddmmBackward0>)