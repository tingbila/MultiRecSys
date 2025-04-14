import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 定义 CrossNetwork的核心组件
class CrossNetwork(layers.Layer):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.ws = []
        self.bs = []

    def build(self, input_shape):
        # build()方法的作用：
        # 它在第一次接收到输入张量（call()方法被调用）时自动执行一次，用于：推断输入形状（input_shape）初始化依赖输入维度的参数（如权重）
        input_dim = input_shape[-1]
        for i in range(self.num_layers):
            self.ws.append(self.add_weight(
                shape=(input_dim, 1),
                initializer='random_normal',
                trainable=True,
                name=f'cross_weight_{i}'
            ))
            self.bs.append(self.add_weight(
                shape=(input_dim,),
                initializer='zeros',
                trainable=True,
                name=f'cross_bias_{i}'
            ))

    def call(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = tf.matmul(x, self.ws[i])  # (batch_size, 1)
            x = x0 * xw + self.bs[i] + x   # (batch_size, input_dim)
        return x


# ===================== 2. DCN 模型定义 =====================
class DCNModel(tf.keras.Model):
    def __init__(self, input_dim, num_cross_layers=2, hidden_units=[128, 64]):
        super().__init__()
        self.cross_network = CrossNetwork(num_layers=num_cross_layers)

        self.deep_network = tf.keras.Sequential([
            layers.Dense(hidden_units[0], activation='relu'),
            layers.Dense(hidden_units[1], activation='relu'),
            layers.Dense(1)
        ])

        self.cross_proj = layers.Dense(1)
        self.final_activation = layers.Activation('sigmoid')

    def call(self, inputs):
        cross_out = self.cross_network(inputs)         # (batch, input_dim)
        cross_out = self.cross_proj(cross_out)         # (batch, 1)

        deep_out = self.deep_network(inputs)           # (batch, 1)

        output = cross_out + deep_out
        return self.final_activation(output)


if __name__ == '__main__':
    # 生成试算输入数据
    input_dim   = 3  # 输入特征的维度，输入特征的数量
    num_samples = 10  # 样本数量

    # 随机生成输入数据
    inputs = np.random.randn(num_samples, input_dim).astype(np.float32)
    inputs_tensor = tf.convert_to_tensor(inputs)  # 转换为 TensorFlow 张量
    print(inputs_tensor)  # shape=(10, 3)

    # ===================== 3. 创建模型并进行预测 =====================
    # 创建模型
    model = DCNModel(input_dim=input_dim, num_cross_layers=2, hidden_units=[128, 64])

    # 通过模型执行一次前向传播，获取预测结果
    predictions = model(inputs_tensor)

    # 打印预测结果
    print(f"预测结果形状: {predictions.shape}")
    print(f"前5个预测结果: {predictions.numpy()[:5]}")
