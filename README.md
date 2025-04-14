## 1. 环境依赖说明
| 组件           | 版本要求       | 说明                        |
|----------------|----------------|-----------------------------|
| Python         | 3.9.X            | Python 3.10 版本会报错       |
| numpy          | 1.23.5         | —                           |
| tensorflow     | 2.15.0         | 稳定版本，推荐使用          |
| Keras          | 随 TensorFlow 安装 | 无需单独安装               |
| protobuf       | 3.20.3         | 避免与 TensorFlow 冲突     |
| pandas         | 1.5.3          | 用于数据预处理              |
| deepctr        | 0.9.3          | 推荐系统常用推荐库之一      |
| 其余包          |xxx          | 正常安装即可       |


## 2. 安装命令汇总
### 1、卸载所有包 - 遇到报错时请多次卸载
 `````
pip uninstall tensorflow numpy protobuf pandas deepctr
`````
### 2、清空缓存 - 执行几遍
 `````
pip cache purge
`````

### 3、安装核心包
 `````
pip install tensorflow==2.15.0 numpy==1.23.5 protobuf==3.20.3 pandas==1.5.3 deepctr==0.9.3 --user
`````

### 4、安装完成后查看一下版本
 `````
import tensorflow as tf
import numpy as np
import pandas as pd
import deepctr

print("tensorflow:", tf.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("deepctr:", deepctr.__version__)
`````

### 5、其他包安装
`````
pip install scikit-learn matplotlib jupyter -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
`````

## 3. 常见报错
### 1、执行from deepctr.models import DeepFM显示报错信息:
`````
ImportError: cannot import name 'LSTM' from 'tensorflow.python.keras.layers' (C:\Users\张明阳\AppData\Roaming\Python\Python39\site-packages\tensorflow\python\keras\layers\__init__.py)
`````
报错原因:
`````
deepctr 使用了 TensorFlow 的 私有路径（tensorflow.python.keras.layers.LSTM），而这在 TensorFlow 2.15 中不再兼容，或者已经被限制访问。
这类问题是 deepctr 和 TensorFlow 版本 不兼容 导致的。
`````
解决方案:
`````
1. 找到 deepctr/layers/sequence.py 文件（根据你的错误路径是这个）：
C:\Users\张明阳\AppData\Roaming\Python\Python39\site-packages\deepctr\layers\sequence.py

2. 修改以下代码： 原始代码（会出错）：
from tensorflow.python.keras.layers import LSTM, Lambda, Layer, Dropout

3. 修改为（使用公开 API）：
from tensorflow.keras.layers import LSTM, Lambda, Layer, Dropout
`````
保存文件，然后再次运行你的程序。