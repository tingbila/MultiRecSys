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
保存文件，然后再次运行你的程序进行验证，看是否报错：
`````
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
`````

### 2、AttributeError: module 'tensorflow.python.distribute.input_lib' has no attribute 'DistributedDatasetInterface'

报错原因:
`````
这个错误是由于TensorFlow版本兼容性问题引起的。DistributedDatasetInterface在较新的TensorFlow版本中被移除或更名。
`````
解决方案1:
`````
# 在导入keras前添加以下修复代码
import tensorflow.python.keras.engine.data_adapter as data_adapter

# 重写分布式数据集检查函数
def _is_distributed_dataset_fixed(ds):
    return False  # 直接返回False跳过检查

# 应用猴子补丁
data_adapter._is_distributed_dataset = _is_distributed_dataset_fixed

# 然后继续你的原有代码
import pandas as pd
from deepctr.models import DeepFM
`````

解决方案2:
`````
import tensorflow as tf

# 修改原有的调用方式
train_dataset = tf.data.Dataset.from_tensor_slices((
    {name: train_data[name] for name in feature_names},
    train_data[target].values
)).batch(256).prefetch(1)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {name: val_data[name] for name in feature_names},
    val_data[target].values
)).batch(256)

history = model.fit(
    train_dataset,  # 直接传入Dataset对象
    epochs=10, 
    verbose=2,
    validation_data=val_dataset
)
`````