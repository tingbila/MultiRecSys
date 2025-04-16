## 1. 安装命令汇总-python必须3.8环境
 `````
# 1. 清理环境
pip uninstall tensorflow numpy protobuf pandas deepctr -y
pip cache purge

# 2. 安装TF核心依赖 tensorflow必须是2.6.2 参考:https://github.com/shenweichen/DeepCTR/blob/master/docs/requirements.readthedocs.txt
pip install tensorflow==2.6.2 numpy==1.19.5 protobuf==3.17.3 h5py==3.1.0

# 3. 安装DeepCTR
pip install deepctr==0.9.3 --no-build-isolation --no-deps

# 4. 补充依赖
pip install pandas==1.3.5 
pip install matplotlib==3.3.4  
pip install scikit-learn jupyter tqdm torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
`````


## 2、安装完之后进行结果校验
 `````
C:\Windows\System32>pip list | findstr "tensorflow numpy protobuf h5py pandas deepctr"
deepctr                   0.9.3
h5py                      3.1.0
numpy                     1.19.5
pandas                    1.3.5
protobuf                  3.17.3
tensorflow                2.6.2
tensorflow-estimator      2.6.0
`````
