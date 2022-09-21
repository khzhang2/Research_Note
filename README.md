# Research Cheat Sheet

## Table of Contents
[Conceptual](#conceptual)
- [Terminologies](#terminologies)
- [Machine learning](#machine-learning)
- [Distributions](#distributions)

[Practical](#practical)
- System
  - [Environment](#environment)
  - [Colab](#colab)
  - [Homebrew](#homebrew)
- Python
  - [Parellel coding](#parallel-coding)
  - [Parser](#parser)
- Machine Learning
  - [PyTorch](#pytorch)
  - [Keras](#keras)
  - [Pytorch - Keras](#pytorch---keras)
  - [LSTM](#lstm)
- Data
  - [Numpy](#numpy)
  - [Pandas](#pandas)
  - [GeoPandas](#geopandas)
  - [GIS](#gis)

## Conceptual

### Terminologies
Inductive: Using particular facts and examples to form general rules and principles.

Deductive: Using knowledge about things that are generally true in order to think about and understand particular situations or problems.

### Machine learning
- Filter和kernel size的区别：
https://blog.csdn.net/qq_40243750/article/details/117363922
- padding 的操作就是在图像块的周围加上格子, 从而使得图像经过卷积过后大小不会变化,这种操作是使得图像的边缘数据也能被利用到,这样才能更好地扩张整张图像的边缘特征.
Outsize = 1+ (inputsize – kernelsize + 2*padding) / stride
https://blog.csdn.net/qxqsunshine/article/details/86435404
- 高斯混合模型（K个高斯模型的混合）
(高斯混合模型（GMM） - 戴文亮的文章 - 知乎
https://zhuanlan.zhihu.com/p/30483076 )
- 变分自动编码器（variational autoencoder, VAE）
  1. 我们拥有两部分输入：数据x，模型p(z, x)。
  2. 我们需要推断的是后验概率p(z | x)，但不能直接求。
  3. 构造后验概率p(z | x)的近似分布q(z; v)。
  4. 不断缩小q和p之间的距离直至收敛。
- (如何简单易懂地理解变分推断(variational inference)？ - 过小咩的回答 - 知乎
https://www.zhihu.com/question/41765860/answer/331070683 )
- Variational autoencoders. - JEREMY JORDAN, 2018 
https://www.jeremyjordan.me/variational-autoencoders/

### Distributions
<img src="./img/distributions.png" width="750">


## Practical

### Environment
- Create an environment

  > conda create -n [name] python=[3.7.9]

  > conda env list

  > pip install jupyter

  > python -m ipykernel install --name=[name]

  > python -m ipykernel install --name=[name] --user  # for mac

  > ipython kernelspec list/jupyter kernelspec list

- Remove an environment

  > conda remove -n [name] --all

  > jupyter kernelspec remove [name]

- Clone an environment
  
  > conda create --name [new_envname] --clone [old_envname]  # NOT recommended
  
  Can use:
  > pip freeze -l > requirements.txt<br>
  > pip install -r requirements.txt

- pip list
  - Get piplist
    > pip list > piplist.txt

  - Install packages from an existing piplist. Process piplist via [this notebook](https://github.com/HaTT2018/NET_louvain_DAN/blob/main/env/process_piplist.ipynb) -> obtain "requirements.txt" ->
    > pip install -r requirements.txt

- 安装、卸载cuda和cudnn：从卸载到安装：https://zhuanlan.zhihu.com/p/412838545 
- Kernel died: 查看command line里的报错信息。
  > OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
  
  删除./Anaconda3/envs/**XXX**/Lib/site-packages/torch/lib/libiomp5md.dll

### PyTorch
> **torch.nn.Conv2d**(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

> **torch.nn.BatchNorm2d**(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)

> **torch.nn.AvgPool2d**(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)

> **torch.flatten**(input, start_dim=0, end_dim=-1) -> Tensor

> **torch.nn.functional.pad**(input, pad, mode='constant', value=0)

> **torch.nn.MSELoss**(size_average=None, reduce=None, reduction='mean')

- 疑难杂症1：
  > **RuntimeError**: Function AddmmBackward returned an invalid gradient at index 1 - got [128, 1200] but expected shape compatible with [128, 2400]

  https://blog.csdn.net/Willjzq1/article/details/118030639 (所有数据放到同一个device)
  https://stackoverflow.com/questions/68222763/runtimeerror-function-addmmbackward-returned-an-invalid-gradient （检查网络参数数量）

- Save model:
  > torch.save(**model**.state_dict(), **PATH**)
- Load model:
  > **model** = **TheModelClass(args, kwargs)**<br>
  > **model**.load_state_dict(torch.load(**PATH**))<br>
  > **model**.eval()


### Keras
- Keras中自定义复杂的loss函数：
https://www.cnblogs.com/think90/articles/11652213.html 

### Pytorch - Keras
- BatchNorm：https://stackoverflow.com/questions/60079783/difference-between-keras-batchnormalization-and-pytorchs-batchnorm2d 	
- Lstm参数数量问题：https://blog.csdn.net/So_that/article/details/94731614 

### LSTM
- 如何调参
https://www.cnblogs.com/kamekin/p/10163743.html 

### Numpy
- 保存高纬度数组（array）和读取:
  > np.save(file=**PATH**, arr=**a**)<br>
  > np.load(**PATH**)
### Pandas
- 筛选and赋值语法**易错点**：, e.g. 
  > 正确：b_det.loc[b_det['det']==det_id, 'class'] = class2

  > <font color=red>错误：b_det[b_det['det']==det_id]['class'] = class2</font>

- 多条件筛选：
  > **df**.loc[(**con1**) & (**con2**) & (**con3**) **...**]

- drop multiple columns:
  > **df**.drop([**'col1', 'col2', 'col3'**], axis=1)

- save and load geopandas file:
  > **data_X_all**.to_file(**'./data/data_X_all.shp'**, driver='ESRI Shapefile')<br>
  > **data_X_all** = gpd.read_file(**'./data/data_X_all.shp'**)

### GeoPandas
- 保存GeoDataFrame (.shp file):
  > **gdf**.to_file(**PATH**, driver='ESRI Shapefile')
- 读取GeoDataFrame (.shp file):
  > gpd.read_file(**PATH**)
- 如何安装skmob/geopandas
  > conda install -c conda-forge scikit-mobility

### Matplotlib
- save image in svg format
  > **fig**.savefig(**PATH**, format='svg')

### Homebrew
- 报错 "zsh: command not found: brew" 的问题：

  在terminal里输入“export PATH=/opt/homebrew/bin:$PATH”
	https://stackoverflow.com/questions/36657321/after-installing-homebrew-i-get-zsh-command-not-found-brew

### GIS
- Software
  - How to use QGIS to view gdb file (generated from arcgis):
https://gis.stackexchange.com/questions/26285/installing-file-geodatabase-gdb-support-in-qgis 

  - How to export atibutes data to Excel file:
https://gis.stackexchange.com/questions/135801/exporting-attribute-table-to-excel-from-qgis

- Data
  - [Census] Geography Mapping Files:
https://www.census.gov/programs-surveys/geography/geographies/mapping-files.html

  - [Census] TIGER/Line Geodatabases:
https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-geodatabase-file.html

  - [Census] File Transfer Protocol (FTP) archive:
https://www2.census.gov/geo/tiger/

  - [Census] Understanding Geographic Identifiers (GEOIDs):
https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html

  - [Census] Accessing Census and ACS Data in Python:
https://pygis.io/docs/d_access_census.html

  - [Census] EPSG ID:
https://epsg.io/

  - [BART] BART Hourly Ridership Data:
  http://64.111.127.166/origin-destination/

  - [BART] BART Monthly Ridership Data:
  https://www.bart.gov/about/reports/ridership

### Parallel coding
- python thread:
https://github.com/HaTT2018/get_distance/blob/main/get_distance.ipynb
- python multiprocessing:
https://github.com/HaTT2018/BART_Ridership_Prediction/blob/main/Data%20-%20bart_data.py

### Parser
- reference: https://blog.csdn.net/qq_34243930/article/details/106517985
- example code:

  > import argparse<br>
  > <code>\# (1) 声明一个parser</code><br>
  > parser = argparse.ArgumentParser(description='test')<br>
  > <code>\# (2) 添加参数</code><br>
  > parser.add_argument("parg")  <code> # 位置参数，这里表示第一个出现的参数赋值给parg</code><br>
  > parser.add_argument("--digit",type=int,help="输入数字") <code># 通过 --echo xxx声明的参数，为int类型</code><br>
  > parser.add_argument("--name",help="名字",default="cjf") <code># 同上，default 表示默认值</code><br>
  > <code>\# (3) 读取命令行参数</code><br>
  > args = parser.parse_args()<br>
  > <code>\# (4) 调用这些参数</code><br>
  > print(args.parg)<br>
  > print("echo ={0}".format(args.digit))<br>
  > print("name = {}".format(args.name))w

### Colab
- Download publicly shared google drive file:
  > !pip install -U --no-cache-dir gdown --pre<br>
  > !gdown --id **FILE_ID** --no-cookies
- Command line commands
  - create folder
    > !mkdir **DIR_NAME**  <br>
  - move file
    > !mv **PATH1** **PATH2**

[Back To Top](#table-of-contents)
