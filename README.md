# Image retrieval

## 准备工作

### 数据集

将图片数据集放在 `dataset` 文件夹下，例如 `oxbuild` 和 `holiday` 数据集，则需要这样安排

```
.
|-- dataset
    |-- oxbuild
    |   |-- images
    |       |-- all_souls.jpg
    |       |-- oxford.jpg
    |       |-- ...
    |
    |-- holiday
        |-- images
            |-- 001.jpg
            |-- 002.jpg
            |-- ...
```

### 依赖

建议使用 conda 安装依赖，或使用虚拟环境安装该部分依赖

```sh
pip install -r requirements.txt
```

### 构建索引

#### CLIP

在 `src/clip/constants.py` 修改模型的名称（本程序使用的是相对路径，弱能够连上 huggingface，则可以直接写模型名称）

对于使用 CLIP 的索引，使用下面的命令构建

```sh
python src/clip/run_init_index.py --dataset dataset/oxbuild
```

在此过程中，可能会遇到模型无法下载的问题，可以选择先从 huggingface 上将该模型下载到本地，然后使用相对路径来指定使用的模型。使用镜像下载的方法详见 https://zhuanlan.zhihu.com/p/663712983

索引构建时间较为漫长，打包中将会包含该索引的二进制文件

#### SIFT

```sh
python src/sift/run_init_index.py --dataset dataset/oxbuild --n_clusters 100
```

索引构建时间较为漫长，打包中将会包含该索引的二进制文件

## 查询

使用命令行运行

```sh
streamlit run home.py
```

打开控制台中显示的网页链接（即本地服务链接 http://localhost:8501/ ）即可进入查询界面
