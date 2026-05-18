# SV_DJ_cluster
本项目在随机置换与4L进制编解码的安全向量处理方法的基础上，引入K-Means聚类索引和两阶段检索机制，并采用Damgard-Jurik同态加密扩展明文空间，设计并实现了基于聚类索引与增强加密的特征匹配算法。项目实现了：

- 加密特征离线登记与持久化存储
- 聚类索引构建（球面 K-Means + 软聚类分配）
- 在线两阶段匹配：簇中心粗配 + 候选库精配
- 可选 CRT 加速解密，以提升 Damgård–Jurik 解密效率

## 目录结构
```
SV-main/
├── data/
│   ├── agedb/
│   │   ├── agedb_feat.list
│   │   └── pair.list
│   ├── cfp/
│   │   ├── cfp_feat.list
│   │   └── pair.list
│   ├── ijb/
│   │   ├── ijbb_feat.list
│   │   ├── ijbc_feat.list
│   │   └── meta/
│   │       ├── ijbb_face_tid_mid.txt
│   │       ├── ijbb_template_pair_label.txt
│   │       ├── ijbc_face_tid_mid.txt
│   │       └── ijbc_template_pair_label.txt
│   └── lfw/
│       ├── lfw_feat.list
│       └── pair.list
├── eval/
│   ├── eval_1v1.py
│   ├── eval_1vn.py
│   ├── eval1.sh
│   ├── evalijbx.sh
│   └── ...
└── libs/
    ├── ASE/
    ├── baseline/
    ├── IronMask/
    ├── SecureVector/
    ├── SFM/
    ├── SV_cluster/
    ├── SV_DJ/
    └── SV_DJ_cluster/
        ├── build_index.py
        ├── cluster_match.py
        ├── crypto_system.py
        ├── dj_crt_decrypt.py
        ├── enrollment.py
        ├── keys/
        └── cache/
```
## 依赖环境

推荐使用 Python 3.8+，安装以下依赖：

```bash
pip install numpy scipy scikit-learn faiss-cpu phe gmpy2 damgard_jurik
```

> 如果在 Windows 环境中运行，建议使用 WSL 或 Linux，因为部分脚本依赖 `resource` 模块。

## 数据准备

1. 特征文件格式：`<id> <feat1> <feat2> ...`
2. pair 文件格式：`<probe_id> <gallery_id> <label>`
3. 特征会在加载时进行 L2 归一化
4. 建议使用 MagFace 或类似人脸特征提取器生成特征

## 快速开始

### 1. 生成密钥

```bash
python libs/SV_DJ_cluster/crypto_system.py --genkey 1 --key_size 512 --s 1
```

生成结果保存在：

- `libs/SV_DJ_cluster/keys/publickey_512_s1.npy`
- `libs/SV_DJ_cluster/keys/privatekey_512_s1.npy`
- `libs/SV_DJ_cluster/keys/factors_512_s1.json`

### 2. 加密特征登记

```bash
python libs/SV_DJ_cluster/enrollment.py \
  --K 64 --s 1 --key_size 512 \
  --feat_list data/lfw/lfw_feat.list \
  --folder data/lfw/enrolled \
  --public_key libs/SV_DJ_cluster/keys/publickey_512_s1.npy
```

### 3. 构建聚类索引

```bash
python libs/SV_DJ_cluster/build_index.py \
  --feat_list data/lfw/lfw_feat.list \
  --pair_list data/lfw/pair.list \
  --folder data/lfw/enrolled \
  --n_clusters 100 \
  --key_size 512 --K 64
```

该命令会生成聚类中心、软聚类分配信息和索引元数据。

### 4. 两阶段聚类匹配

```bash
python libs/SV_DJ_cluster/cluster_match.py \
  --folder data/lfw/enrolled \
  --pair_list data/lfw/pair.list \
  --score_list data/lfw/score.list \
  --top_k 5 --key_size 512 --K 64 --jobs 4
```

启用 CRT 解密：

```bash
python libs/SV_DJ_cluster/cluster_match.py --crt_decrypt \
  --folder data/lfw/enrolled \
  --pair_list data/lfw/pair.list \
  --score_list data/lfw/score.list \
  --top_k 5 --key_size 512 --K 64 --jobs 4
```

## 模块说明

### `build_index.py`

- 使用 Faiss 的球面 K-Means 聚类
- 对每个 gallery 特征执行软聚类分配
- 将 gallery 样本同时分配给多个相似簇，以提高召回率

### `cluster_match.py`

- 阶段一：探针与聚类中心匹配，生成候选簇集合
- 阶段二：对候选 gallery 样本执行精确加密相似度计算
- 输出 `score.list` 以及匹配效率、mAP、召回、候选集统计等指标

### `crypto_system.py`

- 支持密钥生成、密钥加载、加密相似度计算
- 支持 CRT 加速解密
- 可以生成配套性能指标文件

## 运行提示

- `--n_clusters` 和 `--top_k` 控制召回与效率的权衡
- `--jobs` 可提高多核匹配性能
- 保持 `key_size`、`K`、`s` 在登记与匹配时一致

