# ChaosMining

[![arXiv](https://img.shields.io/badge/arXiv-2406.12150-b31b1b.svg)](https://arxiv.org/abs/2406.12150)
[![HuggingFace Datasets](https://img.shields.io/badge/🤗-Datasets-FFD21E)](https://huggingface.co/datasets/geshijoker/chaosmining)
[![DOI](https://img.shields.io/badge/DOI-10.57967/hf/2482-blue)](https://huggingface.co/datasets/geshijoker/chaosmining)

论文 ["ChaosMining: A Benchmark to Evaluate Post-Hoc Local Attribution Methods in Low SNR Environments"](https://arxiv.org/abs/2406.12150) 的源代码。

一个用于评估事后局部归因方法（如积分梯度、显著图、DeepLift 等）在**低信噪比（SNR）**环境下性能的综合基准测试框架，涵盖**音频处理**、**视觉识别**和**符号仿真**三种模态。

---

## 数据集

数据集可通过 [HuggingFace Hub](https://huggingface.co/datasets/geshijoker/chaosmining) 获取 (DOI: [10.57967/hf/2482](https://huggingface.co/datasets/geshijoker/chaosmining))。

每种模态提供四种配置变体：

| 变体 | 含义 |
|---------|---------|
| **RBFP** | 随机背景，固定位置 (Random Background, Fixed Position) |
| **RBRP** | 随机背景，随机位置 (Random Background, Random Position) |
| **SBFP** | 静态背景，固定位置 (Static Background, Fixed Position) |
| **SBRP** | 静态背景，随机位置 (Static Background, Random Position) |

**HuggingFace 配置名称：**

| 模态 | 配置名称 |
|----------|-------------|
| 音频 | `audio_RBFP`, `audio_RBRP`, `audio_SBFP`, `audio_SBRP` |
| 视觉 | `vision_RBFP`, `vision_RBRP`, `vision_SBFP`, `vision_SBRP` |
| 符号仿真 | 本地 data 文件夹中的 formula.csv |

---

## 仓库结构

```
.
├── chaosmining/                      # 核心 Python 包
│   ├── __init__.py
│   ├── data_utils.py                # 共享工具和数据集加载器
│   │                                 #   - create_simulation_data()     生成合成符号数据
│   │                                 #   - ChaosVisionHFDataset        从 HF Hub 加载视觉数据
│   │                                 #   - ChaosAudioDataset           从本地 CSV 加载音频数据
│   │                                 #   - read_formulas()             加载符号公式
│   ├── utils.py                     # 通用工具函数
│   │                                 #   - check_make_dir() / clean_directory()
│   │                                 #   - radar_factory()             雷达图可视化
│   ├── audio/                       # 音频处理模块
│   │   ├── __init__.py
│   │   ├── data_hf.py              # ParquetAudioDataset — 从 Parquet 文件加载音频
│   │   ├── functions.py            # HFChaosMiningAudioDataset — 从 HF Hub 加载音频
│   │   ├── main.py                 # 训练循环、评估、参数解析
│   │   └── models.py               # 8 种音频架构：
│   │                                #   AudioRNN / AudioLSTM / AudioTCN / AudioTrans
│   │                                #   AudioWav2Vec2 / AudioRNNT / AudioConformer
│   ├── vision/                      # 视觉处理模块
│   │   ├── __init__.py
│   │   ├── main.py                 # 训练循环、评估、参数解析
│   │   ├── models.py               # 完整的 torchvision 模型定义
│   │   └── contribs.py             # 归因可视化和 IoU 指标
│   └── simulation/                  # 符号仿真模块
│       ├── __init__.py
│       ├── main.py                 # 参数解析
│       ├── models.py               # MLPRegressor / MLPResRegressor
│       └── functions.py            # 归因评估指标和特征选择评分
├── data/                            # 数据占位符（从 HuggingFace 下载）
│   ├── audio/                       # audio_RBFP / audio_RBRP / audio_SBFP / audio_SBRP
│   ├── vision/                      # vision_RBFP / vision_RBRP / vision_SBFP / vision_SBRP
│   └── symbolic_simulation/         # formula.csv
├── data_engineer/                   # 合成数据生成脚本
│   ├── create_audio_data.py        # 生成前景/背景混合的音频数据集
│   └── create_vision_data.py       # 生成可控合成的视觉数据集
├── examples/                        # 训练、评估和归因基准测试脚本
│   ├── train_eval_audio.py                 # 音频模型训练（支持 HF Hub）
│   ├── eval_audio_localization.py          # 音频归因基准测试
│   ├── train_eval_vision.py                # 视觉模型训练（支持 HF Hub）
│   ├── eval_vision_localization.py         # 视觉归因基准测试
│   ├── train_eval_simulation.py            # 符号数据 MLP 训练
│   ├── train_eval_simulation_topk.py       # Top-k 特征选择评估
│   ├── train_eval_simulation_overparam_topk.py  # 过参数化 top-k 评估
│   └── RFEwNA_simulation.py               # 基于归因的递归特征消除
├── notebooks/                      # Jupyter 笔记本（初步结果、图表）
├── exps/                           # Bash 脚本占位符（依赖系统环境）
├── setup.py
├── requirements.txt
└── README.md
```

---

## 数据加载方式

本框架支持**三种**数据加载策略：

### 1. HuggingFace Hub（推荐）
直接从 HuggingFace 加载数据集，无需本地存储。

**音频：**
```python
from chaosmining.audio.functions import HFChaosMiningAudioDataset

dataset = HFChaosMiningAudioDataset(
    subset_name="audio_RBFP",  # 或 audio_RBRP, audio_SBFP, audio_SBRP
    split="train",             # 或 "val"
    target_length=16000
)
# 返回: (波形 [通道数, 采样数], 类别索引, 位置, 采样率)
```

**视觉：**
```python
from chaosmining.data_utils import ChaosVisionHFDataset

dataset = ChaosVisionHFDataset(
    hf_dataset_name="geshijoker/chaosmining",
    config_name="vision_RBFP",  # 或 vision_RBRP, vision_SBFP, vision_SBRP
    split="train"
)
# 返回: (图像张量, 标签张量)
# 标签: [前景类别, 位置_x, 位置_y]
```

### 2. 本地 Parquet 文件
从本地 Parquet 文件加载音频（高效的列式格式）：

```python
from chaosmining.audio.data_hf import ParquetAudioDataset

dataset = ParquetAudioDataset(
    data_dir="./data/audio/RBFP/",
    target_length=16000
)
# 返回: (波形张量, 标签索引)
```

### 3. 本地 CSV 数据集
传统的 CSV 元数据 + WAV/图像文件加载方式：

```python
from chaosmining.data_utils import ChaosAudioDataset

dataset = ChaosAudioDataset(
    root="./data/audio/RBFP/",
    split="train",
    csv_file="meta_data.csv"
)
# 返回: (波形 [通道数, 采样数], 类别索引, 位置, 采样率)
```

---

## 模型库

### 音频模型（8 种架构）

| 模型 | 描述 | 关键特性 |
|-------|-------------|-------------|
| `AudioRNN` | 基于 RNN 的分类器 | Conv1d 特征提取器，3 层 RNN，可配置隐藏维度 |
| `AudioLSTM` | 基于 LSTM 的分类器 | LSTM 单元，擅长长距离依赖建模 |
| `AudioTCN` | 时序卷积网络 | 4 个 Conv1d 模块，逐步下采样 |
| `AudioTrans` | Transformer 编码器 | 多头自注意力，6 头编码器 |
| `AudioWav2Vec2` | Wav2Vec 2.0 风格 | 自定义 7 层 Conv + Transformer，支持冻结特征提取器 |
| `AudioRNNT` | RNN-Transducer | Mel 频谱图输入，Emformer 编码器 |
| `AudioConformer` | Conformer | 卷积增强的 Transformer，Mel 频谱图特征 |

### 视觉模型

通过 torchvision 提供 14 种架构，支持预训练 (`--pretrained`) 和从头训练：

- **ResNet 系列**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **DenseNet 系列**: `densenet121`, `densenet161`
- **ViT 系列**: `vit_b_16`, `vit_l_32`
- **其他**: `alexnet`, `googlenet`, `vgg16`, `swin_t`, `efficientnet_v2_s`, `convnext_tiny`

### 仿真模型

| 模型 | 描述 |
|-------|-------------|
| `MLPRegressor` | 可配置 MLP，GELU 激活和 Dropout |
| `MLPResRegressor` | 带残差连接的 MLP，适用于深层网络 |

### 归因方法（基于 Captum）

| 方法 | 命令行标识 |
|--------|-----------|
| Saliency | `sa` |
| Integrated Gradients | `ig` |
| DeepLift | `dl` |
| Feature Ablation | `fa` |
| Guided Backprop | `gb` |
| LIME | `lime` |
| KernelShap | `ks` |
| DeepLiftShap | `dls` |

### 评估指标

| 指标 | 描述 |
|--------|-------------|
| Top-k 准确率 | 标准分类准确率 @1, @5 |
| IoU (交并比) | 视觉定位质量评估 |
| 平均绝对误差 (MAE) | 仿真回归精度 |
| Top Features Score | 正确识别相关特征的百分比 |
| Uniformity Score | 归因方法一致性度量 |

---

## 安装

```console
git clone https://github.com/geshijoker/ChaosMining.git
cd ChaosMining
python -m venv env
source env/bin/activate
pip install -e .
```

设置 HuggingFace Token 以访问数据集（可选，公开数据集可匿名访问）：

```console
export HF_TOKEN="your_huggingface_token_here"
```

国内用户可设置 HuggingFace 镜像：

```console
export HF_ENDPOINT="https://hf-mirror.com"
```

---

## 生成合成数据

```console
# 音频数据（需要 Speech Commands + 背景音频）
python data_engineer/create_audio_data.py \
    --input_path_fg FOREGROUND_INPUT_PATH \
    --input_path_bg BACKGROUND_INPUT_PATH \
    --output_path OUTPUT_PATH

# 视觉数据（使用 CIFAR-10 前景 + Flowers102 背景）
python data_engineer/create_vision_data.py \
    --input_path INPUT_PATH \
    --output_path OUTPUT_PATH
```

---

## 运行实验

### 仿真（符号函数数据）

```console
# 在符号函数上训练和评估 MLP
python examples/train_eval_simulation.py \
    -d ./data/symbolic_simulation/formula.csv \
    -e ./runs/simulation/ -n 14 -s SEED \
    --num_noises 100 --ny_var 0.01 \
    --optimizer Adam --learning_rate 0.001 --deterministic

# 基于归因的递归特征消除
python examples/RFEwNA_simulation.py \
    -d ./data/symbolic_simulation/formula.csv \
    -e ./runs/RFEwNA -n rfe_ig -s SEED -g 0 \
    --num_noises 100 --ny_var 0.01 \
    --optimizer Adam --learning_rate 0.001 --dropout 0.0 --xai ig --deterministic
```

### 视觉

```console
# 使用 HuggingFace 数据训练视觉模型（推荐）
python examples/train_eval_vision.py \
    -c RBFP \
    -e ./runs/vision/RBFP/ -n arc_vit_b_16 -s SEED \
    --model_name vit_b_16 --gpu 0 --num_classes 10 \
    --num_epochs 30 --batch_size 128 --learning_rate 0.001 --pretrained --deterministic --debug

# 归因基准测试
python examples/eval_vision_localization.py \
    -d ./data/vision/RBFP/ -e ./runs/vision/RBFP/ \
    -n arc_vit_b_16 -s SEED --model_name vit_b_16 \
    --gpu 0 --num_classes 10 --batch_size 2 --deterministic
```

### 音频

```console
# 使用 HuggingFace 数据训练音频模型（推荐，无需本地下载）
python examples/train_eval_audio.py \
    --hf_dataset audio_SBRP \
    -e ./runs/audio/SBRP/ -n arc_Conformer -s SEED \
    --model_name Conformer --n_channels 10 --length 16000 \
    --gpu 0 --num_epochs 30 --batch_size 128 --learning_rate 0.001 --deterministic --debug

# 或使用本地数据路径训练
python examples/train_eval_audio.py \
    -d ./data/audio/RBFP/ \
    -e ./runs/audio/RBFP/ -n arc_TRAN -s SEED \
    --model_name TRAN --n_channels 10 --length 16000 \
    --gpu 0 --num_epochs 30 --batch_size 128 --learning_rate 0.0001 --deterministic

# 音频归因基准测试
python examples/eval_audio_localization.py \
    -d ./data/audio/RBFP/ -e ./runs/audio/RBFP/ \
    -n arc_RNN -s SEED --model_name RNN \
    --n_channels 10 --length 16000 --gpu 0 --batch_size 32 --deterministic
```

### 支持的音频模型

将 `--model_name` 替换为以下任意一项：`RNN`, `LSTM`, `TCN`, `TRAN`, `Wav2Vec2Model`, `RNNT`, `Conformer`

---

## 引用

如果您在研究中使用本代码或数据集，请引用以下论文：

```bibtex
@misc{shi2024chaosminingbenchmarkevaluateposthoc,
      title={ChaosMining: A Benchmark to Evaluate Post-Hoc Local Attribution Methods 
             in Low SNR Environments}, 
      author={Ge Shi and Ziwen Kan and Jason Smucny and Ian Davidson},
      year={2024},
      eprint={2406.12150},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.12150}, 
}

@misc{ge_shi_2024,
      author       = {{Ge Shi}},
      title        = {chaosmining (Revision 6c23193)},
      year         = 2024,
      url          = {https://huggingface.co/datasets/geshijoker/chaosmining},
      doi          = {10.57967/hf/2482},
      publisher    = {Hugging Face}
}
```

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。