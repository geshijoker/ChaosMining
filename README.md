# ChaosMining

[![arXiv](https://img.shields.io/badge/arXiv-2406.12150-b31b1b.svg)](https://arxiv.org/abs/2406.12150)
[![HuggingFace Datasets](https://img.shields.io/badge/🤗-Datasets-FFD21E)](https://huggingface.co/datasets/geshijoker/chaosmining)
[![DOI](https://img.shields.io/badge/DOI-10.57967/hf/2482-blue)](https://huggingface.co/datasets/geshijoker/chaosmining)

Source code of ["ChaosMining: A Benchmark to Evaluate Post-Hoc Local Attribution Methods in Low SNR Environments"](https://arxiv.org/abs/2406.12150).

A comprehensive benchmarking framework for evaluating post-hoc local attribution methods (e.g., Integrated Gradients, Saliency, DeepLift, Guided Backprop) in low Signal-to-Noise Ratio (SNR) environments, spanning **audio processing**, **vision recognition**, and **symbolic simulation** modalities.

---

## Dataset

The dataset is accessible on [HuggingFace Hub](https://huggingface.co/datasets/geshijoker/chaosmining) (DOI: [10.57967/hf/2482](https://huggingface.co/datasets/geshijoker/chaosmining)).

Four configuration variants are available for each modality:

| Variant | Meaning |
|---------|---------|
| **RBFP** | Random Background, Fixed Position |
| **RBRP** | Random Background, Random Position |
| **SBFP** | Static Background, Fixed Position |
| **SBRP** | Static Background, Random Position |

**HuggingFace config names:**

| Modality | Config Names |
|----------|-------------|
| Audio | `audio_RBFP`, `audio_RBRP`, `audio_SBFP`, `audio_SBRP` |
| Vision | `vision_RBFP`, `vision_RBRP`, `vision_SBFP`, `vision_SBRP` |
| Symbolic Simulation | Single formula.csv file in local data folder |

---

## Repository Structure

```
.
├── chaosmining/                      # Core Python package
│   ├── __init__.py
│   ├── data_utils.py                # Shared utilities & dataset loaders
│   │                                 #   - create_simulation_data()     Generate synthetic symbolic data
│   │                                 #   - ChaosVisionHFDataset        Load vision data from HF Hub
│   │                                 #   - ChaosAudioDataset           Load audio data from local CSV
│   │                                 #   - read_formulas()             Load symbolic formulas
│   ├── utils.py                     # General utilities
│   │                                 #   - check_make_dir() / clean_directory()
│   │                                 #   - radar_factory()            Radar chart visualization
│   ├── audio/                       # Audio processing module
│   │   ├── __init__.py
│   │   ├── data_hf.py              # ParquetAudioDataset — load audio from Parquet files
│   │   ├── functions.py            # HFChaosMiningAudioDataset — load audio from HF Hub
│   │   ├── main.py                 # Training loop, evaluation, argument parser
│   │   └── models.py               # 8 audio architectures:
│   │                                #   AudioRNN / AudioLSTM / AudioTCN / AudioTrans
│   │                                #   AudioWav2Vec2 / AudioRNNT / AudioConformer
│   ├── vision/                      # Vision processing module
│   │   ├── __init__.py
│   │   ├── main.py                 # Training loop, evaluation, argument parser
│   │   ├── models.py               # Full torchvision model definitions
│   │   └── contribs.py             # Attribution visualization & IoU metrics
│   └── simulation/                  # Symbolic simulation module
│       ├── __init__.py
│       ├── main.py                 # Argument parser
│       ├── models.py               # MLPRegressor / MLPResRegressor
│       └── functions.py            # Attribution metrics & feature selection scoring
├── data/                            # Data placeholder (download from HuggingFace)
│   ├── audio/                       # audio_RBFP / audio_RBRP / audio_SBFP / audio_SBRP
│   ├── vision/                      # vision_RBFP / vision_RBRP / vision_SBFP / vision_SBRP
│   └── symbolic_simulation/         # formula.csv
├── data_engineer/                   # Synthetic data generation scripts
│   ├── create_audio_data.py        # Generate audio dataset with foreground/background mixing
│   └── create_vision_data.py       # Generate vision dataset with controlled composition
├── examples/                        # Training, evaluation & attribution benchmarking scripts
│   ├── train_eval_audio.py                 # Audio model training (supports HF Hub)
│   ├── eval_audio_localization.py          # Audio attribution benchmarking
│   ├── train_eval_vision.py                # Vision model training (supports HF Hub)
│   ├── eval_vision_localization.py         # Vision attribution benchmarking
│   ├── train_eval_simulation.py            # MLP training on symbolic data
│   ├── train_eval_simulation_topk.py       # Top-k feature selection evaluation
│   ├── train_eval_simulation_overparam_topk.py  # Overparameterized top-k evaluation
│   └── RFEwNA_simulation.py               # Recursive Feature Elimination with XAI
├── notebooks/                      # Jupyter notebooks (preliminary results, plots)
├── exps/                           # Bash scripts placeholder (system-dependent)
├── setup.py
├── requirements.txt
└── README.md
```

---

## Data Loading

The benchmark supports **three** data loading strategies:

### 1. HuggingFace Hub (Recommended)
Load datasets directly from HuggingFace — no local storage required.

**Audio:**
```python
from chaosmining.audio.functions import HFChaosMiningAudioDataset

dataset = HFChaosMiningAudioDataset(
    subset_name="audio_RBFP",  # or audio_RBRP, audio_SBFP, audio_SBRP
    split="train",             # or "val"
    target_length=16000
)
# Returns: (waveform [channels, samples], class_idx, position, sample_rate)
```

**Vision:**
```python
from chaosmining.data_utils import ChaosVisionHFDataset

dataset = ChaosVisionHFDataset(
    hf_dataset_name="geshijoker/chaosmining",
    config_name="vision_RBFP",  # or vision_RBRP, vision_SBFP, vision_SBRP
    split="train"
)
# Returns: (image_tensor, landmarks_tensor)
# Landmarks: [foreground_label, position_x, position_y]
```

### 2. Local Parquet Files
Load audio from local Parquet files (efficient columnar format):

```python
from chaosmining.audio.data_hf import ParquetAudioDataset

dataset = ParquetAudioDataset(
    data_dir="./data/audio/RBFP/",
    target_length=16000
)
# Returns: (waveform_tensor, label_index)
```

### 3. Local CSV-based Dataset
Traditional CSV metadata + WAV/image files:

```python
from chaosmining.data_utils import ChaosAudioDataset

dataset = ChaosAudioDataset(
    root="./data/audio/RBFP/",
    split="train",
    csv_file="meta_data.csv"
)
# Returns: (waveform [channels, samples], class_idx, position, sample_rate)
```

---

## Model Zoo

### Audio Models (8 architectures)

| Model | Description | Key Features |
|-------|-------------|-------------|
| `AudioRNN` | RNN-based classifier | Conv1d feature extractor, 3-layer RNN, configurable hidden dim |
| `AudioLSTM` | LSTM-based classifier | LSTM cells for long-range dependencies |
| `AudioTCN` | Temporal Convolutional Network | 4 Conv1d blocks with progressive downsampling |
| `AudioTrans` | Transformer encoder | Multi-head self-attention, 6-head encoder |
| `AudioWav2Vec2` | Wav2Vec 2.0-style | Custom 7-layer Conv feature extractor + Transformer, optional frozen extractor |
| `AudioRNNT` | RNN-Transducer | Mel-spectrogram input, Emformer encoder |
| `AudioConformer` | Conformer | Convolution-augmented transformer, Mel-spectrogram features |

### Vision Models

14 architectures via torchvision with custom classification heads. Support both pretrained (`--pretrained`) and scratch training:

- **ResNet** family: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **DenseNet** family: `densenet121`, `densenet161`
- **ViT** family: `vit_b_16`, `vit_l_32`
- **Others**: `alexnet`, `googlenet`, `vgg16`, `swin_t`, `efficientnet_v2_s`, `convnext_tiny`

### Simulation Models

| Model | Description |
|-------|-------------|
| `MLPRegressor` | Configurable MLP with GELU activation and dropout |
| `MLPResRegressor` | MLP with residual connections for deeper networks |

### Attribution Methods (via Captum)

| Method | Code Flag |
|--------|-----------|
| Saliency | `sa` |
| Integrated Gradients | `ig` |
| DeepLift | `dl` |
| Feature Ablation | `fa` |
| Guided Backprop | `gb` |
| LIME | `lime` |
| KernelShap | `ks` |
| DeepLiftShap | `dls` |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Top-k Accuracy | Standard classification accuracy @1, @5 |
| IoU (Intersection over Union) | Vision localization quality |
| Mean Absolute Error | Regression accuracy for simulation |
| Top Features Score | Percentage of correctly identified relevant features |
| Uniformity Score | Attribution method consistency metric |

---

## Installation

```console
git clone https://github.com/geshijoker/ChaosMining.git
cd ChaosMining
python -m venv env
source env/bin/activate
pip install -e .
```

Set your HuggingFace token for dataset access:

```console
export HF_TOKEN="your_huggingface_token_here"
```

---

## Generating Synthetic Data

```console
# Audio data (requires Speech Commands + background audio)
python data_engineer/create_audio_data.py \
    --input_path_fg FOREGROUND_INPUT_PATH \
    --input_path_bg BACKGROUND_INPUT_PATH \
    --output_path OUTPUT_PATH

# Vision data (uses CIFAR-10 foreground + Flowers102 background)
python data_engineer/create_vision_data.py \
    --input_path INPUT_PATH \
    --output_path OUTPUT_PATH
```

---

## Running Experiments

### Simulation (Symbolic Functional Data)

```console
# Train and evaluate MLP on symbolic functions
python examples/train_eval_simulation.py \
    -d ./data/symbolic_simulation/formula.csv \
    -e ./runs/simulation/ -n 14 -s SEED \
    --num_noises 100 --ny_var 0.01 \
    --optimizer Adam --learning_rate 0.001 --deterministic

# Recursive Feature Elimination with attribution (XAI)
python examples/RFEwNA_simulation.py \
    -d ./data/symbolic_simulation/formula.csv \
    -e ./runs/RFEwNA -n rfe_ig -s SEED -g 0 \
    --num_noises 100 --ny_var 0.01 \
    --optimizer Adam --learning_rate 0.001 --dropout 0.0 --xai ig --deterministic
```

### Vision

```console
# Train vision model using HuggingFace data (recommended)
python examples/train_eval_vision.py \
    -c RBFP \
    -e ./runs/vision/RBFP/ -n arc_vit_b_16 -s SEED \
    --model_name vit_b_16 --gpu 0 --num_classes 10 \
    --num_epochs 30 --batch_size 128 --learning_rate 0.001 --pretrained --deterministic --debug

# Attribution benchmarking
python examples/eval_vision_localization.py \
    -d ./data/vision/RBFP/ -e ./runs/vision/RBFP/ \
    -n arc_vit_b_16 -s SEED --model_name vit_b_16 \
    --gpu 0 --num_classes 10 --batch_size 2 --deterministic
```

### Audio

```console
# Train audio model using HuggingFace data (recommended, no local download)
python examples/train_eval_audio.py \
    --hf_dataset audio_SBRP \
    -e ./runs/audio/SBRP/ -n arc_Conformer -s SEED \
    --model_name Conformer --n_channels 10 --length 16000 \
    --gpu 0 --num_epochs 30 --batch_size 128 --learning_rate 0.001 --deterministic --debug

# Or train using local data path
python examples/train_eval_audio.py \
    -d ./data/audio/RBFP/ \
    -e ./runs/audio/RBFP/ -n arc_TRAN -s SEED \
    --model_name TRAN --n_channels 10 --length 16000 \
    --gpu 0 --num_epochs 30 --batch_size 128 --learning_rate 0.0001 --deterministic

# Audio attribution benchmarking
python examples/eval_audio_localization.py \
    -d ./data/audio/RBFP/ -e ./runs/audio/RBFP/ \
    -n arc_RNN -s SEED --model_name RNN \
    --n_channels 10 --length 16000 --gpu 0 --batch_size 32 --deterministic
```

### Supported Audio Models

Replace `--model_name` with any of: `RNN`, `LSTM`, `TCN`, `TRAN`, `Wav2Vec2Model`, `RNNT`, `Conformer`

---

## Citation

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.