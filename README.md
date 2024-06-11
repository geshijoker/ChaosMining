# ChaosMining
Source code of "A Benchmark to Evaluate Post-Hoc Local Attribution Methods in Low SNR Environments"

# Repo Structure

    .
    ├── chaosmining              # Source files, define the major modules to build machine learning pipeline with pytorch
    ├── data                    # Data folder, including functional, vision, and audio data
    ├── data_engineer           # Source files, create synthetic dataset and conduct preprocessing
    ├── examples                # Source files, perform model training, evaluation, and localization 
    ├── notebooks               # Source files, notebooks to get preliminary results
    ├── exps                    # Bash scripts, linux commands to run bash experiments
    ├── LICENSE
    └── README.md

# Dataset Structure  
The code to generate data from existing datasets is included in this repository. The metadata of this dataset is organized and stored on [zenodo](https://zenodo.org/records/11582545) with a DOI 10.5281/zenodo.11582544

./data/
├── audio/
│   ├── RBFP/
│   │   ├── train/
│   │   │   ├── meta_data.csv
│   │   │   └── ...
│   │   ├── eval/
│   │   │   ├── meta_data.csv
│   │   │   └── ...
│   ├── RBRP/
│   │   └── ...
│   ├── SBFP/
│   │   └── ...
│   ├── SBRP/
│   │   └── ...
├── vision/
│   ├── RBFP/
│   │   ├── train/
│   │   │   ├── meta_data.csv
│   │   │   └── ...
│   │   ├── eval/
│   │   │   ├── meta_data.csv
│   │   │   └── ...
│   ├── RBRP/
│   │   └── ...
│   ├── SBFP/
│   │   └── ...
│   ├── SBRP/
│   │   └── ...
└── symbolic_simulation/
    └── formula.csv

# Install Environment
## Install a list of requirements specified in a Requirements File.
```console
foo@bar:~$ python3 -m pip install -r requirements.txt
```
