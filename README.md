# DREAM
deep learning-based approach for synthetic enhancer design


# SE-ResNet Training Script

This repository contains a TensorFlow script for training an SE-ResNet model on custom data. SE-ResNet is a variant of ResNet that incorporates Squeeze-and-Excitation blocks for improved performance.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.10.13
- TensorFlow 2.13
- scikit-learn
- h5py
- pysam

### Install Dependencies

```bash
pip install -r requirements.txt
```
## Clone Repository

```
git clone https://github.com/cisGrammar/DREAM.git
```

## Usage

### Command Line Arguments

The script `train.py` supports the following command line arguments:

- `--cuda_devices`: Specifies CUDA_VISIBLE_DEVICES setting (default: '0').
- `--data_path`: Path to the HDF5 data file containing training, validation, and test datasets (default: 'alldata.h5').
- `--batch_size`: Batch size for training (default: 512).
- `--epochs`: Number of epochs for training (default: 100).
- `--patience`: Patience for early stopping (default: 10).
- `--checkpoint_path`: Prefix for the model checkpoint files (default: 'checkpoint_keras_0314').

### Example Usage

To train the SE-ResNet model, use the following command:

```bash
python train.py --cuda_devices '0' --data_path 'path/to/your/data.h5' --batch_size 512 --epochs 100 --patience 10 --checkpoint_path 'your_checkpoint_prefix'
```





