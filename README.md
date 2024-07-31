# DREAM

DREAM (DNA cis-Regulatory Elements with controllable Activity design platforM) is an efficient, scalable and explainable computational framework to design CREs from scratch. 
![flowchart](https://github.com/cisGrammar/DREAM/blob/master/img/dream_fig.jpg)

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dna-sequence-encoding-and-dataset-preparation)
- [DREAM Training](#dream-training)
- [CRE Optimization](#cre-optimization)

## Installation

### Prerequisites

- Python 3.10.13
- TensorFlow 2.13
- scikit-learn
- h5py
- pysam
- deap 1.4.1

### Install Dependencies

```bash
conda install -r requirements.txt
```
### Clone Repository

```
git clone https://github.com/cisGrammar/DREAM.git
```

## DNA Sequence Encoding and Dataset Preparation

The script `encode_sequences.py` encodes DNA sequences into one-hot format and saves them as an HDF5 dataset.

### Usage

Command Line Arguments
- `--REFERENCE`: Path to the reference genome in FASTA format (default: '/data/reference/melanogaster/dm3.fa').
- `--output_h5`: Path to save the output HDF5 file (default: 'alldata.h5').
- `--traindata_df`: Path to the training dataset CSV file (default: 'train_dataset.csv').
- `--valdata_df`: Path to the validation dataset CSV file (default: 'val_dataset.csv').
- `--testdata_df`: Path to the test dataset CSV file (default: 'test_dataset.csv').
### Example Usage
To encode DNA sequences and prepare datasets, use the following command:
```bash
python dataset/encode_sequences.py --REFERENCE '/data/reference/melanogaster/dm3.fa' --output_h5 'alldata.h5' --traindata_df 'train_dataset.csv' --valdata_df 'val_dataset.csv' --testdata_df 'test_dataset.csv'

```

## DREAM Training

 The script `train.py` TensorFlow script for training an SE-ResNet model on enhancer activity dataset(e.g., STARR-seq dataset, MPRA dataset).
### Usage

### Command Line Arguments

The script `train.py` supports the following command line arguments:

- `--cuda_devices`: Specifies CUDA_VISIBLE_DEVICES setting (default: '0').
- `--data_path`: Path to the HDF5 data file containing training, validation, and test datasets (default: 'alldata.h5').
- `--batch_size`: Batch size for training (default: 512).
- `--epochs`: Number of epochs for training (default: 100).
- `--patience`: Patience for early stopping (default: 10).
- `--checkpoint_path`: Prefix for the model checkpoint files (default: 'checkpoint').

### Example Usage

To train the SE-ResNet model, use the following command:

```bash
python train.py --cuda_devices '0' --data_path 'path/to/your/data.h5' --batch_size 512 --epochs 100 --patience 10 --checkpoint_path 'your_checkpoint_prefix'
```

## CRE Optimization
The `GA/ga.py` is the implementation of an evolutionary algorithm using the DEAP framework to optimize the CRE activity based on the CRE activtiy prediction module's output.

### Usage
### Command-line Arguments
The `GA/ga.py` script supports the following command-line arguments:

- `--sequence_length`: Length of the DNA sequences to generate (default: 249).
- `--nucleotide_frequency`: Frequencies of nucleotides `[A, C, G, T]` (default: [0.25, 0.25, 0.25, 0.25]).
- `--seed`: Random seed for TensorFlow and NumPy (default: 12345).
- `--cuda_devices`: CUDA visible devices (default: "0").
- `--best_model_checkpoint`: Path to the best model checkpoint for fitness prediction (default: "checkpoint_keras/").
- `--indpb`: Probability of mutating each nucleotide in an individual (default: 0.025).
- `--population_size`: Initial population size (default: 100000).
- `--NGEN`: Number of generations to evolve (default: 90).
- `--output_file`: Path to save the evolution fitness dataframe CSV (default: "evolution_fits_df.csv").
## Example
```bash
python evolutionary_algorithm.py --sequence_length 249 --nucleotide_frequency 0.25 0.25 0.25 0.25 --seed 12345 --cuda_devices "0" --best_model_checkpoint "/path/to/best_model/" --indpb 0.025 --population_size 100000 --NGEN 90 --output_file "/path/to/output.csv"
```