import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
from pysam import FastaFile
import os 
import h5py
from sklearn.model_selection import train_test_split

REFERENCE = "/data/reference/melanogaster/dm3.fa"

class DNAoneHotEncoding:
    """
    DNA sequences one hot encoding
    """

    def __call__(self, sequence: str):
        assert (len(sequence) > 0)
        encoding = np.zeros((len(sequence), 4), dtype="float32")
        A = np.array([1, 0, 0, 0])
        C = np.array([0, 1, 0, 0])
        G = np.array([0, 0, 1, 0])
        T = np.array([0, 0, 0, 1])
        for index, nuc in enumerate(sequence):
            if nuc == "A":
                encoding[index, :] = A
            elif nuc == "C":
                encoding[index, :] = C
            elif nuc == "G":
                encoding[index, :] = G
            elif nuc == "T":
                encoding[index, :] = T
        return encoding


def rc(sequence:str)->str:
	complement = {"A":"T", "C":"G", "G":"C", "T":"A"}
	return "".join( complement.get(base, base) for base in reversed(sequence) )


def getOnehotEncoding(df, ref_genome, verbose = True): 
    fasta_obj = FastaFile(ref_genome)
    dna_encoding = DNAoneHotEncoding()
    if verbose:
        seq_num = 0
        print('one-hot encoding in progress ...', flush = True)
    assert "seqnames" in df.columns.values
    assert "start" in df.columns.values
    assert "end" in df.columns.values
    assert "strand" in df.columns.values
    sequences = []

    for _, i in df.iterrows():
    	sequence = fasta_obj.fetch(reference=str(i[0]), start=int(i[1]), end=int(i[2]) + 1)
    	sequence = sequence.upper()
    	if i.strand == "-":
    		sequences.append(rc(sequence))
    	else:
    		sequences.append(sequence)

    one_hot_sequences = []
    for sequence in sequences:
        one_hot_sequences.append(dna_encoding(sequence))
        if verbose:
            seq_num += 1
            if seq_num % 1000 == 0:
                print(seq_num, 'sequences processed', flush = True, end = '\r')      
    if verbose:
        print('finished one-hot encoding:', seq_num, 'sequences processed', flush = True)
    
    one_hot_sequences = np.stack(one_hot_sequences)

    return one_hot_sequences


def load_data(file_path):

    # load dataset
    dataset = h5py.File(file_path, 'r')
    x_train = np.array(dataset['x_train']).astype(np.float32)
    y_train = np.array(dataset['y_train']).astype(np.float32)
    x_val = np.array(dataset['x_val']).astype(np.float32)
    y_val = np.array(dataset['y_val']).astype(np.float32)
    x_test = np.array(dataset['x_test']).astype(np.float32)
    y_test = np.array(dataset['y_test']).astype(np.float32)

    return x_train, y_train, x_val, y_val, x_test, y_test

def save_dataset(savepath, x_train, y_train, x_valid, y_valid, x_test, y_test):
    f = h5py.File(savepath, "w")
    dset = f.create_dataset("x_train", data=x_train, compression="gzip")
    dset = f.create_dataset("y_train", data=y_train, compression="gzip")
    dset = f.create_dataset("x_val", data=x_valid, compression="gzip")
    dset = f.create_dataset("y_val", data=y_valid, compression="gzip")
    dset = f.create_dataset("x_test", data=x_test, compression="gzip")
    dset = f.create_dataset("y_test", data=y_test, compression="gzip")
    f.close()
 


