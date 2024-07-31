import numpy as np
import pandas as pd
from pysam import FastaFile
import h5py
import argparse

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


def rc(sequence: str) -> str:
    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
    return "".join(complement.get(base, base) for base in reversed(sequence))


def getOnehotEncoding(df, ref_genome, verbose=True):
    fasta_obj = FastaFile(ref_genome)
    dna_encoding = DNAoneHotEncoding()
    if verbose:
        seq_num = 0
        print('one-hot encoding in progress ...', flush=True)
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
                print(seq_num, 'sequences processed', flush=True, end='\r')
    if verbose:
        print('finished one-hot encoding:', seq_num, 'sequences processed', flush=True)

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


def main(args):
    REFERENCE = args.REFERENCE
    output_h5 = args.output_h5
    traindata_df = args.traindata_df
    valdata_df = args.valdata_df
    testdata_df = args.testdata_df

    traindata = pd.read_csv(traindata_df, header=0, delimiter="\t")
    traindata.columns = ["seqnames", "start", "end", "strand", "dev", "hk"]

    valdata = pd.read_csv(valdata_df, header=0, delimiter="\t")
    valdata.columns = ["seqnames", "start", "end", "strand", "dev", "hk"]

    testdata = pd.read_csv(testdata_df, header=0, delimiter="\t")
    testdata.columns = ["seqnames", "start", "end", "strand", "dev", "hk"]

    x_train = getOnehotEncoding(traindata, ref_genome=REFERENCE)
    y_train = traindata[[i for i in traindata.columns.values if "dev" in i or "hk" in i]].values

    x_val = getOnehotEncoding(valdata, ref_genome=REFERENCE)
    y_val = valdata[[i for i in valdata.columns.values if "dev" in i or "hk" in i]].values

    x_test = getOnehotEncoding(testdata, ref_genome=REFERENCE)
    y_test = testdata[[i for i in testdata.columns.values if "dev" in i or "hk" in i]].values

    save_dataset(output_h5, x_train, y_train, x_val, y_val, x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DNA Sequence Encoding and Dataset Preparation')
    parser.add_argument('--REFERENCE', default='/data/reference/melanogaster/dm3.fa', type=str, help='Path to the reference genome')
    parser.add_argument('--output_h5', default='alldata.h5', type=str, help='Path to save the HDF5 output file')
    parser.add_argument('--traindata_df', default='train_dataset.csv', type=str, help='Path to training dataset CSV file')
    parser.add_argument('--valdata_df', default='val_dataset.csv', type=str, help='Path to validation dataset CSV file')
    parser.add_argument('--testdata_df', default='test_dataset.csv', type=str, help='Path to test dataset CSV file')

    args = parser.parse_args()
    main(args)
