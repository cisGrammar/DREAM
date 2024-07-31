import sys
import random
from deap import creator, base, tools, algorithms
import numpy as np
import copy
from tqdm import tqdm
import copy
from os.path import splitext, exists, dirname, join, basename
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import argparse

randomizer = np.random

# Default values
DEFAULT_SEQUENCE_LENGTH = 249
DEFAULT_NUCLEOTIDE_FREQUENCY = [0.25, 0.25, 0.25, 0.25]
DEFAULT_SEED = 12345
DEFAULT_BEST_MODEL_CHECKPOINT = "/data/lizhaohong/project/notebook/melanogaster_enhancer_regression_v3/checkpoint_keras_0317_relu_all/"
DEFAULT_INDPB = 0.025
DEFAULT_POPULATION_SIZE = 100000
DEFAULT_NGEN = 90
DEFAULT_OUTPUT_FILE = "/data/lizhaohong/project/notebook/melanogaster_enhancer_regression_v3/evolution_fits_df.csv"

# Command line argument parsing
parser = argparse.ArgumentParser(description='Evolutionary Algorithm for DNA Sequences')
parser.add_argument('--sequence_length', type=int, default=DEFAULT_SEQUENCE_LENGTH,
                    help='Length of DNA sequences')
parser.add_argument('--nucleotide_frequency', type=float, nargs=4, default=DEFAULT_NUCLEOTIDE_FREQUENCY,
                    help='Nucleotide frequencies [A, C, G, T]')
parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                    help='Random seed for numpy and TensorFlow')
parser.add_argument('--best_model_checkpoint', type=str, default=DEFAULT_BEST_MODEL_CHECKPOINT,
                    help='Path to the best model checkpoint')
parser.add_argument('--indpb', type=float, default=DEFAULT_INDPB,
                    help='Mutation probability')
parser.add_argument('--n', type=int, default=DEFAULT_POPULATION_SIZE,
                    help='Initial population size')
parser.add_argument('--NGEN', type=int, default=DEFAULT_NGEN,
                    help='Number of generations for optimization')
parser.add_argument('--output_file', type=str, default=DEFAULT_OUTPUT_FILE,
                    help='Output file path for evolution fitness dataframe CSV')
args = parser.parse_args()

# Setting random seed
tf.keras.utils.set_random_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Setting CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Enable GPU memory growth:
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Load best model
best_model = load_model(args.best_model_checkpoint)

# DNA one-hot encoding class
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

# Fitness function evaluation
def fitness(dna_population: list):
    onehot_encoding = DNAoneHotEncoding()
    dna_dataset = ["".join(dna_list) for dna_list in dna_population]
    dna_encoding = np.stack([onehot_encoding(dna) for dna in dna_dataset])
    predict_fitness = best_model.predict(dna_encoding)
    predict_fitness = np.stack(predict_fitness).squeeze(axis=2).T[:,0]
    return [(i, ) for i in predict_fitness.squeeze().tolist()]

# Mutation function
def mutation(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            if individual[i] == 'A':
                individual[i] = (randomizer.choice(list('CGT'), p=[
                    args.nucleotide_frequency[1] / (1 - args.nucleotide_frequency[0]),
                    args.nucleotide_frequency[2] / (1 - args.nucleotide_frequency[0]),
                    args.nucleotide_frequency[3] / (1 - args.nucleotide_frequency[0])]))
            elif individual[i] == 'C':
                individual[i] = (randomizer.choice(list('AGT'), p=[
                    args.nucleotide_frequency[0] / (1 - args.nucleotide_frequency[1]),
                    args.nucleotide_frequency[2] / (1 - args.nucleotide_frequency[1]),
                    args.nucleotide_frequency[3] / (1 - args.nucleotide_frequency[1])]))
            elif individual[i] == 'G':
                individual[i] = (randomizer.choice(list('CGT'), p=[
                    args.nucleotide_frequency[2] / (1 - args.nucleotide_frequency[2]),
                    args.nucleotide_frequency[1] / (1 - args.nucleotide_frequency[2]),
                    args.nucleotide_frequency[3] / (1 - args.nucleotide_frequency[2])]))
            elif individual[i] == 'T':
                individual[i] = (randomizer.choice(list('CGT'), p=[
                    args.nucleotide_frequency[0] / (1 - args.nucleotide_frequency[3]),
                    args.nucleotide_frequency[1] / (1 - args.nucleotide_frequency[3]),
                    args.nucleotide_frequency[2] / (1 - args.nucleotide_frequency[3])]))
    return individual,

# Random sequence generator
def random_sequence_generator(randomizer, args):
    return randomizer.choice(list('ACGT'), p=args.nucleotide_frequency)

def main():
    # DEAP Toolbox initialization
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("base", random_sequence_generator, randomizer, args)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.base, n=args.sequence_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutation, indpb=args.indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Population initialization
    population = toolbox.population(n=args.n)
    NGEN = args.NGEN

    # Evolutionary algorithm loop
    evolution_fits = {
        "generation": [],
        "fitness": [],
        "sequences":[],
        "id":[]
    }

    for gen in tqdm(range(NGEN)):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.1, mutpb=0.1)
        fits = toolbox.evaluate(offspring)
        id = 0
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            evolution_fits["fitness"].append(fit[0])
            evolution_fits["generation"].append(gen)
            evolution_fits["sequences"].append("".join(ind))
            evolution_fits["id"].append(id)
            id = id + 1
        population = toolbox.select(offspring, k=len(population))
        
    # Selecting top 10 individuals
    top10 = tools.selBest(population, k=10)

    # Saving evolution data to CSV
    import pandas as pd
    evolution_fits_df = pd.DataFrame(evolution_fits)
    evolution_fits_df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
