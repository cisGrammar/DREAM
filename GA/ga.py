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
randomizer = np.random

args = {'sequence_length': 249, 'nucleotide_frequency': [0.25, 0.25, 0.25, 0.25]}

tf.keras.utils.set_random_seed(12345)

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


best_model_checkpoint = "/data/lizhaohong/project/notebook/melanogaster_enhancer_regression_v3/checkpoint_keras_0317_relu_all/"
best_model = load_model(best_model_checkpoint)


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

def fitness(dna_population: list):
    onehot_encoding = DNAoneHotEncoding()
    dna_dataset = ["".join(dna_list) for dna_list in dna_population]
    dna_encoding = np.stack([onehot_encoding(dna) for dna in dna_dataset])
    #predict_fitness = best_model.predict(dna_encoding)[:,0]
    predict_fitness = best_model.predict(dna_encoding)
    predict_fitness = np.stack(predict_fitness).squeeze(axis=2).T[:,0]
    return [(i, ) for i in predict_fitness.squeeze().tolist()]


def mutation(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            if individual[i] == 'A':
                individual[i] = (randomizer.choice(list('CGT'), p=[
                    args['nucleotide_frequency'][1] / (1 - args['nucleotide_frequency'][0]),
                    args['nucleotide_frequency'][2] / (1 - args['nucleotide_frequency'][0]),
                    args['nucleotide_frequency'][3] / (1 - args['nucleotide_frequency'][0])]))
            elif individual[i] == 'C':
                individual[i] = (randomizer.choice(list('AGT'), p=[
                    args['nucleotide_frequency'][0] / (1 - args['nucleotide_frequency'][1]),
                    args['nucleotide_frequency'][2] / (1 - args['nucleotide_frequency'][1]),
                    args['nucleotide_frequency'][3] / (1 - args['nucleotide_frequency'][1])]))
            elif individual[i] == 'G':
                individual[i] = (randomizer.choice(list('CGT'), p=[
                    args['nucleotide_frequency'][2] / (1 - args['nucleotide_frequency'][2]),
                    args['nucleotide_frequency'][1] / (1 - args['nucleotide_frequency'][2]),
                    args['nucleotide_frequency'][3] / (1 - args['nucleotide_frequency'][2])]))
            elif individual[i] == 'T':
                individual[i] = (randomizer.choice(list('CGT'), p=[
                    args['nucleotide_frequency'][0] / (1 - args['nucleotide_frequency'][3]),
                    args['nucleotide_frequency'][1] / (1 - args['nucleotide_frequency'][3]),
                    args['nucleotide_frequency'][2] / (1 - args['nucleotide_frequency'][3])]))
    return individual,


def random_sequence_generator(randomizer, args):
    return randomizer.choice(list('ACGT'), p=args['nucleotide_frequency'])


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)
#creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("base", random_sequence_generator, randomizer, args)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.base, n=args['sequence_length'])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutation, indpb=0.025)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=100000)
NGEN = 90

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
    
top10 = tools.selBest(population, k=10)

import pandas as pd
evolution_fits_df = pd.DataFrame(evolution_fits)
evolution_fits_df.to_csv("/data/lizhaohong/project/notebook/melanogaster_enhancer_regression_v3/evolution_fits_df.csv", index=False)

