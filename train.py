import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate,Flatten, Conv1D, AveragePooling1D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pysam import FastaFile
import os 
import h5py
from sklearn.model_selection import train_test_split
from model import se_resnet

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


def __main__():
    x_train, y_train, x_val, y_val, x_test, y_test = load_data("alldata.h5")
    model = se_resnet()
    model.compile(loss = 'mse', optimizer = Nadam(learning_rate=1e-5), metrics=['mse'])
    batch_size = 512
    epochs = 100
    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(filepath="checkpoint_keras_0314", monitor="val_loss", save_best_only=True)
    ]
    model.fit(
    	x_train, y_train,
    	epochs=epochs, shuffle=True, 
    	batch_size=batch_size, validation_data= (x_val, y_val),
    	callbacks=callbacks_list
	)









