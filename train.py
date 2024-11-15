import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, Flatten, Conv1D, AveragePooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from model import se_resnet  # Assuming se_resnet is defined 
from utils import load_data  # Assuming load_data is defined

# Set random seed for reproducibility
tf.random.set_seed(12345)

def parse_args():
    parser = argparse.ArgumentParser(description='SE-ResNet Training Script')
    parser.add_argument('--cuda_devices', default='0', type=str, help='CUDA_VISIBLE_DEVICES setting')
    parser.add_argument('--data_path', default='alldata.h5', type=str, help='Path to data file')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs for training')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping')
    parser.add_argument('--checkpoint_path', default='checkpoint_keras', type=str, help='Checkpoint file path prefix')

    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.data_path)

    # Build model
    model = se_resnet()  # Assuming se_resnet() builds the SE-ResNet model

    # Compile model
    model.compile(loss='mse', optimizer=Nadam(learning_rate=1e-5), metrics=['mse'])

    # Define callbacks
    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=args.patience),
        ModelCheckpoint(filepath=args.checkpoint_path, monitor='val_loss', save_best_only=True)
    ]

    # Train model
    model.fit(
        x_train, y_train,
        epochs=args.epochs, shuffle=True,
        batch_size=args.batch_size, validation_data=(x_val, y_val),
        callbacks=callbacks_list
    )

if __name__ == '__main__':
    main()
    
