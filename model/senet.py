import numpy as np
from scipy.stats import spearmanr, pearsonr
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate,Flatten, Conv1D, AveragePooling1D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os 


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=8):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
        self.fc2 = tf.keras.layers.Dense(units=input_channels)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = self.fc1(branch)
        branch = tf.nn.gelu(branch)
        branch = self.dropout(self.fc2(branch))
        branch = tf.nn.sigmoid(branch)
        output = tf.keras.layers.multiply(inputs=[inputs, branch])
        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=1,
                                            strides=1,
                                            padding='valid')
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,
                                            kernel_size=3,
                                            strides=stride,
                                            padding='same',
                                            kernel_regularizer = tf.keras.regularizers.l2(1e-4)
                                            )
        self.conv3 = tf.keras.layers.Conv1D(filters=filter_num * 4,
                                            kernel_size=1,
                                            strides=1,
                                            padding='same')
        self.se = SEBlock(input_channels=filter_num * 4)

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv1D(filters=filter_num * 4,
                                                   kernel_size=1,
                                                   strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):
        identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = tf.nn.gelu(x)
        x = self.conv2(x)
        x = tf.nn.gelu(x)
        x = self.conv3(x)
        x = self.se(x)
        output = tf.nn.gelu(tf.keras.layers.add([identity, x]))
        return output


class SEResNet(tf.keras.Model):
    def __init__(self, block_num, model_name):
        super(SEResNet, self).__init__()
        self.model_name = model_name

        self.pre1 = tf.keras.layers.Conv1D(filters=512,
                                           kernel_size=7,
                                           strides=1,
                                           padding='same',
                                           kernel_initializer = 'glorot_normal',
                                           kernel_regularizer = tf.keras.regularizers.l2(2e-4)
                                           )
        self.pre2 = tf.keras.layers.BatchNormalization()
        self.pre3 = tf.keras.layers.Activation(tf.keras.activations.exponential)
        self.pre4 = tf.keras.layers.AveragePooling1D(pool_size=5, strides=2)

        self.layer1 = self._make_res_block(filter_num=128,
                                           blocks=block_num[0])
        self.layer2 = self._make_res_block(filter_num=256,
                                           blocks=block_num[1],
                                           stride=1)
        self.layer3 = self._make_res_block(filter_num=512,
                                           blocks=block_num[2],
                                           stride=2)
        self.layer4 = self._make_res_block(filter_num=512,
                                           blocks=block_num[3],
                                           stride=2)

        self.flatten = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.fc1 = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.gelu,
         kernel_regularizer = tf.keras.regularizers.l2(2e-5))
        self.fc2 = tf.keras.layers.Dense(units=64, activation=tf.keras.activations.gelu,
         kernel_regularizer = tf.keras.regularizers.l2(1e-4))
        self.fc3 = tf.keras.layers.Dense(units=2, activation='linear')

    def _make_res_block(self, filter_num, blocks, stride=1):
        res_block = tf.keras.Sequential()
        res_block.add(BottleNeck(filter_num, stride=stride))

        for _ in range(1, blocks):
            res_block.add(BottleNeck(filter_num, stride=1))

        return res_block

    @tf.function(input_signature=[
        tf.TensorSpec([None, 249, 4], tf.float32, name="inputs")
        ])
    def call(self, inputs):
        pre1 = self.pre1(inputs)
        pre2 = self.pre2(pre1, training=None)
        pre3 = self.pre3(pre2)
        pre4 = self.pre4(pre3)
        l1 = self.layer1(pre4, training=None)
        l2 = self.layer2(l1, training=None)
        l3 = self.layer3(l2, training=None)
        l4 = self.layer4(l3, training=None)
        flatten = self.dropout1(self.flatten(l4))
        out = self.fc2(self.dropout2(self.fc1(flatten)))
        out = self.fc3(out)
        return out

    def __repr__(self):
        return "SE_ResNet_{}".format(self.model_name)


def se_resnet():
    return SEResNet(block_num=[2, 2, 2, 2], model_name="model")
