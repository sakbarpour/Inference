import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import Graph
from sklearn.model_selection import train_test_split
import re
import logging
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Concatenate, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.metrics import accuracy
import imageio.v2
import matplotlib.pyplot as plt
import math
from scipy import ndimage as ndi
import random
from sklearn.model_selection import KFold
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Dense, GlobalAveragePooling2D, Multiply, Reshape, BatchNormalization, Activation
from tensorflow.keras.models import Model








# Define custom metrics for binary segmentation
def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(tf.maximum(y_true, y_pred))
    iou_score = intersection / (union + tf.keras.backend.epsilon())
    return iou_score

def dice_coef(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2 * intersection) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())
    return dice
def binary_accuracy(y_true, y_pred):
    # Flatten the true and predicted masks to compute binary accuracy
    y_true_flat = tf.keras.backend.flatten(y_true)
    y_pred_flat = tf.keras.backend.flatten(y_pred)
    return tf.keras.metrics.binary_accuracy(y_true_flat, y_pred_flat)

def step_decay(epoch, initial_lr=0.001, drop_factor=0.5, epochs_drop=10):
    learning_rate = initial_lr * (drop_factor ** (epoch // epochs_drop))
    return learning_rate


def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y

def binary_unet(input_shape, num_classes, activation='sigmoid'):
    # down
    input_layer = Input(input_shape)
    conv1 = conv_block(input_layer, nfilters=32)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block(conv1_out, nfilters=32*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block(conv2_out, nfilters=32*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block(conv3_out, nfilters=32*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.3)(conv4_out)
    
    conv5 = conv_block(conv4_out, nfilters=32*16)
    conv5 = Dropout(0.3)(conv5)
    
    # Additional convolutional layers for more depth
    conv6 = conv_block(conv5, nfilters=32*16)
    conv7 = conv_block(conv6, nfilters=32*16)
    
    # up
    deconv6 = deconv_block(conv7, residual=conv4, nfilters=32*8)
    deconv6 = Dropout(0.3)(deconv6)
    
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=32*4)
    deconv7 = Dropout(0.3)(deconv7)
    
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=32*2)
    
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=32)
    
    output_layer = Conv2D(num_classes, 1, activation=activation)(deconv9)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model