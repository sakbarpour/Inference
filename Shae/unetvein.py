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
from keras import backend as K



def binary_unified_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Calculate binary cross-entropy loss
    ce_loss = - (y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)) + (1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-7, 1.0)))
    
    # Calculate focal loss
    pt = tf.exp(-ce_loss)  # Probabilities
    focal_loss = alpha * (1 - pt)**gamma * ce_loss
    
    # Average the focal loss over the batch
    batch_loss = tf.reduce_mean(focal_loss)
    
    return batch_loss
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score
# Define the Binary Focal Loss function for binary segmentation
def binary_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_loss = -alpha * (1 - pt) ** gamma * tf.math.log(tf.clip_by_value(pt, 1e-7, 1.0))
    return tf.reduce_mean(focal_loss)





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

def binary_unet_old(input_shape, num_classes, activation='sigmoid'):
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




# Define a convolution block with dilated convolution
def conv_block_attention(tensor, nfilters, size=3, padding='same', dilation_rate=1, initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, dilation_rate=dilation_rate, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, dilation_rate=dilation_rate, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# Define a convolution block with attention gate
def attention_gate(x, g, nfilters, initializer="he_normal"):
    g = Conv2D(nfilters, kernel_size=(1, 1), kernel_initializer=initializer)(g)
    x = Conv2D(nfilters, kernel_size=(1, 1), kernel_initializer=initializer)(x)
    phi = Activation('relu')(tf.add(x, g))
    phi = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', kernel_initializer=initializer)(phi)
    return tf.multiply(x, phi)

# Define a deconvolution block
def deconv_block_attention(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block_attention(y, nfilters)
    return y

def binary_unet(input_shape, num_classes, activation='sigmoid'):
    # Down
    input_layer = Input(input_shape)
    conv1 = conv_block_attention(input_layer, nfilters=32)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block_attention(conv1_out, nfilters=32*2, dilation_rate=2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block_attention(conv2_out, nfilters=32*4, dilation_rate=4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block_attention(conv3_out, nfilters=32*8, dilation_rate=8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.3)(conv4_out)
    
    conv5 = conv_block_attention(conv4_out, nfilters=32*16, dilation_rate=16)
    conv5 = Dropout(0.3)(conv5)
    
    # Additional convolutional layers for more depth
    conv6 = conv_block_attention(conv5, nfilters=32*16, dilation_rate=32)
    conv7 = conv_block_attention(conv6, nfilters=32*16, dilation_rate=64)
    
    # Up
    deconv6 = deconv_block_attention(conv7, residual=conv4, nfilters=32*8)
    deconv6 = Dropout(0.3)(deconv6)
    
    att6 = attention_gate(deconv6, conv4, nfilters=32*8)
    deconv6_with_attention = concatenate([deconv6, att6], axis=3)
    
    deconv7 = deconv_block_attention(deconv6_with_attention, residual=conv3, nfilters=32*4)
    deconv7 = Dropout(0.3)(deconv7)
    
    att7 = attention_gate(deconv7, conv3, nfilters=32*4)
    deconv7_with_attention = concatenate([deconv7, att7], axis=3)
    
    deconv8 = deconv_block_attention(deconv7_with_attention, residual=conv2, nfilters=32*2)
    
    att8 = attention_gate(deconv8, conv2, nfilters=32*2)
    deconv8_with_attention = concatenate([deconv8, att8], axis=3)
    
    deconv9 = deconv_block_attention(deconv8_with_attention, residual=conv1, nfilters=32)
    
    att9 = attention_gate(deconv9, conv1, nfilters=32)
    deconv9_with_attention = concatenate([deconv9, att9], axis=3)
    
    # Additional layers - You can add more if needed
    conv10 = conv_block_attention(deconv9_with_attention, nfilters=32)
    
    # Output layer
    output_layer = Conv2D(num_classes, 1, activation=activation)(conv10)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
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
from keras import backend as K



def binary_unified_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Calculate binary cross-entropy loss
    ce_loss = - (y_true * tf.math.log(tf.clip_by_value(y_pred, 1e-7, 1.0)) + (1 - y_true) * tf.math.log(tf.clip_by_value(1 - y_pred, 1e-7, 1.0)))
    
    # Calculate focal loss
    pt = tf.exp(-ce_loss)  # Probabilities
    focal_loss = alpha * (1 - pt)**gamma * ce_loss
    
    # Average the focal loss over the batch
    batch_loss = tf.reduce_mean(focal_loss)
    
    return batch_loss
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score
# Define the Binary Focal Loss function for binary segmentation
def binary_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_loss = -alpha * (1 - pt) ** gamma * tf.math.log(tf.clip_by_value(pt, 1e-7, 1.0))
    return tf.reduce_mean(focal_loss)





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

def binary_unet_old(input_shape, num_classes, activation='sigmoid'):
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




# Define a convolution block with dilated convolution
def conv_block_attention(tensor, nfilters, size=3, padding='same', dilation_rate=1, initializer="he_normal"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, dilation_rate=dilation_rate, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, dilation_rate=dilation_rate, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

# Define a convolution block with attention gate
def attention_gate(x, g, nfilters, initializer="he_normal"):
    g = Conv2D(nfilters, kernel_size=(1, 1), kernel_initializer=initializer)(g)
    x = Conv2D(nfilters, kernel_size=(1, 1), kernel_initializer=initializer)(x)
    phi = Activation('relu')(tf.add(x, g))
    phi = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', kernel_initializer=initializer)(phi)
    return tf.multiply(x, phi)

# Define a deconvolution block
def deconv_block_attention(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block_attention(y, nfilters)
    return y

def binary_unet(input_shape, num_classes, activation='sigmoid'):
    # Down
    input_layer = Input(input_shape)
    conv1 = conv_block_attention(input_layer, nfilters=32)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_block_attention(conv1_out, nfilters=32*2, dilation_rate=2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_block_attention(conv2_out, nfilters=32*4, dilation_rate=4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_block_attention(conv3_out, nfilters=32*8, dilation_rate=8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.3)(conv4_out)
    
    conv5 = conv_block_attention(conv4_out, nfilters=32*16, dilation_rate=16)
    conv5 = Dropout(0.3)(conv5)
    
    # Additional convolutional layers for more depth
    conv6 = conv_block_attention(conv5, nfilters=32*16, dilation_rate=32)
    conv7 = conv_block_attention(conv6, nfilters=32*16, dilation_rate=64)
    
    # Up
    deconv6 = deconv_block_attention(conv7, residual=conv4, nfilters=32*8)
    deconv6 = Dropout(0.3)(deconv6)
    
    att6 = attention_gate(deconv6, conv4, nfilters=32*8)
    deconv6_with_attention = concatenate([deconv6, att6], axis=3)
    
    deconv7 = deconv_block_attention(deconv6_with_attention, residual=conv3, nfilters=32*4)
    deconv7 = Dropout(0.3)(deconv7)
    
    att7 = attention_gate(deconv7, conv3, nfilters=32*4)
    deconv7_with_attention = concatenate([deconv7, att7], axis=3)
    
    deconv8 = deconv_block_attention(deconv7_with_attention, residual=conv2, nfilters=32*2)
    
    att8 = attention_gate(deconv8, conv2, nfilters=32*2)
    deconv8_with_attention = concatenate([deconv8, att8], axis=3)
    
    deconv9 = deconv_block_attention(deconv8_with_attention, residual=conv1, nfilters=32)
    
    att9 = attention_gate(deconv9, conv1, nfilters=32)
    deconv9_with_attention = concatenate([deconv9, att9], axis=3)
    
    # Additional layers - You can add more if needed
    conv10 = conv_block_attention(deconv9_with_attention, nfilters=32)
    
    # Output layer
    output_layer = Conv2D(num_classes, 1, activation=activation)(conv10)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
