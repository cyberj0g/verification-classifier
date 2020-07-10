"""
This module contains various CNN models used for verification
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Concatenate, Subtract
from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input
IMG_SIZE = 160
BATCH_SIZE = 32
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_learning_rate = 0.005


def create_model2():
    """
    Dual-head model based on VGG16
    @return:
    """
    # Create the base model from the pre-trained model MobileNet V2
    base_model1 = tf.keras.applications.VGG16 (input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # keep layer names unique
    for l in base_model1.layers:
        l._name = l._name+'_1'

    base_model1.trainable = False
    base_model2 = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model2.trainable = False

    flat1 = tf.keras.layers.Flatten()(base_model1.output)
#    flat1 = tf.keras.layers.Dropout(rate=0.5)(flat1)

    flat2 = tf.keras.layers.Flatten()(base_model2.output)
#    flat2 = tf.keras.layers.Dropout(rate=0.5)(flat2)

    dense = tf.keras.layers.Dense(1, activation='tanh')
    dense1 = dense(flat1)
    dense2 = dense(flat2)

    concat = tf.keras.layers.Concatenate()([dense1, dense2])
    predict_layer = tf.keras.layers.Softmax()(concat)

    model = Model(inputs=[base_model1.inputs, base_model2.inputs], outputs=predict_layer)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

def create_model3():
    # Create the base model from the pre-trained model MobileNet V2
    base_model1 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model1.trainable = True

    flat1 = tf.keras.layers.GlobalAveragePooling2D()(base_model1.output)
    #flat1 = tf.keras.layers.Dropout(rate=0.5)(flat1)

    dense = tf.keras.layers.Dense(2, activation='tanh')
    dense1 = dense(flat1)

    predict_layer = tf.keras.layers.Softmax()(dense1)

    model = Model(inputs=base_model1.inputs, outputs=predict_layer)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

def create_model():
    """
    Dual-head model based on MobileNet v2
    @return:
    """
    # Create the base model from the pre-trained model MobileNet V2
    base_model1 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # keep layer names unique
    for l in base_model1.layers:
        l._name = l._name+'_1'

    base_model1.trainable = True
    base_model2 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model2.trainable = True

    flat1 = tf.keras.layers.GlobalAveragePooling2D()(base_model1.output)
    flat1 = tf.keras.layers.Dropout(rate=0.5)(flat1)

    flat2 = tf.keras.layers.GlobalAveragePooling2D()(base_model2.output)
    flat2 = tf.keras.layers.Dropout(rate=0.5)(flat2)

    dense = tf.keras.layers.Dense(1, activation='tanh')
    dense1 = dense(flat1)
    dense2 = dense(flat2)

    concat = tf.keras.layers.Concatenate()([dense1, dense2])
    predict_layer = tf.keras.layers.Softmax()(concat)

    model = Model(inputs=[base_model1.inputs, base_model2.inputs], outputs=predict_layer)
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

def create_feature_extractor():
    """
    Feature extractor network based on MobileNet v2
    @return:
    """
    # Create the base model from the pre-trained model MobileNet V2
    base_model1 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    flat1 = tf.keras.layers.Flatten()(base_model1.output)

    model = Model(inputs=base_model1.inputs, outputs=flat1)
    return model