import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Concatenate, Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint


def conv2D_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first Conv2D layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # second Conv2D layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x

def UnetPP_Model(input_image, n_filters=32, dropout=0.1, batchnorm=True):
    # Encoder (Contraction Path)
    E1 = conv2D_block(input_image, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(E1)
    #p1 = Dropout(dropout)(p1)

    E2 = conv2D_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(E2)
    #p2 = Dropout(dropout)(p2)

    E3 = conv2D_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(E3)
    #p3 = Dropout(dropout)(p3)

    E4 = conv2D_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(E4)
    #p4 = Dropout(dropout)(p4)

    E5 = conv2D_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    #Di1
    D_1_1 = Conv2DTranspose(n_filters * 1, kernel_size=(3, 3), strides=(2, 2), padding='same')(E2)
    D_1_1 = Concatenate()([D_1_1, E1])
    #D_1_1 = Dropout(dropout)(D_1_1)
    D_1_1 = conv2D_block(D_1_1, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    
    D_2_1 = Conv2DTranspose(n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(E3)
    D_2_1 = Concatenate()([D_2_1, E2])
    #D_2_1 = Dropout(dropout)(D_2_1)
    D_2_1 = conv2D_block(D_2_1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    
    D_3_1 = Conv2DTranspose(n_filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')(E4)
    D_3_1 = Concatenate()([D_3_1, E3])
    #D_3_1 = Dropout(dropout)(D_3_1)
    D_3_1 = conv2D_block(D_3_1, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    
    D_4_1 = Conv2DTranspose(n_filters * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')(E5)
    D_4_1 = Concatenate()([D_4_1, E4])
    #D_4_1 = Dropout(dropout)(D_4_1)
    D_4_1 = conv2D_block(D_4_1, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    
    #Di2
    D_1_2 = Conv2DTranspose(n_filters * 1, kernel_size=(3, 3), strides=(2, 2), padding='same')(D_2_1)
    D_1_2 = Concatenate()([D_1_2, D_1_1, E1])
    #D_1_2 = Dropout(dropout)(D_1_2)
    D_1_2 = conv2D_block(D_1_2, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    
    D_2_2 = Conv2DTranspose(n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(D_3_1)
    D_2_2 = Concatenate()([D_2_2, D_2_1, E2])
    #D_2_2 = Dropout(dropout)(D_2_2)
    D_2_2 = conv2D_block(D_2_2, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    
    D_3_2 = Conv2DTranspose(n_filters * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')(D_4_1)
    D_3_2 = Concatenate()([D_3_2, D_3_1, E3])
    #D_3_2 = Dropout(dropout)(D_3_2)
    D_3_2 = conv2D_block(D_3_2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    
    #Di3
    D_1_3 = Conv2DTranspose(n_filters * 1, kernel_size=(3, 3), strides=(2, 2), padding='same')(D_2_2)
    D_1_3 = Concatenate()([D_1_3, D_1_2, D_1_1, E1])
    #D_1_3 = Dropout(dropout)(D_1_3)
    D_1_3 = conv2D_block(D_1_3, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    
    D_2_3 = Conv2DTranspose(n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(D_3_2)
    D_2_3 = Concatenate()([D_2_3, D_2_2, D_2_1, E2])
    #D_2_3 = Dropout(dropout)(D_2_3)
    D_2_3 = conv2D_block(D_2_3, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    
    #Di4
    D_1_4 = Conv2DTranspose(n_filters * 1, kernel_size=(3, 3), strides=(2, 2), padding='same')(D_2_3)
    D_1_4 = Concatenate()([D_1_4, D_1_3, D_1_2, D_1_1, E1])
    #D_1_4 = Dropout(dropout)(D_1_4)
    D_1_4 = conv2D_block(D_1_4, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(4, (1, 1), activation='sigmoid')(D_1_4)
    model = Model(inputs=[input_image], outputs=[outputs])
    return model