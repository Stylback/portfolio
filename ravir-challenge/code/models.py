#-------------------
# Prerequisite modules

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras import Model

#--------------------
# Basic binary UNET model

def get_unet_BI(base, img_w, img_h, img_c):
    
    '''
    Basic UNET model for binary classification
    Parameters:
        base: Int larger than 0. Number of filters to start
        img_w: Int larger than 0. Pixel width of input image
        img_h: Int larger than 0. Pixel height of input image
        img_c: Int larger than 0. Pixel depth of input image
    
    Input block:
        One input layer
    Contraction block:
        Two convolution layers
        One pool layer for downsampling
    Bottleneck block:
        Two convolution layers
    Expansion block:
        One transposed convolution layer for upsampling
        One concatenate operation
        One convolution layer
    Output block:
        One convolution layer
    '''
    
    input_size = (img_w, img_h, img_c)
    input_layer = Input(shape=input_size, name='input_layer')
    
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(input_layer)
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(contraction_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(contraction_1)
    
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(pool_1)
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(contraction_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(contraction_2)
    
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(pool_2)
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(contraction_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(contraction_3)
    
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(pool_3)
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(contraction_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(contraction_4)
    
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(pool_4)
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(bottleneck)
    
    transpose_1 = Conv2DTranspose(filters=base*8, kernel_size=3, strides=2, padding='same')(bottleneck)
    expansion_1 = concatenate([transpose_1, contraction_4])
    expansion_1 = Conv2D(filters=base*8, kernel_size=3, padding="same", activation="relu")(expansion_1)
    
    transpose_2 = Conv2DTranspose(filters=base*4, kernel_size=3, strides=2, padding='same')(expansion_1)
    expansion_2 = concatenate([transpose_2, contraction_3])
    expansion_2 = Conv2D(filters=base*4, kernel_size=3, padding="same", activation="relu")(expansion_2)

    transpose_3 = Conv2DTranspose(filters=base*2, kernel_size=3, strides=2, padding='same')(expansion_2)
    expansion_3 = concatenate([transpose_3, contraction_2])
    expansion_3 = Conv2D(filters=base*2, kernel_size=3, padding="same", activation="relu")(expansion_3)

    transpose_4 = Conv2DTranspose(filters=base, kernel_size=3, strides=2, padding='same')(expansion_3)
    expansion_4 = concatenate([transpose_4, contraction_1])
    expansion_4 = Conv2D(filters=base, kernel_size=3, padding="same", activation="relu")(expansion_4)
    
    output_layer = Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(expansion_4)
    
    model = Model(input_layer, output_layer, name='Basic_UNET')
    model.summary()
    
    return model

#--------------------
# Binary UNET model with dropout layers

def get_unet_BI_DO(base, img_w, img_h, img_c, dropout_rate):
    
    '''
    UNET model with dropout layers for binary classification
    Parameters:
        base: Int larger than 0. Number of filters to start
        img_w: Int larger than 0. Pixel width of input image
        img_h: Int larger than 0. Pixel height of input image
        img_c: Int larger than 0. Pixel depth of input image
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop
    
    Input block:
        One input layer
    Contraction block:
        Two convolution layers
        One pool layer for downsampling
        One dropout layer
    Bottleneck block:
        Two convolution layers
    Expansion block:
        One transposed convolution layer for upsampling
        One concatenate operation
        One dropout layer
        One convolution layer
    Output block:
        One convolution layer
    '''
    
    input_size = (img_w, img_h, img_c)
    input_layer = Input(shape=input_size, name='input_layer')
    
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(input_layer)
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(contraction_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(contraction_1)
    pool_1 = Dropout(dropout_rate)(pool_1)
    
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(pool_1)
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(contraction_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(contraction_2)
    pool_2 = Dropout(dropout_rate)(pool_2)
    
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(pool_2)
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(contraction_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(contraction_3)
    pool_3 = Dropout(dropout_rate)(pool_3)
    
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(pool_3)
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(contraction_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(contraction_4)
    pool_4 = Dropout(dropout_rate)(pool_4)
    
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(pool_4)
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(bottleneck)
    
    transpose_1 = Conv2DTranspose(filters=base*8, kernel_size=3, strides=2, padding='same')(bottleneck)
    expansion_1 = concatenate([transpose_1, contraction_4])
    expansion_1 = Dropout(dropout_rate)(expansion_1)
    expansion_1 = Conv2D(filters=base*8, kernel_size=3, padding="same", activation="relu")(expansion_1)
    
    transpose_2 = Conv2DTranspose(filters=base*4, kernel_size=3, strides=2, padding='same')(expansion_1)
    expansion_2 = concatenate([transpose_2, contraction_3])
    expansion_2 = Dropout(dropout_rate)(expansion_2)
    expansion_2 = Conv2D(filters=base*4, kernel_size=3, padding="same", activation="relu")(expansion_2)

    transpose_3 = Conv2DTranspose(filters=base*2, kernel_size=3, strides=2, padding='same')(expansion_2)
    expansion_3 = concatenate([transpose_3, contraction_2])
    expansion_3 = Dropout(dropout_rate)(expansion_3)
    expansion_3 = Conv2D(filters=base*2, kernel_size=3, padding="same", activation="relu")(expansion_3)

    transpose_4 = Conv2DTranspose(filters=base, kernel_size=3, strides=2, padding='same')(expansion_3)
    expansion_4 = concatenate([transpose_4, contraction_1])
    expansion_4 = Dropout(dropout_rate)(expansion_4)
    expansion_4 = Conv2D(filters=base, kernel_size=3, padding="same", activation="relu")(expansion_4)
    
    output_layer = Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(expansion_4)
    
    model = Model(input_layer, output_layer, name='UNET_with_Dropout')
    model.summary()
    
    return model

#--------------------
# Binary UNET model with dropout layers and batch normalization

def get_unet_BI_DO_BN(base, img_w, img_h, img_c, dropout_rate):

    '''
    UNET model with dropout layers and batch normalization for binary classification
    Parameters:
        base: Int larger than 0. Number of filters to start
        img_w: Int larger than 0. Pixel width of input image
        img_h: Int larger than 0. Pixel height of input image
        img_c: Int larger than 0. Pixel depth of input image
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop
    
    Input block:
        One input layer
    Contraction block:
        Two convolution layers
        One pool layer for downsampling
        One dropout layer
    Bottleneck block:
        Two convolution layers
    Expansion block:
        One transposed convolution layer for upsampling
        One batch normalization layer
        One concatenate operation
        One dropout layer
        One convolution layer
    Output block:
        One convolution layer
    '''
    
    input_size = (img_w, img_h, img_c)
    input_layer = Input(shape=input_size, name='input_layer')
    
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(input_layer)
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(contraction_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(contraction_1)
    pool_1 = Dropout(dropout_rate)(pool_1)
    
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(pool_1)
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(contraction_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(contraction_2)
    pool_2 = Dropout(dropout_rate)(pool_2)
    
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(pool_2)
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(contraction_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(contraction_3)
    pool_3 = Dropout(dropout_rate)(pool_3)
    
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(pool_3)
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(contraction_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(contraction_4)
    pool_4 = Dropout(dropout_rate)(pool_4)
    
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(pool_4)
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(bottleneck)
    
    transpose_1 = Conv2DTranspose(filters=base*8, kernel_size=3, strides=2, padding='same')(bottleneck)
    transpose_1 = BatchNormalization()(transpose_1)
    expansion_1 = concatenate([transpose_1, contraction_4])
    expansion_1 = Dropout(dropout_rate)(expansion_1)
    expansion_1 = Conv2D(filters=base*8, kernel_size=3, padding="same", activation="relu")(expansion_1)
    
    transpose_2 = Conv2DTranspose(filters=base*4, kernel_size=3, strides=2, padding='same')(expansion_1)
    transpose_2 = BatchNormalization()(transpose_2)
    expansion_2 = concatenate([transpose_2, contraction_3])
    expansion_2 = Dropout(dropout_rate)(expansion_2)
    expansion_2 = Conv2D(filters=base*4, kernel_size=3, padding="same", activation="relu")(expansion_2)

    transpose_3 = Conv2DTranspose(filters=base*2, kernel_size=3, strides=2, padding='same')(expansion_2)
    transpose_3 = BatchNormalization()(transpose_3)
    expansion_3 = concatenate([transpose_3, contraction_2])
    expansion_3 = Dropout(dropout_rate)(expansion_3)
    expansion_3 = Conv2D(filters=base*2, kernel_size=3, padding="same", activation="relu")(expansion_3)

    transpose_4 = Conv2DTranspose(filters=base, kernel_size=3, strides=2, padding='same')(expansion_3)
    transpose_4 = BatchNormalization()(transpose_4)
    expansion_4 = concatenate([transpose_4, contraction_1])
    expansion_4 = Dropout(dropout_rate)(expansion_4)
    expansion_4 = Conv2D(filters=base, kernel_size=3, padding="same", activation="relu")(expansion_4)
    
    output_layer = Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(expansion_4)
    
    model = Model(input_layer, output_layer, name='UNET_with_Dropout_BatchNorm')
    model.summary()
    
    return model

#--------------------
# Basic multiclass UNET model

def get_unet_MC(base, img_w, img_h, img_c):
    
    '''
    Basic UNET model for multiclass classification
    Parameters:
        base: Int larger than 0. Number of filters to start
        img_w: Int larger than 0. Pixel width of input image
        img_h: Int larger than 0. Pixel height of input image
        img_c: Int larger than 0. Pixel depth of input image
    
    Input block:
        One input layer
    Contraction block:
        Two convolution layers
        One pool layer for downsampling
    Bottleneck block:
        Two convolution layers
    Expansion block:
        One transposed convolution layer for upsampling
        One concatenate operation
        One convolution layer
    Output block:
        One convolution layer
    '''
    
    input_size = (img_w, img_h, img_c)
    input_layer = Input(shape=input_size, name='input_layer')
    
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(input_layer)
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(contraction_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(contraction_1)
    
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(pool_1)
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(contraction_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(contraction_2)
    
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(pool_2)
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(contraction_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(contraction_3)
    
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(pool_3)
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(contraction_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(contraction_4)
    
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(pool_4)
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(bottleneck)
    
    transpose_1 = Conv2DTranspose(filters=base*8, kernel_size=3, strides=2, padding='same')(bottleneck)
    expansion_1 = concatenate([transpose_1, contraction_4])
    expansion_1 = Conv2D(filters=base*8, kernel_size=3, padding="same", activation="relu")(expansion_1)
    
    transpose_2 = Conv2DTranspose(filters=base*4, kernel_size=3, strides=2, padding='same')(expansion_1)
    expansion_2 = concatenate([transpose_2, contraction_3])
    expansion_2 = Conv2D(filters=base*4, kernel_size=3, padding="same", activation="relu")(expansion_2)

    transpose_3 = Conv2DTranspose(filters=base*2, kernel_size=3, strides=2, padding='same')(expansion_2)
    expansion_3 = concatenate([transpose_3, contraction_2])
    expansion_3 = Conv2D(filters=base*2, kernel_size=3, padding="same", activation="relu")(expansion_3)

    transpose_4 = Conv2DTranspose(filters=base, kernel_size=3, strides=2, padding='same')(expansion_3)
    expansion_4 = concatenate([transpose_4, contraction_1])
    expansion_4 = Conv2D(filters=base, kernel_size=3, padding="same", activation="relu")(expansion_4)
    
    output_layer = Conv2D(filters=3, kernel_size=1, padding='same', activation='softmax')(expansion_4)
    
    model = Model(input_layer, output_layer, name='Basic_UNET')
    model.summary()
    
    return model

#--------------------
# Multiclass UNET model with dropout layers

def get_unet_MC_DO(base, img_w, img_h, img_c, dropout_rate):
    
    '''
    UNET model with dropout layers for multiclass classification
    Parameters:
        base: Int larger than 0. Number of filters to start
        img_w: Int larger than 0. Pixel width of input image
        img_h: Int larger than 0. Pixel height of input image
        img_c: Int larger than 0. Pixel depth of input image
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop
    
    Input block:
        One input layer
    Contraction block:
        Two convolution layers
        One pool layer for downsampling
        One dropout layer
    Bottleneck block:
        Two convolution layers
    Expansion block:
        One transposed convolution layer for upsampling
        One concatenate operation
        One dropout layer
        One convolution layer
    Output block:
        One convolution layer
    '''
    
    input_size = (img_w, img_h, img_c)
    input_layer = Input(shape=input_size, name='input_layer')
    
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(input_layer)
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(contraction_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(contraction_1)
    pool_1 = Dropout(dropout_rate)(pool_1)
    
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(pool_1)
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(contraction_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(contraction_2)
    pool_2 = Dropout(dropout_rate)(pool_2)
    
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(pool_2)
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(contraction_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(contraction_3)
    pool_3 = Dropout(dropout_rate)(pool_3)
    
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(pool_3)
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(contraction_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(contraction_4)
    pool_4 = Dropout(dropout_rate)(pool_4)
    
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(pool_4)
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(bottleneck)
    
    transpose_1 = Conv2DTranspose(filters=base*8, kernel_size=3, strides=2, padding='same')(bottleneck)
    expansion_1 = concatenate([transpose_1, contraction_4])
    expansion_1 = Dropout(dropout_rate)(expansion_1)
    expansion_1 = Conv2D(filters=base*8, kernel_size=3, padding="same", activation="relu")(expansion_1)
    
    transpose_2 = Conv2DTranspose(filters=base*4, kernel_size=3, strides=2, padding='same')(expansion_1)
    expansion_2 = concatenate([transpose_2, contraction_3])
    expansion_2 = Dropout(dropout_rate)(expansion_2)
    expansion_2 = Conv2D(filters=base*4, kernel_size=3, padding="same", activation="relu")(expansion_2)

    transpose_3 = Conv2DTranspose(filters=base*2, kernel_size=3, strides=2, padding='same')(expansion_2)
    expansion_3 = concatenate([transpose_3, contraction_2])
    expansion_3 = Dropout(dropout_rate)(expansion_3)
    expansion_3 = Conv2D(filters=base*2, kernel_size=3, padding="same", activation="relu")(expansion_3)

    transpose_4 = Conv2DTranspose(filters=base, kernel_size=3, strides=2, padding='same')(expansion_3)
    expansion_4 = concatenate([transpose_4, contraction_1])
    expansion_4 = Dropout(dropout_rate)(expansion_4)
    expansion_4 = Conv2D(filters=base, kernel_size=3, padding="same", activation="relu")(expansion_4)
    
    output_layer = Conv2D(filters=3, kernel_size=1, padding='same', activation='softmax')(expansion_4)
    
    model = Model(input_layer, output_layer, name='UNET_with_Dropout')
    model.summary()
    
    return model
#--------------------
# Multiclass UNET model with dropout layers and batch normalization

def get_unet_MC_DO_BN(base, img_w, img_h, img_c, dropout_rate):

    '''
    UNET model with dropout layers and batch normalization for multiclass classification
    Parameters:
        base: Int larger than 0. Number of filters to start
        img_w: Int larger than 0. Pixel width of input image
        img_h: Int larger than 0. Pixel height of input image
        img_c: Int larger than 0. Pixel depth of input image
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop
    
    Input block:
        One input layer
    Contraction block:
        Two convolution layers
        One pool layer for downsampling
        One dropout layer
    Bottleneck block:
        Two convolution layers
    Expansion block:
        One transposed convolution layer for upsampling
        One batch normalization layer
        One concatenate operation
        One dropout layer
        One convolution layer
    Output block:
        One convolution layer
    '''
    
    input_size = (img_w, img_h, img_c)
    input_layer = Input(shape=input_size, name='input_layer')
    
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(input_layer)
    contraction_1 = Conv2D(filters=base, kernel_size=3, padding='same', activation='relu')(contraction_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(contraction_1)
    pool_1 = Dropout(dropout_rate)(pool_1)
    
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(pool_1)
    contraction_2 = Conv2D(filters=base*2, kernel_size=3, padding='same', activation='relu')(contraction_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(contraction_2)
    pool_2 = Dropout(dropout_rate)(pool_2)
    
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(pool_2)
    contraction_3 = Conv2D(filters=base*4, kernel_size=3, padding='same', activation='relu')(contraction_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(contraction_3)
    pool_3 = Dropout(dropout_rate)(pool_3)
    
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(pool_3)
    contraction_4 = Conv2D(filters=base*8, kernel_size=3, padding='same', activation='relu')(contraction_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(contraction_4)
    pool_4 = Dropout(dropout_rate)(pool_4)
    
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(pool_4)
    bottleneck = Conv2D(filters=base*16, kernel_size=3, padding='same', activation='relu')(bottleneck)
    
    transpose_1 = Conv2DTranspose(filters=base*8, kernel_size=3, strides=2, padding='same')(bottleneck)
    transpose_1 = BatchNormalization(axis=3)(transpose_1)
    expansion_1 = concatenate([transpose_1, contraction_4])
    expansion_1 = Dropout(dropout_rate)(expansion_1)
    expansion_1 = Conv2D(filters=base*8, kernel_size=3, padding="same", activation="relu")(expansion_1)
    
    transpose_2 = Conv2DTranspose(filters=base*4, kernel_size=3, strides=2, padding='same')(expansion_1)
    transpose_2 = BatchNormalization(axis=3)(transpose_2)
    expansion_2 = concatenate([transpose_2, contraction_3])
    expansion_2 = Dropout(dropout_rate)(expansion_2)
    expansion_2 = Conv2D(filters=base*4, kernel_size=3, padding="same", activation="relu")(expansion_2)

    transpose_3 = Conv2DTranspose(filters=base*2, kernel_size=3, strides=2, padding='same')(expansion_2)
    transpose_3 = BatchNormalization(axis=3)(transpose_3)
    expansion_3 = concatenate([transpose_3, contraction_2])
    expansion_3 = Dropout(dropout_rate)(expansion_3)
    expansion_3 = Conv2D(filters=base*2, kernel_size=3, padding="same", activation="relu")(expansion_3)

    transpose_4 = Conv2DTranspose(filters=base, kernel_size=3, strides=2, padding='same')(expansion_3)
    transpose_4 = BatchNormalization(axis=3)(transpose_4)
    expansion_4 = concatenate([transpose_4, contraction_1])
    expansion_4 = Dropout(dropout_rate)(expansion_4)
    expansion_4 = Conv2D(filters=base, kernel_size=3, padding="same", activation="relu")(expansion_4)
    
    output_layer = Conv2D(filters=3, kernel_size=1, padding='same', activation='softmax')(expansion_4)
    
    model = Model(input_layer, output_layer, name='Multiclass_UNET_with_Dropout_BatchNorm')
    model.summary()
    
    return model