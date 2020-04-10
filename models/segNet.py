from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, UpSampling2D
from tensorflow.python.keras.layers.core import Lambda, RepeatVector, Reshape
from tensorflow.python.keras.layers.convolutional import Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from tensorflow.python.keras.layers.merge import concatenate, add

import configparser
config = configparser.ConfigParser()
config.read('config.ini')
croppedImageSize = int(config['Image']['croppedImageSize'])

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True, secondLayer=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if secondLayer:
      # second layer
      x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
      if batchnorm:
        x = BatchNormalization()(x)
      x = Activation('relu')(x)
    
    return x

    

def SegNet(classes=1, dropout=0.4):
    input_img = Input((croppedImageSize, croppedImageSize, 1), name='img')
    # Encoder
    x = conv2d_block(input_img, 64, kernel_size = 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = conv2d_block(x, 128, kernel_size = 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = conv2d_block(x, 256, kernel_size = 3)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = conv2d_block(x, 512, kernel_size = 3)
    x = Dropout(dropout)(x)
    # Decoder
    x = conv2d_block(x, 512, kernel_size = 3)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = conv2d_block(x, 256, kernel_size = 3)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = conv2d_block(x, 128, kernel_size = 3)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = conv2d_block(x, 64, kernel_size = 3)
    x = Dropout(dropout)(x)
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=[input_img],  outputs=[x])
    return model