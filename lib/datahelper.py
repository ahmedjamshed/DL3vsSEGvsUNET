import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize

import configparser
config = configparser.ConfigParser()
config.read('config.ini')
croppedImageSize = int(config['Image']['croppedImageSize'])
resizedImage = int(config['Image']['resizedImageSize'])
cacheImagePath = config['Image']['cacheImagePath']
cacheMaskPath = config['Image']['cacheMaskPath']
actualImagePath = config['Image']['actualImagePath']
actualMaskPath = config['Image']['actualMaskPath']
testImagePath = config['Image']['testImagePath']
testMaskPath = config['Image']['testMaskPath']



def loadDataFromCache(): 
    ids = next(os.walk(cacheImagePath))[2] # list of names all images in the given path
    imgs = np.zeros((len(ids), croppedImageSize, croppedImageSize, 1), dtype=np.float32)
    gts = np.zeros((len(ids), croppedImageSize, croppedImageSize, 1), dtype=np.float32)
    for n, id_ in enumerate(ids):
        # Load images
        x_img = img_to_array(load_img(cacheImagePath+id_, color_mode="grayscale"))
        # Load masks
        mask = img_to_array(load_img(cacheMaskPath+id_, color_mode="grayscale"))
        # # Save images
        imgs[n] = x_img/255.0
        gts[n] = mask/255.0
    return train_test_split(imgs, gts, test_size=0.3, random_state=42)

def loadTestData(): 
    ids = next(os.walk(testImagePath))[2] # list of names all images in the given path
    imgs = np.zeros((len(ids), resizedImage, resizedImage, 1), dtype=np.float32)
    gts = np.zeros((len(ids), resizedImage, resizedImage, 1), dtype=np.float32)
    for n, id_ in enumerate(ids):
        iid = id_.replace('.png', '').replace('.tif', '')
        # Load images
        x_img = img_to_array(load_img(testImagePath+id_, color_mode="grayscale"))
        x_img = resize(x_img, (resizedImage, resizedImage, 1), mode = 'constant', preserve_range = True)
        # Load masks
        mask = img_to_array(load_img(testMaskPath+iid+"_bin_mask.png", color_mode="grayscale"))
        mask = resize(mask, (resizedImage, resizedImage, 1), mode = 'constant', preserve_range = True)
        # # Save images
        imgs[n] = x_img/255.0
        gts[n] = mask/255.0
    return [imgs, gts]