import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize, rotate

import configparser
config = configparser.ConfigParser()
config.read('config.ini')
croppedImage = int(config['Image']['croppedImageSize'])
resizedImage = int(config['Image']['resizedImageSize'])
cacheImagePath = config['Image']['cacheImagePath']
cacheMaskPath = config['Image']['cacheMaskPath']
actualImagePath = config['Image']['actualImagePath']
actualMaskPath = config['Image']['actualMaskPath']


def crop_img(oimg, name, isImg):
   imageParts = resizedImage/croppedImage
   for z in range(1, int(imageParts+1)):
    for x in range(1, int(imageParts+1)):
      img = oimg[(x*croppedImage)-croppedImage: x*croppedImage, (z*croppedImage)-croppedImage: z*croppedImage]
      folder = ''
      if(isImg):
        folder = cacheImagePath
      else: 
        folder = cacheMaskPath
      im = array_to_img(img/255.0)
      im.save(folder+name+ "_"+str(x)+"-"+str(z)+".png")

def prepareData():
    if os.path.exists(cacheImagePath):
      print('\nDELETE THE CACHE FOLDER!\n')
      return
    else:
      os.makedirs(cacheImagePath)
      os.makedirs(cacheMaskPath)
    ids = next(os.walk(actualImagePath))[2]
    for id_ in ids:
        iid = id_.replace('.png', '')
        # Load images
        img = load_img(actualImagePath+id_, color_mode="rgb")
        x_img = img_to_array(img)
        x_img = resize(x_img, (resizedImage, resizedImage, 3), mode = 'constant', preserve_range = True)
        crop_img(x_img, iid, True)

        x_img = rotate(x_img, 90)
        crop_img(x_img, iid+"90deg", True)

        x_img = rotate(x_img, 180)
        crop_img(x_img, iid+"180deg", True)

        x_img = rotate(x_img, 270)
        crop_img(x_img, iid+"270deg", True)

        # Load masks
        mask = img_to_array(load_img(actualMaskPath+iid+"_bin_mask.png", color_mode="grayscale"))

        mask = resize(mask, (resizedImage, resizedImage, 1), mode = 'constant', preserve_range = True)
        crop_img(mask,iid, False)
        
        mask = rotate(mask, 90)
        crop_img(mask, iid+"90deg", False)

        mask = rotate(mask, 180)
        crop_img(mask, iid+"180deg", False)

        mask = rotate(mask, 270)
        crop_img(mask, iid+"270deg", False)
        

prepareData()