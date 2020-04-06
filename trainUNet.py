from models import uNet
from tensorflow.python.keras.layers import Input
from lib.callbacks import getCallbacks
from lib.datahelper import loadDataFromCache
from lib.plotting import plotGraph
from lib.metrics import get_f1, iou_coef, dice_coef, dice_coef_loss

import configparser
config = configparser.ConfigParser()
config.read('config.ini')
croppedImageSize = int(config['Image']['croppedImageSize'])

X_train, X_valid, y_train, y_valid = loadDataFromCache()

input_img = Input((croppedImageSize, croppedImageSize, 1), name='imga')
uNetModel = uNet.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
uNetModel.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy", get_f1, iou_coef, dice_coef])

resultsuNet = uNetModel.fit(X_train, y_train, batch_size=64, epochs=10, callbacks=getCallbacks('./trainedModels/model-uNet.h5'),\
                    validation_data=(X_valid, y_valid))

plotGraph('UNET', resultsuNet)