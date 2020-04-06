from models import deepLab
from lib.callbacks import getCallbacks
from lib.datahelper import loadDataFromCache
from lib.plotting import plotGraph
from lib.metrics import get_f1, iou_coef, dice_coef, dice_coef_loss

import configparser
config = configparser.ConfigParser()
config.read('config.ini')
croppedImageSize = int(config['Image']['croppedImageSize'])

X_train, X_valid, y_train, y_valid = loadDataFromCache()

deeplab_model = deepLab.Deeplabv3((croppedImageSize,croppedImageSize,1), classes=1, OS=16)
deeplab_model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy", get_f1, iou_coef, dice_coef])

resultsdlNet = deeplab_model.fit(X_train, y_train, batch_size=8, epochs=10,  callbacks=getCallbacks('./trainedModels/model-deepLabv3p.h5'),\
                    validation_data=(X_valid, y_valid))

plotGraph('DeepLabv3p', resultsdlNet)