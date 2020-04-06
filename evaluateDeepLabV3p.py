from tensorflow.python.keras.models import load_model
from lib.datahelper import loadDataFromCache
from lib.metrics import get_f1, iou_coef, dice_coef, dice_coef_loss, allMetrics
from models import deepLab

import configparser
config = configparser.ConfigParser()
config.read('config.ini')
croppedImageSize = int(config['Image']['croppedImageSize'])

X_train, X_valid, y_train, y_valid = loadDataFromCache()

deeplab_model = deepLab.Deeplabv3((croppedImageSize,croppedImageSize,1), classes=1, OS=16)
deeplab_model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy", get_f1, iou_coef, dice_coef])
deeplab_model.load_weights('./trainedModels/model-deepLabv3p.h5')

loss, acc, f1, iou_coef, dice = deeplab_model.evaluate(X_valid, y_valid, verbose=1)

print("Loss: ", loss )
print("Accuracy: ",  acc)
print("F1: ", f1 )
print("IOU: ", iou_coef )
print("DICE: ", dice )