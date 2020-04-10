from tensorflow.python.keras.models import load_model
from lib.datahelper import loadDataFromCache
from lib.metrics import get_f1, iou_coef, dice_coef, dice_coef_loss, allMetrics
from models import deepLab
from lib.plotting import plot_sample
import numpy as np
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

print("Results Saving IN Graphs Folder") 

modelNameFT =  "DEEPLABV3PNET_Train"
modelNameFV =  "DEEPLABV3PNET_Validation"

preds_train = deeplab_model.predict(X_train, verbose=1)
preds_val = deeplab_model.predict(X_valid, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

plot_sample(X_train, y_train, preds_train, preds_train_t, modelNameFT, ix=14)
plot_sample(X_valid, y_valid, preds_val, preds_val_t, modelNameFV, ix=19)
plot_sample(X_train, y_train, preds_train, preds_train_t, modelNameFT)
plot_sample(X_valid, y_valid, preds_val, preds_val_t, modelNameFV)
plot_sample(X_train, y_train, preds_train, preds_train_t, modelNameFT)
plot_sample(X_valid, y_valid, preds_val, preds_val_t, modelNameFV)
plot_sample(X_train, y_train, preds_train, preds_train_t, modelNameFT)
plot_sample(X_valid, y_valid, preds_val, preds_val_t, modelNameFV)
plot_sample(X_train, y_train, preds_train, preds_train_t, modelNameFT)
plot_sample(X_valid, y_valid, preds_val, preds_val_t, modelNameFV)
plot_sample(X_train, y_train, preds_train, preds_train_t, modelNameFT)
plot_sample(X_valid, y_valid, preds_val, preds_val_t, modelNameFV)
plot_sample(X_train, y_train, preds_train, preds_train_t, modelNameFT)
plot_sample(X_valid, y_valid, preds_val, preds_val_t, modelNameFV)
plot_sample(X_train, y_train, preds_train, preds_train_t, modelNameFT)
plot_sample(X_valid, y_valid, preds_val, preds_val_t, modelNameFV)