from tensorflow.python.keras.models import load_model
from lib.datahelper import loadDataFromCache
from lib.metrics import get_f1, iou_coef, dice_coef, dice_coef_loss, allMetrics

from models import uNet
from tensorflow.python.keras.layers import Input
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
croppedImageSize = int(config['Image']['croppedImageSize'])

X_train, X_valid, y_train, y_valid = loadDataFromCache()

input_img = Input((croppedImageSize, croppedImageSize, 1), name='imga')
uNetModel = uNet.get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
uNetModel.compile(optimizer='adam', loss=dice_coef_loss, metrics=["accuracy", get_f1, iou_coef, dice_coef])
uNetModel.load_weights('./trainedModels/model-uNet.h5')


loss, acc, f1, iou_coef, dice = uNetModel.evaluate(X_valid, y_valid, verbose=1)

print("Loss: ", loss )
print("Accuracy: ",  acc)
print("F1: ", f1 )
print("IOU: ", iou_coef )
print("DICE: ", dice )