from models import segNet
from lib.callbacks import getCallbacks
from lib.datahelper import loadDataFromCache
from lib.plotting import plotGraph
from lib.metrics import get_f1, iou_coef, dice_coef, dice_coef_loss

X_train, X_valid, y_train, y_valid = loadDataFromCache()

segNetModel = segNet.SegNet()
segNetModel.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy", get_f1, iou_coef, dice_coef])

resultsSegNet = segNetModel.fit(X_train, y_train, batch_size=64, epochs=30, callbacks=getCallbacks('./trainedModels/model-segNet.h5'),\
                    validation_data=(X_valid, y_valid))

plotGraph('SegNet', resultsSegNet)