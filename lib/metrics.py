from keras import backend as K
from tensorflow.python.keras.losses import binary_crossentropy
# Metrics and Losses
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    res = dice_coef(y_true, y_pred)
    return -res

def hybrid_loss(y_true, y_pred):
    res = dice_coef(y_true, y_pred)
    return (0.3*res) + (0.7*binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0))

def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

allMetrics ={'get_f1' :get_f1, 
            'iou_coef' :iou_coef, 
            'dice_coef' :dice_coef, 
            'dice_coef_loss' :dice_coef_loss}