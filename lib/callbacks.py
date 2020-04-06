from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def getCallbacks(name):
    return [EarlyStopping(patience=5, verbose=1),
      ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.1, verbose=1),
      ModelCheckpoint(name, verbose=1, save_best_only=True, save_weights_only=True)]