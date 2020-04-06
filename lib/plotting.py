import matplotlib.pyplot as plt
import numpy as np
plt.style.use("ggplot")

def plotGraph(name, result):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve "+name)
    plt.plot(result.history["loss"], label="loss")
    plt.plot(result.history["val_accuracy"], label="accuracy")
    plt.plot(result.history["val_loss"], label="val_loss")
    plt.plot(result.history["val_get_f1"], label="F1_accuracy")
    plt.plot(result.history["val_iou_coef"], label="IOU_accuracy")
    plt.plot( np.argmin(result.history["val_loss"]), np.min(result.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig('graphs/'+name+'Training.png')