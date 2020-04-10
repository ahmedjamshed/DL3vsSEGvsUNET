import matplotlib.pyplot as plt
import numpy as np
import random
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

def plot_sample(X, y, preds, binary_preds, name, ix=None):
    
    """Function to plot the results"""
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    fig.suptitle(name, fontsize=14)
    ax[0].imshow(X[ix, ..., 0], cmap='Accent')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('base')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('cell')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('cell Predicted')
    
    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[3].set_title('cell Predicted binary')
    plt.savefig('graphs/'+name+"_res"+str(ix)+'.png')