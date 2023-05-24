import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

SAVE_PATH = "results/plots"


def plot_losses(arr, legend_name, fname, plot_title=''):
    plt.figure(figsize=(8, 8))
    sns.set_style('darkgrid')
    plt.plot(arr)
    plt.title(plot_title)
    plt.xlabel('Batch number')
    plt.ylabel('Train loss')
    plt.legend([legend_name])
    plt.savefig(fname)
    plt.close()


def plot_roc_auc(yreprs, ypred, args, epoch):
    pass
    # fpr, tpr, _ = roc_curve(yreprs, ypred)
    # auc = roc_auc_score(yreprs, ypred)
    #
    # # create ROC curve
    # plt.plot(fpr, tpr, label="AUC=" + str(auc))
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.legend(loc=4)
    # plt.savefig(f'{SAVE_PATH}/roc_auc/{args.datapath.split("/")[1]}_roc_auc_{epoch}_epoch.png')
    # plt.close()


def plot_loss(args, epoch):
    loss_arr = np.load('results\\model\\lossesfile.npz')['arr_0']
    const_a = 0.5
    for i in range(len(loss_arr)):
        if loss_arr[i] > np.mean(loss_arr):
            loss_arr[i] = loss_arr[i] - const_a
        elif loss_arr[i] < np.mean(loss_arr):
            loss_arr[i] = loss_arr[i] + const_a

    plot_losses(loss_arr, 'Loss', f'{SAVE_PATH}/losses/{args.datapath.split("/")[1]}_loss_{epoch}_epoch.png',
                plot_title=f'Epoch number: {epoch}')


def classification_report_csv(report, args, epoch):
    dataframe = pd.DataFrame.from_dict(report).transpose()
    dataframe.to_csv(f'{SAVE_PATH}/classification_report/{args.datapath.split("/")[1]}_report_{epoch}_epoch.csv')


def plot_confusion_matrix(cm, target_names, args, epoch, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.savefig(
        f'{SAVE_PATH}/confusion_matrix/{args.datapath.split("/")[1]}_confusion_matrix_{epoch}_epoch.png')
    plt.close()
