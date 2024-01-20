import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from collections import deque

from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef


def encode_sequences(sequences, encoding):
    return [encode_sequence(x, encoding) for x in sequences]


def encode_sequence(sequence, encoding):
    return [encoding[x] for x in sequence]


def evaluate(targets, predictions, labels=None, encoding=None, decoding=None, mask=None):
    if not labels:
        labels = list(encoding.values())

    CM = confusion_matrix(
        y_true=targets,
        y_pred=predictions,
        labels=labels,
    )
    FP = CM.sum(axis=0) - np.diag(CM)
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)

    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    FOR = FN/(TN+FN)

    ACC = (TP+TN)/(TP+FP+FN+TN)
    F1 = 2*(PPV*TPR)/(PPV+TPR)

    MCC_1 = matthews_corrcoef(targets, predictions)
    MCC_2 = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    MCC_3 = np.sqrt(PPV*TPR*TNR*NPV) - np.sqrt(FDR*FNR*FPR*FOR)
    
    ERROR = 0
    TOTAL_TARGETS = len(targets)
    TOTAL_PREDICTIONS = len(predictions)
    
    values_dict = {
        'CM': CM,
        'ACC': ACC,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TPR': TPR,
        'TNR': TNR,
        'PPV': PPV,
        'FPR': FPR,
        'FNR': FNR,
        'NPV': NPV,
        'FDR': FDR,
        'F1': F1,
        'MCC_1': MCC_1,
        'MCC_2': MCC_2,
        'MCC_3': MCC_3,
        'ERROR': ERROR,
        'TOTAL_TARGETS': TOTAL_TARGETS,
        'TOTAL_PREDICTIONS': TOTAL_PREDICTIONS,
    }
    return values_dict


def plot_mcc(data_mcc):
    ax = sns.barplot(data=data_mcc, errorbar="se", edgecolor="black")
    ax.set_title('MCC Error')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Error')
    
    return ax


def plot_confusion_matrix(data_cm, labels, title='Confusion Matrix'):
    ax = sns.heatmap(
        data_cm,
        annot=True,
        # xticklabels=[decoding[label] for label in range(len(decoding))],
        xticklabels=labels,
        yticklabels=labels,
        fmt='d',
    )
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set(rc={'figure.figsize': (16, 9)})
    sns.set(font_scale=2.4)
    plt.yticks(rotation=0)

    ax.set_title(title)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

    return ax