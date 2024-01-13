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


def evaluate(targets, predictions, encoding, decoding, mask=None):
    result = deque()
    # for targets, predictions in zip(predictions, targets):
    CM = confusion_matrix(
        y_true=targets,
        y_pred=predictions,
        labels=list(encoding.values()),
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
    
    values_dict = {
        'CM': CM,
        'FP': FP,
        'FN': FN,
        'TP': TP,
        'TN': TN,
        'TPR': TPR,
        'TNR': TNR,
        'PPV': PPV,
        'NPV': NPV,
        'FPR': FPR,
        'FNR': FNR,
        'FDR': FDR,
        'ACC': ACC,
        'F1': F1,
        'MCC_1': MCC_1,
        'MCC_2': MCC_2,
        'MCC_3': MCC_3
    }
    
    result.append(values_dict)

        # print('CM', CM)
        # print('FP', FP)
        # print('FN', FN)
        # print('TP', TP)
        # print('TN', TN)

        # print('TPR', TPR, np.nanmean(TPR))
        # print('TNR', TNR, np.nanmean(TNR))
        # print('PPV', PPV, np.nanmean(PPV))
        # print('NPV', NPV, np.nanmean(NPV))
        # print('FPR', FPR, np.nanmean(FPR))
        # print('FNR', FNR, np.nanmean(FNR))
        # print('FDR', FDR, np.nanmean(FDR))

        # print('ACC', ACC, np.nanmean(ACC))
        # print('F1', F1, np.nanmean(F1))

        # print('MCC_1', MCC_1)
        # print('MCC_2', MCC_2, np.nanmean(MCC_2))
        # print('MCC_3', MCC_3, np.nanmean(MCC_3))
    return list(result)


def plot_mcc(data_mcc):
    ax = sns.barplot(data=data_mcc, errorbar="se", edgecolor="black")
    ax.set_title('MCC Error')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Error')
    
    return ax


def plot_confusion_matrix(data_cm, decoding):

    ax = sns.heatmap(
        data_cm,
        annot=True,
        xticklabels=[decoding[label] for label in range(len(decoding))],
        yticklabels=[decoding[label] for label in range(len(decoding))],
        fmt='d'
    )

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

    return ax