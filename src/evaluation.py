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
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'ACC': ACC,
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
        # 'MCC_3': MCC_3,
        # 'ERROR': ERROR,
        # 'TOTAL_TARGETS': TOTAL_TARGETS,
        # 'TOTAL_PREDICTIONS': TOTAL_PREDICTIONS,
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
    plt.yticks(rotation=0)
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set(rc={'figure.figsize': (16, 9)})
    sns.set(font_scale=2.4)

    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    return ax


def plot_mcc_split_label(df_mcc_values, palette=None):
    sns.set(font_scale=1.6)
    plt.rcParams["patch.force_edgecolor"] = False

    barplot = sns.barplot(
        x='Label',
        y='MCC',
        hue='Model',
        data=df_mcc_values,
        edgecolor=None,
        palette=palette
        )
    
    n_bars = len(df_mcc_values['Label'].unique())
    n_hues = len(df_mcc_values['Model'].unique())
    group_width = min(0.8, n_hues/(n_hues + 1.5))
    xvals = np.arange(n_bars)
    for i in range(n_hues):
        hue_xvals = xvals - (group_width/2.) + (i + 0.5) * group_width / n_hues
        barplot.errorbar(
            hue_xvals,
            df_mcc_values[df_mcc_values['Model'] == df_mcc_values['Model'].unique()[i]]['MCC'],
            yerr=df_mcc_values['Error'][i::n_hues],
            fmt='none',
            c='k'
        )

    barplot.set_title('MCC Scores for Different Models')

    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set(rc={'figure.figsize': (16, 9)})

    handles, labels = barplot.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3, columnspacing=0.7, handletextpad=0.5, fontsize='x-large')

    barplot.set(ylim=(0.5, None))
    barplot.set_yticks(np.arange(0.5, barplot.get_ylim()[1], 0.05))
    return barplot


def plot_mcc_split_label_kingdom_facet(df_mcc_values, Title="Barplot of MCC Scores for Different Models", ax=None, errors=None):
    sns.set(font_scale=1.6)
    plt.rcParams["patch.force_edgecolor"] = False

    barplot = sns.barplot(
        x='Type',
        y='MCC',
        hue='Model',
        data=df_mcc_values,
        edgecolor=None,
        ax=ax
    )

    barplot.set_title(Title)  # Set the title to the provided Title
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set(rc={'figure.figsize': (16, 9)})

    handles, labels = barplot.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3, columnspacing=0.7, handletextpad=0.5, fontsize='x-large')

    barplot.set(ylim=(0.5, None))
    barplot.set_yticks(np.arange(0.5, barplot.get_ylim()[1], 0.05))
    return barplot


def plot_mcc_split_label_kingdom_facet_simple(df_mcc_values, Title="Barplot of MCC Scores for Different Models", ax=None, palette=None, errors=None):
    sns.set(font_scale=1.6)
    plt.rcParams["patch.force_edgecolor"] = False

    barplot = sns.barplot(
        x='Simple_Type',
        y='MCC',
        hue='Model',
        data=df_mcc_values,
        edgecolor=None,
        ax=ax,
        palette=palette,
        )

    barplot.set_title(Title)  # Set the title to the provided Title
    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    sns.set(rc={'figure.figsize': (16, 9)})
    
    if errors is not None:
        n_bars = len(df_mcc_values['Type'].unique())
        n_hues = len(df_mcc_values['Model'].unique())
        group_width = min(0.8, n_hues/(n_hues + 1.5))
        xvals = np.arange(n_bars)
        for i in range(n_hues):
            hue_xvals = xvals - (group_width/2.) + (i + 0.5) * group_width / n_hues
            barplot.errorbar(
                hue_xvals,
                df_mcc_values[df_mcc_values['Model'] == df_mcc_values['Model'].unique()[i]]['MCC'],
                yerr=errors[i::n_hues],
                fmt='none',
                c='k'
            )

    barplot.set(xlabel=None)  # Remove the x-label

    barplot.legend_.remove()
    # handles, labels = barplot.get_legend_handles_labels()
    # plt.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3, columnspacing=0.7, handletextpad=0.5, fontsize='x-large')

    barplot.set(ylim=(0.3, None))
    barplot.set_yticks(np.arange(0.3, barplot.get_ylim()[1], 0.1))
    return barplot


def evaluate_mcc(targets, predictions, labels):
    p = {}
    for x in labels:
        target = [1 if y == x else 0 for y in targets]
        prediction = [1 if y == x else 0 for y in predictions]
        # print(target)
        # print(prediction)
        # if target == prediction:
        #     p.update({x: 1})
        # else:
        error = 0
        p.update({x: [matthews_corrcoef(target, prediction), error]})  # noqa: E999

    return p