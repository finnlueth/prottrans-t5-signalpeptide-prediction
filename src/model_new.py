import gc
import time

import seaborn as sns

from transformers import (
    T5EncoderModel,
    T5PreTrainedModel,
    T5Tokenizer,
    T5Config,
    modeling_outputs,
)

from torch.nn import (
    CrossEntropyLoss,
    MSELoss,
    BCEWithLogitsLoss
)

import evaluate

from torchcrf import CRF

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sklearn.metrics

from datasets import Dataset, DatasetDict

import src.config


class T5EncoderModelForTokenClassification(T5EncoderModel):
    def __init__(
        self,
        config: T5Config,
        custom_num_labels,
        custom_dropout_rate,
        use_crf=False,
    ):
        super().__init__(config)
        self.custom_num_labels = custom_num_labels
        self.custom_dropout_rate = custom_dropout_rate
        self.use_crf = use_crf

        self.custom_dropout = nn.Dropout(self.custom_dropout_rate)
        self.custom_classifier = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.custom_num_labels
        )

        if self.use_crf:
            self.crf = CRF(
                num_tags=self.custom_num_labels,
                batch_first=True
            )

    # Adapted from https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/bert/modeling_bert.py#L1716
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs['last_hidden_state']

        # if labels is not None:
        sequence_output = self.custom_dropout(sequence_output)
        logits = self.custom_classifier(sequence_output)
        # print(logits.min(), logits.max())
        # print(torch.isnan(sequence_output).any())

        loss = None
        decoded_tags = None
        if self.use_crf:
            if labels is not None:
                logits_re = logits[:, :-1, :]
                labels_re = labels[:, :-1]
                attention_mask_re = attention_mask[:, :-1]
                # print('logits', logits_re.shape, logits_re.dtype, logits_re)
                print('logits', logits_re.shape, logits_re.dtype, logits_re.min(), logits_re.max())
                # print('labels', labels_re.shape, labels_re.dtype, labels_re)
                print('labels', labels_re.shape, labels_re.dtype, labels_re.min(), labels_re.max())
                # print('attention_mask', attention_mask_re.shape, attention_mask_re.dtype, attention_mask_re)
                log_likelihood = self.crf(
                    emissions=logits_re,
                    tags=labels_re,
                    mask=attention_mask_re.type(torch.uint8)
                )
                loss = -log_likelihood
                # print('neglog_likelihood', loss)
            else:
                decoded_tags = self.crf.modules_to_save.default.decode(logits, mask=attention_mask.type(torch.uint8))
        else:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                labels = labels.to(self.device)

                # print('labels', labels.shape, labels)
                # print('logits', logits.view(-1, self.custom_num_labels))
                # print('labels', labels.view(-1))

                loss = loss_fct(logits.view(-1, self.custom_num_labels), labels.view(-1))
                if loss > 2:
                    print('logits', logits.view(-1, self.custom_num_labels))
                    print('labels', labels.view(-1))
                    print(loss)

        # print('decoded_tags', decoded_tags, type(decoded_tags))
        # print('logits', logits, logits.shape)
        # print('loss', loss)
        # print('logic', logits if not self.use_crf else decoded_tags)

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits if decoded_tags is None else decoded_tags,
            # logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Adapted from https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/t5/modeling_t5.py#L775
class T5EncoderModelForSequenceClassification(T5EncoderModel):
    def __init__(
        self,
        config: T5Config,
        custom_num_labels,
        custom_dropout_rate,
    ):
        super().__init__(config)
        self.custom_num_labels = custom_num_labels
        self.custom_dropout_rate = custom_dropout_rate

        self.custom_dropout = nn.Dropout(self.custom_dropout_rate)

        self.custom_classifier_in = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.config.hidden_size
        )
        self.custom_classifier_out = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.custom_num_labels
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # position_ids=None,
        # head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print('-------------')
        # print('encoder_outputs', encoder_outputs.hidden_states)
        sequence_output = encoder_outputs['last_hidden_state']
        # sequence_output = self.custom_dropout(sequence_output)

        # print(mean_sequence_output.shape)
        # print(mean_sequence_output)

        # print('sequence_output', sequence_output[:, 0, :].shape, sequence_output[:, 0, :])
        # print()
        # print('sequence_output', sequence_output.shape, sequence_output)

        mean_sequence_output = torch.mean(sequence_output, dim=1)
        # print('mean_sequence_output', mean_sequence_output)
        logits = self.custom_dropout(mean_sequence_output)
        # print('custom_dropout', logits)
        logits = self.custom_classifier_in(logits)
        # print('custom_classifier_in', logits)
        logits = torch.tanh(logits)
        # print('tanh', logits)
        logits = self.custom_dropout(logits)
        # print('custom_dropout', logits)
        logits = self.custom_classifier_out(logits)
        # print('custom_classifier_out', logits)

        # logits = self.custom_classifier(mean_sequence_output)

        # print(logits.shape)
        # print(logits)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.custom_num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


###################################################
# Helper Functions
###################################################


def df_to_dataset(tokenizer: T5Tokenizer, sequences: list, labels: list, encoder: dict) -> Dataset:
    tokenized_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    dataset = Dataset.from_dict(tokenized_sequences)
    dataset = dataset.add_column("labels", [encoder[x] for x in labels], new_fingerprint=None)
    return dataset


def create_datasets(splits: dict, tokenizer: T5Tokenizer, data: pd.DataFrame, annotations_name: str, dataset_size: int, encoder: dict, sequence_type=None) -> DatasetDict:
    """
    Creates datasets for different splits based on the given parameters.

    Args:
        splits (dict): A dictionary where keys are split names (e.g., 'train', 'test') and values are lists of partition numbers.
        tokenizer (T5Tokenizer): An instance of the T5Tokenizer for tokenizing sequences.
        data (pd.DataFrame): A pandas DataFrame containing the data to be split into datasets.
        annotations_name (str): The name of the column in 'data' that contains annotations ('Label' or 'Type').
        dataset_size (int): The number of examples per partition to include in each split. If None, include all examples.
        encoder (dict): A dictionary mapping annotations to their encoded form.
        sequence_type (Optional[str]): The type of sequence to filter from 'data'. If None, use all data.

    Returns:
        DatasetDict: A dictionary where keys are split names and values are corresponding 'Dataset' objects with tokenized sequences and labels.

    Description:
        This function processes and tokenizes data for different splits (e.g., 'train', 'test') based on specified parameters. It filters data by sequence type, samples a specified number of examples per partition, tokenizes sequences, and adds labels. The resulting tokenized data and labels are returned as 'Dataset' objects within a 'DatasetDict'.
    """

    datasets = {}

    for split_name, split in splits.items():
        if sequence_type:
            data_split = data[data.Type == sequence_type]
        else:
            data_split = data

        if dataset_size:
            data_split = data_split[data_split.Partition_No.isin(split)].sample(n=dataset_size * len(split), random_state=1)
        else:
            data_split = data_split[data_split.Partition_No.isin(split)]
        tokenized_sequences = tokenizer(data_split.Sequence.to_list(), padding=True, truncation=True, return_tensors="pt", max_length=1024)
        dataset = Dataset.from_dict(tokenized_sequences)
        if annotations_name == 'Label':
            dataset = dataset.add_column("labels", [[encoder[y] for y in x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
        if annotations_name == 'Type':
            dataset = dataset.add_column("labels", [encoder[x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
        datasets[split_name] = dataset

    # for x in datasets.values():
    #     x.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return DatasetDict(datasets)


# def create_datasets_head(splits: dict, tokenizer: T5Tokenizer, data: pd.DataFrame, annotations_name: str, dataset_size: int, encoder: dict) -> DatasetDict:
#     datasets = {}

#     for split_name, split in splits.items():
#         data_split = data[data.Partition_No.isin(split)].head(dataset_size * len(split) if dataset_size else dataset_size)
#         if
#         tokenized_sequences = tokenizer(data_split.Sequence.to_list(), padding=True, truncation=True, return_tensors="pt", max_length=1024)
#         dataset = Dataset.from_dict(tokenized_sequences)
#         if annotations_name == 'Label':
#             dataset = dataset.add_column("labels", [[encoder[y] for y in x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
#         if annotations_name == 'Type':
#             dataset = dataset.add_column("labels", [encoder[x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
#         datasets[split_name] = dataset
#     return DatasetDict(datasets)


###################################################
# Inference
###################################################


def predict_model(sequence: str, tokenizer: T5Tokenizer, model: T5EncoderModelForTokenClassification, attention_mask=None, labels=None, device='cpu', viterbi_decoding=False):
    tokenized_string = tokenizer.encode(sequence, padding=True, truncation=True, return_tensors="pt", max_length=1024).to(device)
    # print(tokenized_string.shape)
    # print(labels.shape)
    # print(attention_mask.shape)
    with torch.no_grad():
        output = model(
            input_ids=tokenized_string.to(device),
            labels=None if viterbi_decoding else labels,
            attention_mask=attention_mask,
            )
    return output


def translate_logits(logits, viterbi_decoding=False):
    # print('logits', logits, type(logits))
    if viterbi_decoding:
        return [src.config.label_decoding[x] for x in logits[0]]
    else:
        return [src.config.label_decoding[x] for x in logits.cpu().numpy().argmax(-1).tolist()[0]]


###################################################
# Plots
###################################################


def make_confusion_matrix(data_cm, decoding):

    print(decoding)
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


###################################################
# Metrics
###################################################


accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
# roc_auc_score_metric = evaluate.load("roc_auc", "multiclass")
matthews_correlation_metric = evaluate.load("matthews_correlation")


def batch_eval_elementwise(predictions: np.ndarray, references: np.ndarray):
    results = {}

    if np.isnan(predictions).any():
        print('has nan')
        predictions = np.nan_to_num(predictions)

    # print('predictions', predictions, type(predictions))

    argmax_predictions = predictions.argmax(axis=-1)
    vals = list((np.array(p)[(r != -100)], np.array(r)[(r != -100)]) for p, r in zip(argmax_predictions.tolist(), references))

    lst1, lst2 = zip(*vals)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=np.concatenate(lst1), y_pred=np.concatenate(lst2))

    results.update({'accuracy_metric': np.average([accuracy_metric.compute(predictions=x, references=y)['accuracy'] for x, y in vals])})
    results.update({'precision_metric': np.average([precision_metric.compute(predictions=x, references=y, average='micro')['precision'] for x, y in vals])})
    results.update({'recall_metric': np.average([recall_metric.compute(predictions=x, references=y, average='micro')['recall'] for x, y in vals])})
    results.update({'f1_metric': np.average([f1_metric.compute(predictions=x, references=y, average='micro')['f1'] for x, y in vals])})
    # results.update({'roc_auc': [roc_auc_score_metric.compute(prediction_scores=x, references=y, multi_class='ovr', average=None)['roc_auc'] for x, y in zip(softmax_predictions, references)]})
    results.update({'matthews_correlation': np.average([matthews_correlation_metric.compute(predictions=x, references=y, average='micro')['matthews_correlation'] for x, y in vals])})
    results.update({'confusion_matrix': confusion_matrix})

    return results


def compute_metrics(p):
    predictions, references = p
    results = batch_eval_elementwise(predictions=predictions, references=references)
    return results


