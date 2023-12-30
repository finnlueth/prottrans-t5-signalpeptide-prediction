import gc
import time

from transformers import (
    T5EncoderModel,
    T5PreTrainedModel,
    T5Tokenizer,
    T5Config,
    modeling_outputs,
)

from torch.nn import (
    CrossEntropyLoss,
    MSELoss
)

import evaluate

import torch
import torch.nn as nn
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
    ):
        super().__init__(config)
        self.custom_num_labels = custom_num_labels
        self.custom_dropout_rate = custom_dropout_rate

        self.custom_dropout = nn.Dropout(self.custom_dropout_rate)
        self.custom_classifier = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.custom_num_labels
        )
        self.printed_initial_loss = False

    # From https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/bert/modeling_bert.py#L1716
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
        # print('--------------- forward ---------------')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print('custom_classifier', self.custom_classifier.weight)
        # print('custom_classifier nans', self.custom_classifier.weight.isnan().any())
        # print('custom_classifier', self.custom_classifier.modules_to_save.default.weight)
        # print('custom_classifier nans', self.custom_classifier.modules_to_save.default.weight.isnan().any())
        # print('attention_mask', attention_mask)
        # print(self.encoder)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print('self.encoder', super().forward)

        # print('self.config.hidden_size', self.config.hidden_size)
        # print('self.custom_num_labels', self.custom_num_labels)
        # print()
        # print('encoder_outputs.last_hidden_state', encoder_outputs.last_hidden_state)
        # print('encoder_outputs.last_hidden_state shape', encoder_outputs.last_hidden_state.shape)

        sequence_output = encoder_outputs['last_hidden_state']

        # print('sequence_output', sequence_output)
        # print('sequence_output min', sequence_output.min())
        # print('sequence_output max', sequence_output.max())
        # print('sequence_output hasnan', sequence_output.isnan().any())
        sequence_output = self.custom_dropout(sequence_output)
        # print('sequence_output dropout', sequence_output)
        logits = self.custom_classifier(sequence_output)
        # print('sequence_output linear', sequence_output)

        # print(self.custom_num_labels)
        # print(self.custom_classifier.modules_to_save.default.weight)

        # print(labels)
        loss = None
        if labels is not None:
            # print('found labels')
            # print('labels', labels.view(-1, self.custom_num_labels))
            loss_fct = CrossEntropyLoss()

            # print('logits.device', logits.device)
            labels = labels.to(logits.device)
            # print(labels)
            # print('labels', labels.view(-1))
            # print('labels', labels.view(-1).shape)
            # print('labels hasnan', labels.view(-1).isnan().any())
            # print('logits', logits.view(-1, self.custom_num_labels))
            # print('logits', logits.view(-1, self.custom_num_labels).shape)
            # print('logits hasnan', logits.view(-1, self.custom_num_labels).isnan().any())
            # print('logits', logits.view(-1, self.custom_num_labels).argmax(dim=-1))
            # print()

            loss = loss_fct(logits.view(-1, self.custom_num_labels), labels.view(-1))
            # print('loss', loss)

        # print('return_dict', return_dict)
        # print('loss', loss)
        # print('logits', logits)
        # print('labels', labels)
        # print('encoder_outputs.attentions', encoder_outputs.attentions)
        # print('encoder_outputs.hidden_states', encoder_outputs.hidden_states)
        # print(*encoder_outputs)
        # print('------------- end forward -------------')
        # print('custom_classifier', self.custom_classifier.modules_to_save.default.weight)
        # print('custom_classifier nans', self.custom_classifier.modules_to_save.default.weight.isnan().any())
        if not self.printed_initial_loss:
            print('loss', loss)
            self.printed_initial_loss = True

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


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
        self.custom_classifier = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.custom_num_labels
        )
        self.printed_initial_loss = False

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

        sequence_output = encoder_outputs['last_hidden_state']
        sequence_output = self.custom_dropout(sequence_output)

        mean_sequence_output = torch.mean(sequence_output, dim=1)

        print(mean_sequence_output.shape)
        print(mean_sequence_output)

        logits = self.custom_classifier(mean_sequence_output)

        print(logits.shape)
        print(logits)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.custom_num_labels), labels.view(-1))

        if not self.printed_initial_loss:
            print('loss', loss)
            self.printed_initial_loss = True

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )



def df_to_dataset(tokenizer: T5Tokenizer, sequences: list, labels: list, encoder: dict) -> Dataset:
    tokenized_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    dataset = Dataset.from_dict(tokenized_sequences)
    dataset = dataset.add_column("labels", [encoder[x] for x in labels], new_fingerprint=None)
    return dataset


def create_datasets(splits: dict, tokenizer: T5Tokenizer, data: pd.DataFrame, annotations_name: str, dataset_size: int, encoder: dict, sequence_type=None) -> DatasetDict:
    datasets = {}

    for split_name, split in splits.items():
        if sequence_type:
            data_split = data[data.Type == sequence_type]
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


def predict_model(sequence: str, tokenizer: T5Tokenizer, model: T5EncoderModelForTokenClassification, attention_mask=None, labels=None, device='cpu'):
    tokenized_string = tokenizer.encode(sequence, padding=True, truncation=True, return_tensors="pt", max_length=1024).to(device)
    with torch.no_grad():
        output = model(
            input_ids=tokenized_string.to(device),
            labels=labels,
            attention_mask=attention_mask,
            )
    return output


def translate_logits(logits):
    return [src.config.label_decoding[x] for x in logits.argmax(-1).tolist()[0]]


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
