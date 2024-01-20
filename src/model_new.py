import gc
import time

import matplotlib.pyplot as plt
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
                logits_crf = logits[:, :-1, :]
                labels_crf = labels[:, :-1]
                # fix mask for shorter sequences
                attention_mask_crf = attention_mask[:, :-1]
                attention_mask_crf[labels_crf == -100] = 0
                labels_crf[labels_crf == -100] = 0
                # print('logits', logits_crf.shape, logits_crf.dtype, logits_crf, logits_crf.min(), logits_crf.max())
                # print('labels', labels_crf.shape, labels_crf.dtype, labels_crf, labels_crf.min(), labels_crf.max())
                # print('attention_mask', attention_mask_crf.shape, attention_mask_crf.dtype, attention_mask_crf)
                log_likelihood = self.crf(
                    emissions=logits_crf,
                    tags=labels_crf,
                    mask=attention_mask_crf.type(torch.uint8)
                )
                loss = -log_likelihood/1000
                # print('neglog_likelihood', loss)
            else:
                # decoded_tags = self.crf.modules_to_save.default.decode(logits, mask=attention_mask.type(torch.uint8))
                decoded_tags = self.crf.decode(logits, mask=attention_mask.type(torch.uint8))
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

        # print('sequence_output.requires_grad', sequence_output.requires_grad)
        # print(mean_sequence_output.shape)
        # print(mean_sequence_output)
        # print('sequence_output', sequence_output[:, 0, :].shape, sequence_output[:, 0, :])
        # print()
        # print('sequence_output', sequence_output.shape, sequence_output)

        logits = sequence_output.mean(dim=1)
        logits = self.custom_dropout(logits)
        logits = self.custom_classifier_in(logits)
        logits = logits.tanh()
        logits = self.custom_dropout(logits)
        logits = self.custom_classifier_out(logits)

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
            # hidden_states=encoder_outputs.last_hidden_state,
            # attentions=encoder_outputs.attentions,
        )


######################################################################################################
# Helper Functions
######################################################################################################


def df_to_dataset(tokenizer: T5Tokenizer, sequences: list, labels: list, encoder: dict) -> Dataset:
    tokenized_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    dataset = Dataset.from_dict(tokenized_sequences)
    dataset = dataset.add_column("labels", [encoder[x] for x in labels], new_fingerprint=None)
    return dataset


def create_datasets(splits: dict, tokenizer: T5Tokenizer, data: pd.DataFrame, annotations_name: list, dataset_size: int = None, sequence_type=None) -> DatasetDict:
    datasets = {}

    for split_name, split in splits.items():
        if sequence_type and sequence_type != 'ALL':
            data_split = data[data['Type'] == sequence_type]
        else:
            data_split = data

        if dataset_size:
            data_split = data_split[data_split.Partition_No.isin(split)].sample(n=dataset_size * len(split), random_state=1)
        else:
            data_split = data_split[data_split.Partition_No.isin(split)]

        tokenized_sequences = tokenizer(data_split['Sequence'].to_list(), padding=True, truncation=True, return_tensors="pt", max_length=1024)
        dataset = Dataset.from_dict(tokenized_sequences)
        if 'Label' in annotations_name:
            encoder = src.config.select_encoding_type[sequence_type]
            # print(data_split)
            dataset = dataset.add_column("labels", [[encoder[y] for y in x] for x in data_split['Label'].to_list()], new_fingerprint=None)
        if 'Type' in annotations_name:
            encoder = src.config.type_encoding
            dataset = dataset.add_column("type", [encoder[x] for x in data_split['Type'].to_list()], new_fingerprint=None)
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
    if tokenizer:
        tokenized_string = tokenizer.encode(sequence, padding='max_length', max_length=71, truncation=True, return_tensors="pt").to(device)
    # print(tokenized_string.shape)
    # print(attention_mask.shape)
    with torch.no_grad():
        output = model(
            input_ids=tokenized_string.to(device),
            labels=None if viterbi_decoding else labels,
            attention_mask=attention_mask,
            )
    return output


def translate_logits(logits, decoding, viterbi_decoding=False):
    # print('logits', logits, type(logits))
    if viterbi_decoding:
        return [decoding[x] for x in logits[0]]
    else:
        return [decoding[x] for x in logits.cpu().numpy().argmax(-1).tolist()[0]]


def moe_inference(sequence, tokenizer, model_gate, model_expert, labels=None, attention_mask=None, device='cpu', result_type=None, use_crf=False):
    if not result_type:
        print('Determining type...')
        gate_preds = src.model_new.predict_model(
            sequence=sequence,
            tokenizer=tokenizer,
            model=model_gate,
            labels=labels,
            attention_mask=attention_mask,
            device=device,
            )

        result_type = src.model_new.translate_logits(
            logits=gate_preds.logits.unsqueeze(0),
            decoding=src.config.type_decoding
            )[0]
        print('Found Type: ', result_type)

    try:
        assert result_type in src.config.type_encoding.keys()
    except Exception as e:
        print(e)
        print(f'Type "{result_type}" not found. Exiting...')

    tmp_custom_classifier = nn.Linear(
            in_features=model_expert.config.hidden_size,
            out_features=len(src.config.select_decoding_type[result_type])
        )
    if use_crf:
        tmp_crf = CRF(
            num_tags=len(src.config.select_decoding_type[result_type]),
            batch_first=True
        )
        model_expert.custom_classifier.weight = tmp_custom_classifier.weight
        model_expert.custom_classifier.bias = tmp_custom_classifier.bias
        model_expert.crf.start_transitions = tmp_crf.start_transitions
        model_expert.crf.end_transitions = tmp_crf.end_transitions
        model_expert.crf.transitions = tmp_crf.transitions

    model_expert.set_adapter(result_type)

    pred_logits = src.model_new.predict_model(
        sequence=sequence,
        tokenizer=tokenizer,
        model=tokenizer,
        labels=labels,
        attention_mask=attention_mask,
        device=device,
        viterbi_decoding=use_crf,
        )

    pred_sequence = src.model_new.translate_logits(
        logits=pred_logits.logits,
        decoding=src.config.select_decoding_type[result_type],
        viterbi_decoding=use_crf,
        )

    model_expert.unload()

    return result_type, pred_sequence


###################################################
# Plots
###################################################


def confusion_matrix_plot(df_cm, decoding, title='Confusion Matrix'):
    plt.figure(figsize=(16, 9))
    sns.set(font_scale=2)

    ax = sns.heatmap(
        df_cm,
        annot=True,
        xticklabels=[decoding[label] for label in range(len(decoding))],
        yticklabels=[decoding[label] for label in range(len(decoding))],
        fmt='d',
        label='big'
    )

    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    return ax


def loss_plot(df_log):
    plt.figure(figsize=(16, 9))
    sns.lineplot(x=df_log.index, y=df_log['loss'], label='Training Loss', color='orange')
    sns.lineplot(x=df_log.index, y=df_log['eval_loss'], label='Validation Loss', color='blue')
    sns.lineplot(x=df_log.index, y=df_log['eval_accuracy_metric'], label='Accuracy', linestyle='--', color='green')
    # sns.lineplot(x=df_log.index, y=df_log['eval_precision_metric'], label='Precision', linestyle=':', color='purple')
    # sns.lineplot(x=df_log.index, y=df_log['eval_recall_metric'], label='Recall', linestyle='-.', color='orange')
    # sns.lineplot(x=df_log.index, y=df_log['eval_f1_metric'], label='F1 Score', linestyle='-', color='red')
    sns.lineplot(x=df_log.index, y=df_log['eval_matthews_correlation'], label='Matthews Correlation', linestyle='--', color='brown')

    plt.xlabel('Step')
    plt.ylabel('Metrics')
    plt.title('Training Loss and Evaluation Metrics')

    plt.legend()

    return plt


# def validation_evaluation_plots(df_training_log: pd.DataFrame, decoding):
#     # print(decoding)
#     CM = []

#     # for x in df_training_log['eval_confusion_matrix'][df_training_log['eval_confusion_matrix'].notnull()]:
#     #     cm = confusion_matrix_plot(
#     #         data_cm=x,
#     #         decoding=decoding
#     #         )
#         # CM.append(cm)

#     lp = loss_plot(df_training_log)
    # return lp


###################################################
# Metrics
###################################################


accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
roc_auc_score_metric = evaluate.load("roc_auc", "multiclass")
matthews_correlation_metric = evaluate.load("matthews_correlation")


def batch_eval_elementwise(predictions: np.ndarray, references: np.ndarray):
    results = {}
    # print(predictions, references)
    # print(type(predictions), type(references))

    # if np.isnan(predictions).any():
    #     print('has nan')
    #     predictions = np.nan_to_num(predictions)

    argmax_predictions = predictions.argmax(axis=-1)
    vals = list((np.array(p)[(r != -100)], np.array(r)[(r != -100)]) for p, r in zip(argmax_predictions.tolist(), references))

    lst_pred, lst_true = zip(*vals)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=np.concatenate(lst_true), y_pred=np.concatenate(lst_pred))

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
    # if type(predictions) is tuple:
    #     predictions = predictions[0]
    # print(predictions)
    # print(type(predictions))
    # print(predictions[0].shape)
    # print(predictions[1].shape)
    # print(predictions[1])

    results = batch_eval_elementwise(predictions=predictions, references=references)
    return results


# def compute_metrics_crf(p):
#     predictions, references = p
#     results = {}
#     if np.isnan(predictions).any():
#         print('has nan')
#         predictions = np.nan_to_num(predictions)

#     vals = list((np.array(p)[(r != -100)], np.array(r)[(r != -100)]) for p, r in zip(predictions.tolist(), references))

#     lst_pred, lst_true = zip(*vals)
#     confusion_matrix = sklearn.metrics.confusion_matrix(y_true=np.concatenate(lst_true), y_pred=np.concatenate(lst_pred))

#     results.update({'accuracy_metric': np.average([accuracy_metric.compute(predictions=x, references=y)['accuracy'] for x, y in vals])})
#     results.update({'precision_metric': np.average([precision_metric.compute(predictions=x, references=y, average='micro')['precision'] for x, y in vals])})
#     results.update({'recall_metric': np.average([recall_metric.compute(predictions=x, references=y, average='micro')['recall'] for x, y in vals])})
#     results.update({'f1_metric': np.average([f1_metric.compute(predictions=x, references=y, average='micro')['f1'] for x, y in vals])})
#     # results.update({'roc_auc': [roc_auc_score_metric.compute(prediction_scores=x, references=y, multi_class='ovr', average=None)['roc_auc'] for x, y in zip(softmax_predictions, references)]})
#     results.update({'matthews_correlation': np.average([matthews_correlation_metric.compute(predictions=x, references=y, average='micro')['matthews_correlation'] for x, y in vals])})
#     results.update({'confusion_matrix': confusion_matrix})

#     return results
