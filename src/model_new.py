import gc
import time

from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    T5Config,
    modeling_outputs,
)

from torch.nn import (
    CrossEntropyLoss,
    MSELoss
)

import torch.nn as nn

import peft

import pandas as pd

from datasets import Dataset, DatasetDict


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

        # print(self.encoder)

        encoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        print('self.encoder', super().forward)
        
        print('self.config.hidden_size', self.config.hidden_size)
        print('self.custom_num_labels', self.custom_num_labels)
        print()
        print('encoder_outputs.last_hidden_state', encoder_outputs.last_hidden_state)
        
        sequence_output = encoder_outputs.last_hidden_state

        sequence_output = self.custom_dropout(sequence_output)
        print('sequence_output dropout', sequence_output)
        logits = self.custom_classifier(sequence_output)
        print('sequence_output linear', sequence_output)
        
        # print(self.custom_num_labels)

        loss = None
        if labels is not None:
            print('found labels')
            # print(labels)
            loss_fct = CrossEntropyLoss()
            
            print('logits.device', logits.device)
            labels = labels.to(logits.device)
            print(labels)
            # print(logits.view(-1, self.custom_num_labels))
            # print(labels.view(-1))
            # print()
            # print(logits.view(-1, self.custom_num_labels).shape)
            # print(labels.view(-1).shape)
            
            loss = loss_fct(logits.view(-1, self.custom_num_labels), labels.view(-1))
            print('loss', loss)

        # print('return_dict', return_dict)
        # print('loss', loss)
        # print('logits', logits)
        # print('labels', labels)
        # print('encoder_outputs.attentions', encoder_outputs.attentions)
        # print('encoder_outputs.hidden_states', encoder_outputs.hidden_states)
        # print(*encoder_outputs)
        # print('------------- end forward -------------')
        print('loss', loss)
        
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


def create_datasets(splits: dict, tokenizer: T5Tokenizer, data: pd.DataFrame, annotations_name: str, dataset_size: int, encoder: dict) -> DatasetDict:
    datasets = {}
    
    if dataset_size:
        for split_name, split in splits.items():
            data_split = data[data.Partition_No.isin(split)].sample(n=dataset_size * len(split) if dataset_size else dataset_size, random_state=1)
            tokenized_sequences = tokenizer(data_split.Sequence.to_list(), padding=True, truncation=True, return_tensors="pt", max_length=1024)
            dataset = Dataset.from_dict(tokenized_sequences)
            if annotations_name == 'Label':
                dataset = dataset.add_column("labels", [[encoder[y] for y in x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
            if annotations_name == 'Type':
                dataset = dataset.add_column("labels", [encoder[x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
            datasets[split_name] = dataset
        return DatasetDict(datasets)
    else:
        for split_name, split in splits.items():
            data_split = data[data.Partition_No.isin(split)]
            tokenized_sequences = tokenizer(data_split.Sequence.to_list(), padding=True, truncation=True, return_tensors="pt", max_length=1024)
            dataset = Dataset.from_dict(tokenized_sequences)
            if annotations_name == 'Label':
                dataset = dataset.add_column("labels", [[encoder[y] for y in x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
            if annotations_name == 'Type':
                dataset = dataset.add_column("labels", [encoder[x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
            datasets[split_name] = dataset
        return DatasetDict(datasets)

# [encoder[x] for x in data[annotations_name].to_list()]


def create_datasets_head(splits: dict, tokenizer: T5Tokenizer, data: pd.DataFrame, annotations_name: str, dataset_size: int, encoder: dict) -> DatasetDict:
    datasets = {}
    
    for split_name, split in splits.items():
        data_split = data[data.Partition_No.isin(split)].head(dataset_size * len(split) if dataset_size else dataset_size)
        tokenized_sequences = tokenizer(data_split.Sequence.to_list(), padding=True, truncation=True, return_tensors="pt", max_length=1024)
        dataset = Dataset.from_dict(tokenized_sequences)
        if annotations_name == 'Label':
            dataset = dataset.add_column("labels", [[encoder[y] for y in x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
        if annotations_name == 'Type':
            dataset = dataset.add_column("labels", [encoder[x] for x in data_split[annotations_name].to_list()], new_fingerprint=None)
        datasets[split_name] = dataset
    return DatasetDict(datasets)
