import re
import os
import math
import copy
import types
import yaml

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import (
    CrossEntropyLoss,
    MSELoss
)

import evaluate
from evaluate import load

from transformers import (
    AutoModelForTokenClassification,
    AutoConfig,
    T5ForConditionalGeneration,
    pipeline,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    )

from transformers import (
#     T5EncoderModel,
    T5Tokenizer,
    T5PreTrainedModel,
#     T5Config,
#     T5Stack,
#     BaseModelOutput
)

from transformers.modeling_outputs import TokenClassifierOutput

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    get_peft_config,
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training
    )

from datasets import Dataset

import src.config as config

seqeval = evaluate.load("seqeval")


def get_prottrans_tokenizer_model(base_model_name, model_architecture) -> (T5Tokenizer, T5PreTrainedModel):
    tokenizer = T5Tokenizer.from_pretrained(
        base_model_name,
        do_lower_case=False,
        use_fast=True,
        legacy=False
    )
    base_model = model_architecture.from_pretrained(
        base_model_name,
        device_map='auto',
        load_in_8bit=False
    )
    return (tokenizer, base_model)


def df_to_dataset(tokenizer: T5Tokenizer, sequences: list, annotations: list) -> Dataset:
    tokenized_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    dataset = Dataset.from_dict(tokenized_sequences)
    dataset = dataset.add_column("labels", annotations, new_fingerprint=None)
    return dataset


def compute_metrics_full(p):
    predictions, labels = p
    # predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [config.label_decoding[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [config.label_decoding[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def compute_metrics_fast(p):
    metric = evaluate.load("accuracy")
    print(p)
    predictions, labels = p

    labels = labels.reshape((-1,))

    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.reshape((-1,))

    predictions = predictions[labels != -100]
    labels = labels[labels != -100]

    m = metric.compute(predictions=predictions, references=labels)

    return m


def injected_forward(
    self,
    input_ids=None,
    attention_mask=None,
    # position_ids=None,
    # head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return_dict = return_dict if return_dict is not None else self.get_base_model().config.use_return_dict
    print('abc')
    print(return_dict is not None)
    print()

    encoder_outputs = self.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    print(type(self))
    print(self)
    print(self.classifier)
    print()
    print(type(encoder_outputs))
    print()
    print(encoder_outputs)
    print()
    print(encoder_outputs.last_hidden_state.shape)
    print()
    print(encoder_outputs[0])
    print()
    print(encoder_outputs[0].shape)

    # input_ids = input_ids[:, :70]
    # attention_mask = attention_mask[:, :70]

    # print(input_ids.shape)
    # print(attention_mask.shape)

    # print(type(encoder_outputs[0]))
    # print(outputs[0].shape)

    sequence_output = encoder_outputs[0]

    sequence_output = self.dropout(sequence_output)
    logits = self.classifier(sequence_output)

    # print(logits.view(-1, self.num_labels))

    loss = None
    if labels:
        loss_fct = CrossEntropyLoss()

        labels = labels.to(logits.device)
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
        print('not return_dict')
        output = (logits,) + encoder_outputs[2:]
        return ((loss,) + output) if loss is not None else output

    # print(type(loss))

    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


def inject_linear_layer(t5_lora_model, num_labels, dropout_rate):
    t5_lora_model.forward = types.MethodType(injected_forward, t5_lora_model)

    t5_lora_model.dropout = nn.Dropout(dropout_rate)
    t5_lora_model.classifier = nn.Linear(
        in_features=t5_lora_model.get_base_model().config.hidden_size,
        out_features=num_labels
    )
    t5_lora_model.num_labels = num_labels

    # t5_lora_model.get_base_model().dropout = nn.Dropout(dropout_rate)
    # t5_lora_model.get_base_model().classifier = nn.Linear(
    #     in_features=t5_lora_model.get_base_model().config.hidden_size,
    #     out_features=num_labels
    # )

    return t5_lora_model


# class T5LoraWrapper():
    # def __init__(self, lora_config: LoraConfig):
    #     super().__init__()

    #     self.t5_tokenizer, t5_base_model = get_prottrans_tokenizer_model(config.base_model_name, T5EncoderModel)
    #     self.t5_lora_model = get_peft_model(t5_base_model, lora_config)

    #     self.dropout = nn.Dropout(config.dropout_rate)
    #     self.num_labels = len(config.label_encoding)
    #     self.classifier = nn.Linear(
    #         in_features=self.t5_lora_model.get_base_model().config.hidden_size,
    #         out_features=self.num_labels
    #     )

    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     inputs_embeds=None,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     return_dict=None,
    #     labels=None,
    #     **kwargs,
    # ):
        # encoder_outputs = self.t5_lora_model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        #     **kwargs,
        # )[0]

        # encoder_outputs = self.dropout(encoder_outputs)
        # logits = self.classifier(encoder_outputs)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()

        #     labels = labels.to(logits.device)
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # TokenClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        # )
        # print('hi')

    # def get_lora_model(self):
    #     return self.t5_lora_model

    # def __getattr__(self, attr):
    #     if attr in self.__dict__:
    #         return getattr(self, attr)
    #     return getattr(self.t5_lora_model, attr)

# class T5ForTokenClassification(T5PreTrainedModel):
    # def __init__(self, config: T5Config):
    #     super().__init__(config)
    #     self.model_dim = config.d_model

    #     self.shared = nn.Embedding(config.vocab_size, config.d_model)

    #     encoder_config = copy.deepcopy(config)
    #     encoder_config.is_decoder = False
    #     encoder_config.use_cache = False
    #     encoder_config.is_encoder_decoder = False
    #     self.encoder = T5Stack(encoder_config, self.shared)

    #     decoder_config = copy.deepcopy(config)
    #     decoder_config.is_decoder = True
    #     decoder_config.is_encoder_decoder = False
    #     decoder_config.num_layers = config.num_decoder_layers
    #     self.decoder = T5Stack(decoder_config, self.shared)

    #     self.num_labels = config.num_labels
    #     self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    #     self.post_init()

    #     self.model_parallel = False

    # def get_input_embeddings(self):
    #     return self.shared

    # def set_input_embeddings(self, new_embeddings):
    #     self.shared = new_embeddings
    #     self.encoder.set_input_embeddings(new_embeddings)
    #     self.decoder.set_input_embeddings(new_embeddings)

    # def _tie_weights(self):
    #     if self.config.tie_word_embeddings:
    #         self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
    #         self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # def get_encoder(self):
    #     return self.encoder

    # def get_decoder(self):
    #     return self.decoder

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
    #     pass