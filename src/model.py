import re
import os
import math
import copy
import types
import yaml

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
    T5EncoderModel,
    T5Tokenizer,
    T5PreTrainedModel,
    T5ForConditionalGeneration,
    pipeline,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
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

label_encoding = config.label_encoding
label_list = config.label_decoding
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


def injected_forward(
    self,
    input_ids=None,
    attention_mask=None,
    # position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None
    ):

    return_dict = return_dict if return_dict is not None else self.get_base_model().config.use_return_dict

    outputs = self.get_base_model().encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = outputs[0]

    sequence_output = self.get_base_model().dropout(sequence_output)
    logits = self.classifier(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()

        labels = labels.to(logits.device)
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return TokenClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def inject_linear_layer(t5_lora_model):
    t5_lora_model.forward = types.MethodType(injected_forward, t5_lora_model)
    return t5_lora_model


def compute_metrics_full(p):
    predictions, labels = p
    # predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
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
    metric = load("accuracy")
    predictions, labels = p
    
    labels = labels.reshape((-1,))
    
    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.reshape((-1,))
    
    predictions = predictions[labels != -100]
    labels = labels[labels != -100]
    
    return metric.compute(predictions=predictions, references=labels)