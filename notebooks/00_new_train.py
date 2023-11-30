from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    T5Config,
    modeling_outputs,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

import src.config as config

from peft import (
    LoraConfig,
)

import torch
import torch.nn as nn

import peft

import pandas as pd

from datasets import Dataset, DatasetDict

import gc

import time


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

        self.custom_classifier = nn.Linear(
            in_features=self.config.hidden_size,
            out_features=self.custom_num_labels
        )
        self.custom_dropout = nn.Dropout(self.custom_dropout_rate)

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
        print(self.custom_num_labels, self.custom_dropout_rate)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print(self.device)
        print(next(self.custom_classifier.parameters()).device)
        # print(return_dict is not None)
        # print(return_dict)
        # print()
        # print(super().forward)
        # print()
        encoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # print(type(self))
        # print(self)
        # print(self.classifier)
        # print()
        # print(type(encoder_outputs))
        # print()
        # print(encoder_outputs)
        # print()
        # print(encoder_outputs.last_hidden_state.shape)
        # print()
        # print(encoder_outputs[0])
        # print()
        # print(encoder_outputs[0].shape)

        # print(return_dict)
        # print('Made encoder_outputs')

        sequence_output = self.custom_dropout(encoder_outputs.last_hidden_state)
        print(sequence_output.shape)
        time.sleep(1)
        logits = self.custom_classifier(sequence_output)
        print(logits.shape)
        print(logits)

        print(logits.view(-1, self.custom_num_labels))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.custom_num_labels), labels.view(-1))
            print('Computing loss')

        print('Made computed loss')

        if not return_dict:
            print('not return_dict')
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return modeling_outputs.TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    ROOT = '../'

    t5_tokenizer = T5Tokenizer.from_pretrained(
        pretrained_model_name_or_path=config.base_model_name,
        do_lower_case=False,
        use_fast=True,
        legacy=False
    )

    t5_base_model = T5EncoderModelForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=config.base_model_name,
        device_map='auto',
        load_in_8bit=False,
        custom_num_labels=len(config.label_decoding),
        custom_dropout_rate=0.1,
    )

    lora_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['q', 'k', 'v', 'o'],
        bias="none",
    )
    
    t5_lora_model = peft.get_peft_model(t5_base_model, lora_config)
    t5_lora_model.print_trainable_parameters()
    
    def df_to_dataset(tokenizer: T5Tokenizer, sequences: list, annotations: list) -> Dataset:
        tokenized_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=1024)
        dataset = Dataset.from_dict(tokenized_sequences)
        dataset = dataset.add_column("labels", annotations, new_fingerprint=None)
        return dataset

    df_data = pd.read_parquet(ROOT + '/data/processed/5.0_train.parquet.gzip')

    ds_train = df_data[df_data.Split.isin([0, 1, 2])].head(config.dataset_size*3 if config.dataset_size else None)
    ds_train = df_to_dataset(
        t5_tokenizer,
        ds_train.Sequence.to_list(),
        ds_train.Label.to_list(),
    )

    ds_validate = df_data[df_data.Split.isin([3])].head(config.dataset_size)
    ds_validate = df_to_dataset(
        t5_tokenizer,
        ds_validate.Sequence.to_list(),
        ds_validate.Label.to_list(),
    )

    ds_test = df_data[df_data.Split.isin([4])].head(config.dataset_size)
    ds_test = df_to_dataset(
        t5_tokenizer,
        ds_test.Sequence.to_list(),
        ds_test.Label.to_list()
    )

    dataset_signalp = DatasetDict({
        'train': ds_train,
        'valid': ds_validate,
        'test': ds_test
            })

    del df_data

    data_collator = DataCollatorForTokenClassification(tokenizer=t5_tokenizer)

    training_args = TrainingArguments(
        output_dir='./checkpoints',
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        logging_steps=config.logging_steps,
        # save_strategy="steps",
        save_steps=config.save_steps,
        # evaluation_strategy="steps",
        # eval_steps=config.eval_steps,
        # load_best_model_at_end=True,
        # save_total_limit=5,
        seed=42,
        # fp16=True,
        # deepspeed=deepspeed_config,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=t5_lora_model,
        args=training_args,
        train_dataset=ds_train,
        # eval_dataset=ds_validate,
        data_collator=data_collator,
        # compute_metrics=config.metric,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.mps.empty_cache()
    
    trainer.train()