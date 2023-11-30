# with torch.no_grad():
#     logits = t5_lora_model(inputs['input_ids']).logits


# device = 'cpu'
# t5_lora_model.to(device)

# test_set = ds_test.with_format("torch", device=device)

# # For token classification we need a data collator here to pad correctly
# data_collator = DataCollatorForTokenClassification(t5_tokenizer) 

# # Create a dataloader for the test dataset
# test_dataloader = DataLoader(test_set, batch_size=16, shuffle = False, collate_fn = data_collator)

# # Put the model in evaluation mode
# t5_lora_model.eval()

# # Make predictions on the test dataset
# predictions = []
# # We need to collect the batch["labels"] as well, this allows us to filter out all positions with a -100 afterwards
# padded_labels = []

# counter = 0

# with torch.no_grad():
#     for batch in test_dataloader:
#         print(counter)
#         counter += 1
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         # Padded labels from the data collator
#         padded_labels += batch['labels'].tolist()
#         # Add batch results(logits) to predictions, we take the argmax here to get the predicted class
#         prediction = t5_lora_model(input_ids=input_ids)
#         print(prediction)
#         predictions += prediction.logits.argmax(dim=-1).tolist()


# print(*predictions)


# print(test_set[0]['labels'])
# print(*[config.label_decoding[x] for x in test_set[0]['labels'].tolist()])


# print(*[config.label_decoding[x] for x in predictions[0]])


# t5_lora_model(ds_test['input_ids'][0])


# t5_tokenizer.decode(padded_labels[0])
# print(*[config.label_decoding[x] for x in padded_labels[0]])


# t5_lora_model()


# type(predictions)


# print(*[[config.label_decoding[y] for y in x] for x in predictions][0])


# base_model_test = T5ForConditionalGeneration.from_pretrained(
#     base_model_name,
#     device_map='auto',
#     offload_folder='./offload',
#     load_in_8bit=False
# )
# tsss_ids = t5_tokenizer('M A P T L F Q K L F S K R T G L G A P G R D A', return_tensors="pt").input_ids.to(device)
# tsss_mask = t5_tokenizer('M A P T L F Q K L F S K R T G L G A P G R D A', return_tensors="pt").attention_mask.to(device)
# base_model_test(input_ids=tsss_ids, decoder_input_ids=tsss_ids, attention_mask=tsss_mask)











# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"

# with open(ROOT+'/deepspeed_config.yaml', 'r') as file:
#     deepspeed_config = yaml.safe_load(file)
#     del file

# deepspeed_config = {
#     "fp16": {
#         "enabled": "auto",
#         "loss_scale": 0,
#         "loss_scale_window": 1000,
#         "initial_scale_power": 16,
#         "hysteresis": 2,
#         "min_loss_scale": 1
#     },

#     "optimizer": {
#         "type": "AdamW",
#         "params": {
#             "lr": "auto",
#             "betas": "auto",
#             "eps": "auto",
#             "weight_decay": "auto"
#         }
#     },

#     "scheduler": {
#         "type": "WarmupLR",
#         "params": {
#             "warmup_min_lr": "auto",
#             "warmup_max_lr": "auto",
#             "warmup_num_steps": "auto"
#         }
#     },

#     "zero_optimization": {
#         "stage": 2,
#         "offload_optimizer": {
#             "device": "cpu",
#             "pin_memory": True
#         },
#         "allgather_partitions": True,
#         "allgather_bucket_size": 2e8,
#         "overlap_comm": True,
#         "reduce_scatter": True,
#         "reduce_bucket_size": 2e8,
#         "contiguous_gradients": True
#     },

#     "gradient_accumulation_steps": "auto",
#     "gradient_clipping": "auto",
#     "steps_per_print": 2000,
#     "train_batch_size": "auto",
#     "train_micro_batch_size_per_gpu": "auto",
#     "wall_clock_breakdown": False
# }





# def df_to_dataset(tokenizer: T5Tokenizer, sequences: list, annotations: list) -> Dataset:
#     tokenized_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt", max_length=1024)
#     dataset = Dataset.from_dict(tokenized_sequences)
#     dataset = dataset.add_column("labels", annotations, new_fingerprint=None)
#     return dataset


# ds_train = df_data[df_data.Split.isin([0, 1, 2])].head(config.dataset_size*3 if config.dataset_size else config.dataset_size)
# ds_train = df_to_dataset(
#     t5_tokenizer,
#     ds_train.Sequence.to_list(),
#     ds_train.Label.to_list(),
# )

# ds_validate = df_data[df_data.Split.isin([3])].head(config.dataset_size)
# ds_validate = df_to_dataset(
#     t5_tokenizer,
#     ds_validate.Sequence.to_list(),
#     ds_validate.Label.to_list(),
# )

# ds_test = df_data[df_data.Split.isin([4])].head(config.dataset_size)
# ds_test = df_to_dataset(
#     t5_tokenizer,
#     ds_test.Sequence.to_list(),
#     ds_test.Label.to_list()
# )

# dataset_signalp = DatasetDict({
#     'train': ds_train,
#     'valid': ds_validate,
#     'test': ds_test
#         })

# del df_data








# def compute_metrics(p):
#     print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= preds compute_metrics start =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
#     predictions, references = p
#     print(type(predictions), type(references))
#     predictions = np.nan_to_num(predictions).argmax(axis=-1).tolist()
#     references = references.tolist()
#     # 
#     print('predictions', predictions)
#     print('references', references)
    
#     decoded_predictions = [[[src.config.label_decoding[y] for y in x] for x in predictions]]
#     decoded_references = [[[src.config.label_decoding[y] for y in x] for x in references]]
    
#     print('decoded_predictions', decoded_predictions)
#     print('decoded_references', decoded_references)

#     # print('done')
#     results_seqeval = seqeval.compute(predictions=decoded_predictions, references=decoded_references)
#     results_mcc = np.average([matthews_correlation.compute(predictions=x, references=y)['matthews_correlation'] for x, y in zip(predictions, references)])
#     print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= preds compute_metrics stop =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
#     return {
#         "accuracy": results_seqeval["overall_accuracy"],
#         "precision": results_seqeval["overall_precision"],
#         "recall": results_seqeval["overall_recall"],
#         "f1": results_seqeval["overall_f1"],
#         "mcc": results_mcc,
#     }