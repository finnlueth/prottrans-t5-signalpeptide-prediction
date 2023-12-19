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










# for index, (param_name, param) in enumerate(t5_lora_model.named_parameters()):
#     if index == 11:
#         break
#     # if param.requires_grad:
#     print(param_name)
#     print(param)

# _ds_index = 0
# _ds_type = 'train'

# _instri = t5_tokenizer.decode(dataset_signalp[_ds_type][_ds_index]['input_ids'])
# _restri = dataset_signalp[_ds_type][_ds_index]['labels']
# _restri_decoded = [src.config.label_decoding[x] for x in _restri]
# print(_instri)
# print(_restri)
# print(_restri_decoded)

# def predict_model(sequence: str, tokenizer: T5Tokenizer, model: T5EncoderModelForTokenClassification):
#     print('sequence', sequence)
#     tokenized_string = tokenizer.encode(sequence, padding=True, truncation=True, return_tensors="pt", max_length=1024)
#     print('tokenized_string', tokenized_string)
#     with torch.no_grad():
#         output = model(tokenized_string.to(device))
#     print('output', output)
#     return output

# # test_seq = 'M K N W L L L S V P L L L L L G S S S'
# test_seq = 'M L G T V K M E G H E T S D W N S Y Y A D T Q E A Y S S V P V S N M N S G L G S M N S M N T Y M T M N T M T T S G N M T P A S F N M S Y A N'

# preds = predict_model(test_seq, t5_tokenizer, t5_lora_model)

# preds.logits.argmax(axis=-1)

# print([src.config.label_decoding[x] for x in preds.logits.argmax(axis=-1).tolist()[0]])

# t5_lora_model

# adapter_location = '/models/testing'
# t5_lora_model.save_pretrained(ROOT + adapter_location)

# for x in t5_lora_model.named_parameters():
#     print(x[0])

# params = []
# for index, (param_name, param) in enumerate(t5_base_model.named_parameters()):
#     # if index == 11:
#     #     break
#     if param.requires_grad:
#         print(param_name)
#     # print(param)
#     if param_name in ['custom_classifier.bias', 'custom_classifier.weight']:
#         print(param)
#         params.append(param)

# # params[0].sum()

# t5_base_model_reloaded = T5EncoderModelForTokenClassification.from_pretrained(
#     pretrained_model_name_or_path=src.config.base_model_name,
#     device_map='auto',
#     load_in_8bit=False,
#     custom_num_labels=len(src.config.label_decoding),
#     custom_dropout_rate=0.1,
#     )

# t5_base_model_reloaded_original = copy.deepcopy(t5_base_model_reloaded)

# del t5_base_model_reloaded
# torch.mps.empty_cache()

# adapter_location = '/models/testing'
# t5_lora_model_reloaded = PeftModel.from_pretrained(
#     model = t5_base_model_reloaded,
#     is_trainable=False,
#     model_id=ROOT+adapter_location,
#     custom_num_labels=len(src.config.label_decoding),
#     custom_dropout_rate=0.1,
# )

# t5_lora_model_copy = copy.deepcopy(t5_lora_model)


# for index, (param1, param2) in enumerate(zip(t5_lora_model.parameters(), t5_lora_model_copy.parameters())):
#     if not torch.equal(param1.data.nan_to_num(), param2.data.nan_to_num()):
#         print(f"Models have different weights on layer {index}")
#         # print(param1.data)
#         # print(param2.data)
#         # break
# else:
#     print("Models have identical weights")


# param1.sum()

# param2.sum()


# for index, (param1, param2) in enumerate(zip(t5_lora_model_reloaded.parameters(), t5_lora_model.parameters())):
#     if not torch.equal(param1.data, param2.data):
#         print(f"Models have different weights on layer {index}")
#         break
# else:
#     print("Models have identical weights")


# torch.set_printoptions(profile="default")


# z = [x for x in t5_base_model_reloaded.parameters()]
# a = [x for x in t5_lora_model_reloaded.parameters()]
# b = [x for x in t5_lora_model.parameters()]


# print(len(z)) # base reload
# print(len(a)) # lora reload
# print(len(b)) # lora



# curr_index = 1
# a[curr_index].shape
# torch.equal(a[curr_index], b[curr_index])

# print(z[curr_index])
# print(a[curr_index])
# print(b[curr_index])

# print(sum(sum(z[curr_index])))
# print(sum(sum(a[curr_index])))
# print(sum(sum(b[curr_index])))

# # ds_test[0]

# defaul_reloaded = [x for x in t5_base_model_reloaded.parameters()][195]
# defaul_reloaded_og = [x for x in t5_base_model_reloaded_original.parameters()][195]

# for index, (x, y) in enumerate(zip(defaul_reloaded, defaul_reloaded_og)):
#     if not torch.equal(x, y):
#         print(index)
#         print(x)
#         print(y)
#         print('-------------------')
        
# print(*defaul_reloaded[2].tolist())
# print(*defaul_reloaded_og[2].tolist())

# (defaul_reloaded_og[2].tolist()[9])


# with torch.no_grad():
#     embds_1 = t5_base_model_reloaded.encoder(
#         input_ids=torch.tensor([[7, 7, 7, 7, 7]]).to('mps'),
#         attention_mask=torch.tensor([[1, 1, 1, 1, 1]]).to('mps')
#     )
    
    
# embds_2 = t5_base_model_reloaded_original.forward(
#     input_ids=torch.tensor([[7, 4, 7, 11, 7]]).to('mps'),
#     attention_mask=torch.tensor([[1, 1, 1, 1, 1]]).to('mps')
# )


# embds_1.last_hidden_state




# print(*[(n, type(m)) for n, m in t5_base_model.named_modules()], sep='\n')


# t5_lora_model = t5_base_model
# print(*[(n, type(m)) for n, m in t5_lora_model.named_modules()], sep='\n')
# t5_lora_model
# t5_lora_model.encoder.block[4].layer[0].SelfAttention.v.lora_A.default.weight


# count_params = 0

# for name, param in t5_lora_model.base_model.named_parameters():
#     # if "lora" not in name:
#     #     continue
#     count_params += param.numel()
#     print(f"New parameter {name:<13} | {param.numel():>5} parameters | updated")


# params_before = dict(t5_base_model_copy.named_parameters())
# for name, param in t5_lora_model.base_model.named_parameters():
#     if "lora" in name:
#         continue

#     name_before = name.partition(".")[-1].replace("original_", "").replace("module.", "").replace("modules_to_save.default.", "")
#     param_before = params_before[name_before]
#     if torch.allclose(param, param_before):
#         print(f"Parameter {name_before:<14} | {param.numel():>7} parameters | not updated")
#     else:
#         print(f"Parameter {name_before:<14} | {param.numel():>7} parameters | updated")


# print(*[(n, type(m)) for n, m in t5_base_model.named_modules()], sep='\n')













# predictions = torch.tensor([[[1,0,0,0,1], [2,3,2,1,2], [2,2,5,1,3]],
#                             [[1,2,0,0,0], [1,float('nan'),1,0,100], [1,4,3,7,10]]])
# predictions_argmaxed = np.nan_to_num(predictions).argmax(axis=-1)
# predictions_argmaxed = predictions.nan_to_num().argmax(dim=-1)
# print(predictions_argmaxed)

# references = torch.tensor([[0,1,3], [1,4,3]])
# print(references)

# torch.Size([3, 71, 1024])
# print(predictions.shape)

references = torch.tensor(dataset_signalp['train'][:2]['labels'])[1:2].cpu().numpy()
predictions = torch.tensor(dataset_signalp['train'][:2]['labels'])[1:2].cpu().numpy()

predictions[:,-1] = 4
# predictions[:,-2] = 2
# references = references[:,:-1]

# print(references)
# print(predictions)


# results = recall_metric.compute(predictions=predictions[0], references=references[0], average='micro')
# results = roc_auc_score_metric.compute(prediction_scores=(predictions[0], 6), references=references[0], multi_class='ovr')
# print(results)















# def batch_eval_flatten(predictions: np.ndarray, references: np.ndarray):
#     results = {}
#     predictions = np.nan_to_num(predictions).argmax(axis=-1)
#     predictions = np.ndarray.flatten(predictions)
#     references = np.ndarray.flatten(references)
    
#     results.update(accuracy_metric.compute(predictions=predictions, references=references))
#     results.update(precision_metric.compute(predictions=predictions, references=references, average='micro'))
#     results.update(recall_metric.compute(predictions=predictions, references=references, average='micro'))
#     results.update(f1_metric.compute(predictions=predictions, references=references, average='micro'))
#     # results.update(roc_auc_score_metric.compute(prediction_scores=predictions, references=references, average='micro'))
#     results.update(matthews_correlation_metric.compute(predictions=predictions, references=references, average='micro'))
#     return results
# # display(batch_eval_flatten(predictions.numpy(), references.numpy()))













# def confusion_matrix(vals):
#     """Computes the confusion matrix for a given set of predictions and references.
#     Args:
#         predictions: A list of predicted labels.
#         references: A list of reference labels.
#     Returns:
#         A numpy array representing the confusion matrix.
#     """
#     num_labels = len(src.config.label_decoding)
#     confusion_matrix = np.zeros((num_labels, num_labels), dtype=np.int32)
#     for prediction, reference in vals:
#         confusion_matrix[reference, prediction] += 1
#     return confusion_matrix







    results.update({'FP': FP})
    results.update({'FN': FN})
    results.update({'TP': TP})
    results.update({'TN': TN})
    results.update({'TPR': TPR})
    results.update({'TNR': TNR})
    results.update({'PPV': PPV})
    results.update({'NPV': NPV})
    results.update({'FPR': FPR})
    results.update({'FNR': FNR})
    results.update({'FDR': FDR})
    results.update({'ACC': ACC})
    
    
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    FPR = FP/(FP+TN)
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    
    
    # print(*[(n, type(m)) for n, m in t5_base_model.named_modules()], sep='\n')
    
        # print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= preds compute_metrics start =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    # print(p[0].shape, p[1].shape)
    # print(p[0].argmax(axis=-1))
    # print(p[1])
    
    # metrics = compute_metrics((predictions, references))
    
    # print('=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= preds compute_metrics stop =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')



# t5_lora_model.encoder.block[4].layer[0].SelfAttention.v.lora_A.default.weight