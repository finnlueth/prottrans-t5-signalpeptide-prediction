
# ProtTans Finetuning with LoRA for Signal Peptide Prediction

## Links
### Papers/ Knowledge
- https://www.sciencedirect.com/science/article/pii/S2001037021000945
- https://huggingface.co/blog/peft
- https://ieeexplore.ieee.org/ielx7/34/9893033/9477085/supp1-3095381.pdf?arnumber=9477085
### Architecture
- https://www.philschmid.de/fine-tune-flan-t5-peft
- https://huggingface.co/spaces/evaluate-metric/seqeval
- https://huggingface.co/docs/transformers/v4.33.3/en/model_doc/esm#transformers.EsmForTokenClassification
- https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/builder_classes#datasets.SplitGenerator
- https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.add_column
- https://huggingface.co/docs/transformers/main_classes/data_collator
- https://huggingface.co/docs/transformers/main/en/main_classes/trainer#checkpoints
### Code
- https://github.com/ziegler-ingo/cleavage_extended/blob/master/models/final/c_bilstm_t5_coteaching.ipynb
- https://www.kaggle.com/code/henriupton/proteinet-pytorch-ems2-t5-protbert-embeddings/notebook#7.-Train-the-Model
- https://www.kaggle.com/code/prithvijaunjale/t5-multi-label-classification
### Optmization
- https://huggingface.co/blog/accelerate-large-models
- https://huggingface.co/docs/transformers/hpo_train

## ToDo
- Implement BitsAndBites (QLoRA)
- Implement DeepSpeed
- Fix weird extra char on inference

## Links

- https://huggingface.co/blog/4bit-transformers-bitsandbytes
- https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing#scrollTo=a9EUEDAl0ss3
- https://huggingface.co/docs/transformers/v4.18.0/en/performance
- https://colab.research.google.com/drive/1obr78FY_cBmWY5ODViCmzdY6O1KB65Vc?usp=sharing
