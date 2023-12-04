base_model_name = 'Rostlab/prot_t5_xl_uniref50'

splits = {
    'train': [0, 1, 2],
    'valid': [3],
    'test': [4]
}

#Encodings
label_encoding = {
    "I": 0,
    "L": 1,
    "M": 2,
    "O": 3,
    "S": 4,
    "T": 5,
    }
label_decoding = dict(zip(label_encoding.values(), label_encoding.keys()))

type_encoding = {
    'NO_SP': 0,
    'SP': 1,
    'LIPO': 2,
    'TAT': 3,
}
type_decoding = dict(zip(type_encoding.values(), type_encoding.keys()))

select_encodings = {
    'Label': label_encoding,
    'Type': type_encoding,
}

# Dataset URLs
urls = {
    '6_SignalP_6.0_Training_set.fasta': 'https://services.healthtech.dtu.dk/services/SignalP-6.0/public_data/train_set.fasta',
    '6_SignalP_5.0_Benchmark_set.fasta': 'https://services.healthtech.dtu.dk/services/SignalP-6.0/public_data/benchmark_set_sp5.fasta',
    '5_SignalP_5.0_Training_set.fasta': 'https://services.healthtech.dtu.dk/services/SignalP-5.0/train_set.fasta',
    '5_SignalP_5.0_Benchmark_set.fasta': 'https://services.healthtech.dtu.dk/services/SignalP-5.0/benchmark_set.fasta',
}

# Debug
VERBOSE = True

# Model
dropout_rate = 0.1

# Data
dataset_size = 3

# Training
steps = 30
lr = 1e-3
batch_size = 16
num_epochs = 1
save_steps = 100
logging_steps = 1
eval_steps = steps*5
eval_steps = 1
# metric = compute_metrics_fast

model_name = 'linear_model_v4'
