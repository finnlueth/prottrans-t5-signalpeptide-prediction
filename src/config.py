from src.model import (
    compute_metrics_fast
    )


base_model_name = 'Rostlab/prot_t5_xl_uniref50'

label_encoding = {
    "I": 0,
    "L": 1,
    "M": 2,
    "O": 3,
    "S": 4,
    "T": 5,
    # "</s>": -100,
    # "<pad>": -100,
    # "<unk>": -100
    }

label_decoding = dict(zip(label_encoding.values(), label_encoding.keys()))

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
dataset_size = 2


# Training
lr = 1e-3
batch_size = 16
num_epochs = 1
save_steps = 1
logging_steps = 1
metric = compute_metrics_fast
