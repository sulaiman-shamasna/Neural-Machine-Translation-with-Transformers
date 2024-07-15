HYPERPARAMETERS = {
    'num_layers': 4,
    'd_model': 128,
    'dff': 512,
    'num_heads': 8,
    'dropout_rate': 0.1,
    'epochs': 1,
    'buffer_size': 20000,
    'batch_size': 64,
    'max_tokens': 128,
    'learning_rate_warmup_steps': 4000,
}

DATASET_NAME = 'ted_hrlr_translate/pt_to_en'
MODEL_SAVE_PATH = 'saved_model'