# run.py
from config import HYPERPARAMETERS, DATASET_NAME, MODEL_SAVE_PATH
from data.prepare_data import load_data, load_tokenizers, prepare_batch, make_batches
from train import train_model

def main():
    train_examples, val_examples = load_data(DATASET_NAME)
    tokenizers = load_tokenizers('ted_hrlr_translate_pt_en_converter')
    
    train_batches = make_batches(
        train_examples, 
        HYPERPARAMETERS['buffer_size'], 
        HYPERPARAMETERS['batch_size'], 
        prepare_batch(tokenizers, HYPERPARAMETERS['max_tokens'])
    )
    
    val_batches = make_batches(
        val_examples, 
        HYPERPARAMETERS['buffer_size'], 
        HYPERPARAMETERS['batch_size'], 
        prepare_batch(tokenizers, HYPERPARAMETERS['max_tokens'])
    )

    input_vocab_size = tokenizers.pt.get_vocab_size().numpy()
    target_vocab_size = tokenizers.en.get_vocab_size().numpy()

    HYPERPARAMETERS['input_vocab_size'] = input_vocab_size
    HYPERPARAMETERS['target_vocab_size'] = target_vocab_size

    model = train_model(train_batches, val_batches, HYPERPARAMETERS)
    model.save(MODEL_SAVE_PATH)

if __name__ == '__main__':
    main()
