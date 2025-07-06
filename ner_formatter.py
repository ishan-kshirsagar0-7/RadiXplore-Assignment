# This is the NER Formatter for my old approach of training the model only on annotations with a "Project" label.
# Basically, from our cleaned data, we have extracted the text body as well as the project label and its start-end points.
# Which means the next step for us to do, is to tokenize our text and label each token as either O, I-PROJECT or B-PROJECT.
# "O" stands for Outside (outside the named entity), B-PROJECT stands for Beginning (of the entity, here it's Project) and I means Inside.
# For example if the text is "The Minyari Dome Project is located in Australia", then the labelling would be "O B-PROJECT I-PROJECT I-PROJECT O O O O" (considering that tokenization was word-to-word and not subword).
# By training a small BERT model on such type of tokens, it can learn to identify patterns and hopefully extract project labels from unstructured
# text with an acceptable accuracy!

import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from tqdm import tqdm

# to ignore tokens like padding during loss calculation
IGNORED_LABEL_ID = -100 

# Function to convert our cleaned JSON data to IOB-Labelled data, which we need to train the NER model on.
def format_for_ner(cleaned_data_path, output_dir, tokenizer: PreTrainedTokenizer):
    print(f"Loading cleaned data from: {cleaned_data_path}")
    with open(cleaned_data_path, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    print("Tokenizing and aligning labels...")
    for record in tqdm(cleaned_data):
        text = record['text']
        entities = record['entities']

        tokenized_inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True
        )
        
        labels = [0] * len(tokenized_inputs['input_ids'])
        offset_mapping = tokenized_inputs.pop("offset_mapping")

        for entity in entities:
            start_char = entity['start']
            end_char = entity['end']
            token_start_index = None
            token_end_index = None

            for i, (offset_start, offset_end) in enumerate(offset_mapping):
                if offset_end == 0:
                    continue
                if max(offset_start, start_char) < min(offset_end, end_char):
                    if token_start_index is None:
                        token_start_index = i
                    token_end_index = i
            
            if token_start_index is not None and token_end_index is not None:
                labels[token_start_index] = 1 # B-PROJECT
                for i in range(token_start_index + 1, token_end_index + 1):
                    labels[i] = 2 # I-PROJECT

        for i, (offset_start, offset_end) in enumerate(offset_mapping):
            if offset_end == 0:
                labels[i] = IGNORED_LABEL_ID

        all_input_ids.append(tokenized_inputs['input_ids'])
        all_attention_masks.append(tokenized_inputs['attention_mask'])
        all_labels.append(labels)
        
    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(feature=Value(dtype='int64')),
        'labels': Sequence(feature=Value(dtype='int64')),
    })

    hf_dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    }, features=features)
    
    dataset_dict = hf_dataset.train_test_split(test_size=0.1, seed=42)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    print("-" * 50)
    print("Dataset creation complete!")
    print(f"Total records processed: {len(cleaned_data)}")
    print(f"Train set size: {len(dataset_dict['train'])}")
    print(f"Test set size: {len(dataset_dict['test'])}")
    print(f"Formatted dataset saved to: {output_dir}")
    print("-" * 50)


if __name__ == '__main__':
    INPUT_CLEANED_PATH = 'data/cleaned_training_data.json'
    OUTPUT_DATASET_DIR = 'data/ner_dataset_for_training'
    MODEL_CHECKPOINT = "distilbert-base-cased"

    print(f"Loading tokenizer for '{MODEL_CHECKPOINT}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    format_for_ner(
        cleaned_data_path=INPUT_CLEANED_PATH,
        output_dir=OUTPUT_DATASET_DIR,
        tokenizer=tokenizer
    )

    reloaded_dataset = DatasetDict.load_from_disk(OUTPUT_DATASET_DIR)
    print("\nExample of one processed record from the training set:")
    example = reloaded_dataset['train'][0]
    
    print(f"Tokens (decoded): {tokenizer.convert_ids_to_tokens(example['input_ids'][:50])}")
    print(f"Labels (IDs):     {example['labels'][:50]}")