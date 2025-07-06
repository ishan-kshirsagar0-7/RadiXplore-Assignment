# This is the NER Formatter for my new approach of training the model on annotations with all labels.
# Check out - ner_formatter.py - for details on what we're achieving with this step.

import json
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from tqdm import tqdm

IGNORED_LABEL_ID = -100 

def format_for_ner_multiclass(cleaned_data_path: str, output_dir: str, tokenizer: PreTrainedTokenizer, label_names: list):
    print(f"Loading cleaned data from: {cleaned_data_path}")
    with open(cleaned_data_path, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)

    label_to_id = {label: i for i, label in enumerate(label_names)}

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    print("Tokenizing and aligning multi-class labels...")
    for record in tqdm(cleaned_data):
        text = record['text']
        entities = record['entities']

        tokenized_inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True
        )
        
        labels = [label_to_id["O"]] * len(tokenized_inputs['input_ids'])
        offset_mapping = tokenized_inputs.pop("offset_mapping")

        for entity in entities:
            start_char = entity['start']
            end_char = entity['end']
            entity_label = entity['label']

            b_label_id = label_to_id.get(f"B-{entity_label}")
            i_label_id = label_to_id.get(f"I-{entity_label}")

            if b_label_id is None or i_label_id is None:
                continue

            token_start_index = None
            token_end_index = None

            for i, (offset_start, offset_end) in enumerate(offset_mapping):
                if offset_end == 0: continue
                if max(offset_start, start_char) < min(offset_end, end_char):
                    if token_start_index is None:
                        token_start_index = i
                    token_end_index = i
            
            if token_start_index is not None and token_end_index is not None:
                labels[token_start_index] = b_label_id
                for i in range(token_start_index + 1, token_end_index + 1):
                    labels[i] = i_label_id

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
    print("Multi-class dataset creation complete!")
    print(f"Formatted dataset saved to: {output_dir}")
    print("-" * 50)


if __name__ == '__main__':
    INPUT_CLEANED_PATH = 'data/cleaned_training_data_multiclass.json'
    OUTPUT_DATASET_DIR = 'data/ner_dataset_multiclass'
    MODEL_CHECKPOINT = "distilbert-base-cased"

    entity_types = ["PROJECT", "ORGANIZATION", "LOCATION", "PROSPECT"]
    label_names = ["O"]
    for entity in entity_types:
        label_names.append(f"B-{entity}")
        label_names.append(f"I-{entity}")

    print("Defined Label Schema:")
    for i, name in enumerate(label_names):
        print(f"  {i}: {name}")
    print("-" * 50)
    
    print(f"Loading tokenizer for '{MODEL_CHECKPOINT}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    format_for_ner_multiclass(
        cleaned_data_path=INPUT_CLEANED_PATH,
        output_dir=OUTPUT_DATASET_DIR,
        tokenizer=tokenizer,
        label_names=label_names
    )

    reloaded_dataset = DatasetDict.load_from_disk(OUTPUT_DATASET_DIR)
    print("\nExample of one processed record from the training set:")
    example = reloaded_dataset['train'][0]
    
    print(f"Tokens (decoded): {tokenizer.convert_ids_to_tokens(example['input_ids'][:50])}")
    print(f"Labels (IDs):     {example['labels'][:50]}")
    decoded_labels = [label_names[l] if l != -100 else "IGNORE" for l in example['labels'][:50]]
    print(f"Labels (decoded): {decoded_labels}")