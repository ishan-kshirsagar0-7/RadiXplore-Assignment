# NEW APPROACH

# The entire purpose of this file is to parse the raw, nested JSON annotation file from a tool like Label Studio and transform it into a 
# clean, multli-class format ideal for NER training. The output format will be a list of dictionaries, where each dictionary contains the 
# text and a list of all discovered labels.

import json
from pathlib import Path
from collections import Counter

def create_clean_training_data(raw_annotation_path: str, output_path: str):
    TARGET_LABELS = {"PROJECT", "ORGANIZATION", "LOCATION", "PROSPECT"}
    
    print(f"Loading raw annotation data from: {raw_annotation_path}")
    
    raw_path = Path(raw_annotation_path)
    if not raw_path.exists():
        print(f"Error: File not found at {raw_annotation_path}")
        return

    with open(raw_path, 'r', encoding='utf-8') as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse JSON file. {e}")
            return

    cleaned_data = []
    entity_counter = Counter()

    for record in raw_data:
        try:
            text = record['data']['text']
            entities = []

            for annotation_set in record.get('annotations', []):
                for annotation in annotation_set.get('result', []):
                    if annotation.get('type') != 'labels':
                        continue

                    label_value = annotation.get('value', {})
                    if 'labels' in label_value and label_value['labels'] and label_value['labels'][0] in TARGET_LABELS:
                        
                        label = label_value['labels'][0]
                        
                        entities.append({
                            "start": label_value['start'],
                            "end": label_value['end'],
                            "label": label 
                        })
                        entity_counter.update([label])

            if entities:
                cleaned_data.append({
                    "text": text,
                    "entities": entities
                })

        except (KeyError, IndexError) as e:
            print(f"Skipping a record due to unexpected format: {e}")
            continue
            
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    with open(output_p, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print("-" * 50)
    print("Processing Complete!")
    print(f"Successfully processed {len(cleaned_data)} documents.")
    print("Found and extracted the following entity counts:")
    for label, count in sorted(entity_counter.items()):
        print(f"  - {label}: {count}")
    print(f"Cleaned multi-class training data saved to: {output_path}")
    print("-" * 50)


if __name__ == '__main__':
    INPUT_JSON_PATH = 'sample-annotations.json'
    OUTPUT_JSON_PATH = 'data/cleaned_training_data_multiclass.json'

    create_clean_training_data(
        raw_annotation_path=INPUT_JSON_PATH,
        output_path=OUTPUT_JSON_PATH
    )

    # Printing the first record of the output file
    print("\n--- Verifying first record of the output file ---")
    output_file = Path(OUTPUT_JSON_PATH)
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data:
                print(json.dumps(data[0], indent=2))
            else:
                print("Output file is empty.")
    else:
        print("Output file was not created.")