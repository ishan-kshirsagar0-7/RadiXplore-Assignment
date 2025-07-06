# OLD APPROACH

# The entire purpose of this file is to parse the raw, nested JSON annotation file from a tool like Label Studio and transform it into a 
# clean, simple format ideal for NER training. The output format will be a list of dictionaries, where each dictionary contains the text 
# and a list of its 'PROJECT' entities.

import json
from pathlib import Path

def create_clean_training_data(raw_annotation_path: str, output_path: str):
    print(f"Loading raw annotation data from: {raw_annotation_path}")
    
    raw_path = Path(raw_annotation_path)
    if not raw_path.exists():
        print(f"Error: File not found at {raw_annotation_path}")
        return

    with open(raw_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    cleaned_data = []
    total_project_entities_found = 0

    for record in raw_data:
        try:
            text = record['data']['text']
            
            project_entities = []

            annotations = record['annotations'][0]['result']

            for annotation in annotations:
                if annotation['type'] != 'labels':
                    continue

                label_value = annotation['value']
                if 'labels' in label_value and label_value['labels'][0] == 'PROJECT':
                    
                    start_char = label_value['start']
                    end_char = label_value['end']
                    entity_text = label_value['text']

                    project_entities.append({
                        "start": start_char,
                        "end": end_char,
                        "text": entity_text, 
                        "label": "PROJECT"
                    })
                    total_project_entities_found += 1

            if project_entities:
                cleaned_data.append({
                    "text": text,
                    "entities": project_entities
                })

        except (KeyError, IndexError) as e:
            print(f"Skipping a record due to unexpected format: {e}")
            continue
            
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True) 
    with open(output_p, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4)
    
    print("-" * 50)
    print(f"Processing Complete!")
    print(f"Successfully processed {len(cleaned_data)} documents containing 'PROJECT' entities.")
    print(f"Found a total of {total_project_entities_found} 'PROJECT' annotations.")
    print(f"Cleaned training data saved to: {output_path}")
    print("-" * 50)


if __name__ == '__main__':
    INPUT_JSON_PATH = 'sample-annotations.json' 
    OUTPUT_JSON_PATH = 'data/cleaned_training_data.json'

    create_clean_training_data(
        raw_annotation_path=INPUT_JSON_PATH,
        output_path=OUTPUT_JSON_PATH
    )
    
    # Printing the first clean record to see what it looks like
    # with open(OUTPUT_JSON_PATH, 'r') as f:
    #     data = json.load(f)
    #     if data:
    #         print("\nExample of one cleaned record:")
    #         print(json.dumps(data[0], indent=2))