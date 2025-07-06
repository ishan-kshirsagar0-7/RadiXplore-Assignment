# To scan the entire raw annotation JSON file to identify all unique entity labels and their frequencies.

import json
from pathlib import Path
from collections import Counter

def analyze_annotation_labels(raw_annotation_path: str):
    print(f"Analyzing labels in: {raw_annotation_path}")
    
    raw_path = Path(raw_annotation_path)
    if not raw_path.exists():
        print(f"Error: File not found at {raw_annotation_path}")
        return

    label_counts = Counter()
    total_annotations_found = 0
    
    try:
        with open(raw_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return

    for i, record in enumerate(raw_data):
        try:
            for annotation_set in record.get('annotations', []):
                for annotation in annotation_set.get('result', []):
                    if annotation.get('type') == 'labels':
                        value = annotation.get('value', {})
                        labels_list = value.get('labels')
                        if labels_list and isinstance(labels_list, list) and len(labels_list) > 0:
                            label = labels_list[0]
                            label_counts[label] += 1
                            total_annotations_found += 1
        except (TypeError, KeyError) as e:
            print(f"Warning: Could not process a record #{i}. It might have an unusual format. Error: {e}")
            continue

    print("\n--- Label Analysis Report ---")
    if not label_counts:
        print("No labels found. The file might be empty or in an unexpected format.")
        return
        
    print(f"Found {total_annotations_found} total label annotations across {i+1} documents.")
    print(f"Discovered {len(label_counts)} unique labels:")
    
    for label, count in sorted(label_counts.items()):
        print(f"  - {label}: {count} occurrences")
    print("----------------------------")


if __name__ == '__main__':
    FULL_ANNOTATION_FILE = 'sample-annotations.json'

    analyze_annotation_labels(FULL_ANNOTATION_FILE)