import os
import json
from pathlib import Path
import fitz 
from transformers import pipeline
from sentence_splitter import split_text_into_sentences
import torch

# This function uses our own fine-tuned NER model from HFHub to extract project names from PDF files and generates a JSONL output.
def run_extraction_pipeline(model_repo_id, pdf_dir, output_file, pdf_filename_filter=None):
    print(f"Loading NER model '{model_repo_id}' from Hugging Face Hub...")
    device = 0 if torch.cuda.is_available() else -1
    ner_pipeline = pipeline(
        "token-classification", 
        model=model_repo_id, 
        device=device, 
        aggregation_strategy="simple"
    )
    print(f"NER model loaded successfully on device: {'cuda' if device == 0 else 'cpu'}.")

    try:
        all_pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    except FileNotFoundError:
        print(f"Error: PDF directory '{pdf_dir}' not found.")
        return
    
    if pdf_filename_filter:
        pdf_files_to_process = [f for f in all_pdf_files if f == pdf_filename_filter]
        if not pdf_files_to_process:
            print(f"Warning: Filter '{pdf_filename_filter}' did not match any files in '{pdf_dir}'.")
            return
    else:
        pdf_files_to_process = all_pdf_files

    print(f"Found {len(pdf_files_to_process)} PDF(s) to process.")

    output_records = []
    
    # A set of generic or junk terms to filter out from the results
    JUNK_TERMS = {"project", "Project", "prospect", "Prospect", "mine", "Mine"}

    for filename in pdf_files_to_process:
        print(f"\n--- Processing: {filename} ---")
        filepath = os.path.join(pdf_dir, filename)
        
        try:
            with fitz.open(filepath) as doc:
                for page_num, page in enumerate(doc, 1):
                    page_text = page.get_text()
                    if not page_text.strip():
                        continue
                    
                    entities = ner_pipeline(page_text)
                    
                    project_entities = [e for e in entities if e['entity_group'] == 'PROJECT']

                    if not project_entities:
                        continue
                    
                    print(f"  Page {page_num}: Found {len(project_entities)} potential project mention(s).")
                    
                    sentences = split_text_into_sentences(text=page_text, language='en')

                    for entity in project_entities:
                        project_name = entity['word'].strip()

                        if len(project_name) <= 2 or project_name in JUNK_TERMS or project_name.startswith("##"):
                            continue

                        context = ""
                        for sentence in sentences:
                            if project_name in sentence:
                                context = sentence.strip().replace('\n', ' ')
                                break
                        
                        record = {
                            "pdf_file": filename,
                            "page_number": page_num,
                            "project_name": project_name,
                            "context_sentence": context,
                            "coordinates": None,
                            "ner_confidence": round(float(entity['score']), 4)
                        }
                        output_records.append(record)

        except Exception as e:
            print(f"An error occurred while processing {filename}, page {page_num}: {e}")

    output_p = Path(output_file)
    output_p.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_p, 'w', encoding='utf-8') as f:
        for record in output_records:
            f.write(json.dumps(record) + '\n')

    print(f"\nExtraction complete. Saved {len(output_records)} records to {output_file}")


if __name__ == '__main__':
    MODEL_REPO_ID = "unpairedelectron07/radixplore-ner-multiclass"
    PDF_DIRECTORY = "reports"
    OUTPUT_JSONL_FILE = "output/all_reports_extraction.jsonl"
    
    print("==============================================")
    print("      RUNNING NER EXTRACTION ON ALL PDFs      ")
    print("==============================================")
    
    if not os.path.exists(PDF_DIRECTORY) or not os.listdir(PDF_DIRECTORY):
        print(f"Error: PDF directory '{PDF_DIRECTORY}' is empty or does not exist.")
    else:
        run_extraction_pipeline(
            model_repo_id=MODEL_REPO_ID,
            pdf_dir=PDF_DIRECTORY,
            output_file=OUTPUT_JSONL_FILE
        )
    
    print("==============================================")
    print("      STANDALONE RUN FINISHED      ")
    print("==============================================")