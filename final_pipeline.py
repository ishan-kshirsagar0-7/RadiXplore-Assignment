# This is the actual final pipeline, that :
# Takes input PDF -> Extracts text and passes it to the NER model -> NER model generates a temporary JSONL file -> LLM populates the 
# coordinates key in this JSONL -> saves the final JSONL to directory.
# The second function takes input a whole directory of multiple PDFs and follows along the same workflow as mentioned above.

import os
import json
import time
import tqdm
from pathlib import Path
from dotenv import load_dotenv
from ner_pipeline import run_extraction_pipeline
from llm_coordinates import extract_locations_from_projects
load_dotenv()

NER_MODEL_REPO_ID = "unpairedelectron07/radixplore-ner-multiclass"
OUTPUT_DIR = "output"
PDF_DIRECTORY = "reports"
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "All_Reports_Combined.jsonl")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_ner")

def process_single_pdf_full_pipeline(pdf_path):
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        print(f"\n[!] Error: PDF file not found at '{pdf_path}'")
        return

    pdf_filename = pdf_path_obj.name
    filename = pdf_filename.split(".")[0]
    output_jsonl_path = Path(OUTPUT_DIR) / f"{filename}.jsonl"
    temp_ner_output_path = Path(OUTPUT_DIR) / f"temp_{pdf_filename}_ner_output.jsonl"
    
    print("-" * 60)
    print(f"Starting pipeline for: {pdf_filename}")
    print("-" * 60)

    print("\n[Step 1/4] Running NER extraction...")
    s1 = time.time()
    run_extraction_pipeline(
        model_repo_id=NER_MODEL_REPO_ID,
        pdf_dir=str(pdf_path_obj.parent),
        output_file=str(temp_ner_output_path),
        pdf_filename_filter=pdf_filename
    )
    e1 = time.time()
    print(f"--> NER extraction complete. Time: {round(e1 - s1, 2)}s")
    
    print("\n[Step 2/4] Preparing de-duplicated project list for LLM...")
    s2 = time.time()
    try:
        with open(temp_ner_output_path, 'r', encoding='utf-8') as f:
            ner_records = [json.loads(line) for line in f]
        
        if not ner_records:
            print("--> No projects were extracted by the NER model. Pipeline finished for this file.")
            if os.path.exists(temp_ner_output_path):
                os.remove(temp_ner_output_path)
            return

        unique_project_names = sorted(list({record["project_name"] for record in ner_records}))
        print(f"--> Found {len(ner_records)} mentions, with {len(unique_project_names)} unique names.")
    except FileNotFoundError:
        print("--> NER step did not produce an output file. Skipping geolocation.")
        return
    e2 = time.time()
    print(f"--> LLM input preparation complete. Time: {round(e2 - s2, 2)}s")

    print("\n[Step 3/4] Calling LLM for geolocation...")
    s3 = time.time()
    location_groups_data = extract_locations_from_projects(unique_project_names)
    e3 = time.time()
    print(f"--> LLM processing complete. Time: {round(e3 - s3, 2)}s")

    print("\n[Step 4/4] Building lookup map and writing final JSONL file...")
    s4 = time.time()
    coordinate_lookup_map = {}
    if location_groups_data and 'mappings' in location_groups_data:
        for group in location_groups_data['mappings']:
            location_info = group.get('location')
            coords = None
            if location_info:
                lat = location_info.get('latitude')
                lon = location_info.get('longitude')
                if lat is not None and lon is not None and (lat != 0 or lon != 0):
                    coords = [lat, lon]
            
            all_names_for_group = [group.get('canonical_name')] + group.get('aliases', [])
            for name in all_names_for_group:
                if name:
                    coordinate_lookup_map[name] = coords
    
    print(f"--> Lookup map built with {len(coordinate_lookup_map)} mappings.")

    with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        for record in ner_records:
            record['coordinates'] = coordinate_lookup_map.get(record['project_name'])
            f_out.write(json.dumps(record) + '\n')
            
    e4 = time.time()
    print(f"--> Final file populated. Time: {round(e4 - s4, 2)}s")
    print(f"\nFinal populated file saved to: {output_jsonl_path}")

    if os.path.exists(temp_ner_output_path):
        os.remove(temp_ner_output_path)
        print("--> Temporary NER output file removed.")


def full_directory_pipeline():
    print("\n[STAGE 1/3] Starting Batch NER Extraction for all PDFs...")
    s1 = time.time()
    
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    all_ner_records = []
    
    pdf_files = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith(".pdf")]
    
    for filename in pdf_files:
        print(f"\n  -> Processing NER for: {filename}")
        temp_output_path = os.path.join(TEMP_DIR, f"{filename}.jsonl")
        
        run_extraction_pipeline(
            model_repo_id=NER_MODEL_REPO_ID,
            pdf_dir=PDF_DIRECTORY,
            output_file=temp_output_path,
            pdf_filename_filter=filename
        )
        
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            all_ner_records.extend(records)
        
        os.remove(temp_output_path)
    
    os.rmdir(TEMP_DIR)
    
    e1 = time.time()
    print(f"\n--> STAGE 1 COMPLETE. Extracted {len(all_ner_records)} total mentions. Time: {round(e1 - s1, 2)}s")

    print("\n[STAGE 2/3] Preparing unique project list and calling LLM...")
    s2 = time.time()
    
    if not all_ner_records:
        print("--> No projects found in any documents. Pipeline stopping.")
        return

    unique_project_names = sorted(list({record["project_name"] for record in all_ner_records}))
    print(f"--> Found {len(unique_project_names)} unique project names across all documents.")

    location_groups_data = extract_locations_from_projects(unique_project_names)
    
    e2 = time.time()
    print(f"--> STAGE 2 COMPLETE. LLM processing finished. Time: {round(e2 - s2, 2)}s")

    print("\n[STAGE 3/3] Building lookup map and writing final submission file...")
    s3 = time.time()
    
    coordinate_lookup_map = {}
    if location_groups_data and 'mappings' in location_groups_data:
        for group in location_groups_data['mappings']:
            location_info = group.get('location')
            coords = None
            if location_info:
                lat = location_info.get('latitude')
                lon = location_info.get('longitude')
                if lat is not None and lon is not None and (lat != 0 or lon != 0):
                    coords = [lat, lon]
            
            all_names_for_group = [group.get('canonical_name')] + group.get('aliases', [])
            for name in all_names_for_group:
                if name:
                    coordinate_lookup_map[name] = coords
    
    print(f"--> Lookup map built with {len(coordinate_lookup_map)} mappings.")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for record in tqdm.tqdm(all_ner_records, desc="Populating final records"):
            record['coordinates'] = coordinate_lookup_map.get(record['project_name'])
            f_out.write(json.dumps(record) + '\n')
            
    e3 = time.time()
    print(f"--> STAGE 3 COMPLETE. Final file populated. Time: {round(e3 - s3, 2)}s")


if __name__ == "__main__":
    print("======================================================")
    print("      STARTING FULL DIRECTORY BATCH PIPELINE      ")
    print("======================================================")
    
    pipeline_start_time = time.time()
    
    full_directory_pipeline()
    
    pipeline_end_time = time.time()
    
    print("\n======================================================")
    print("      ENTIRE PIPELINE FINISHED      ")
    print(f"      Final deliverable saved to: {FINAL_OUTPUT_FILE}")
    print(f"      Total execution time: {round(pipeline_end_time - pipeline_start_time, 2)} seconds.")
    print("======================================================")


# if __name__ == "__main__":

#     PDF_DIRECTORY = "reports"
#     PDF_TO_PROCESS = "Report_5.pdf"
    
#     full_pdf_path = os.path.join(PDF_DIRECTORY, PDF_TO_PROCESS)

#     print("==============================================")
#     print("      STARTING FULL PDF-TO-COORDINATES PIPELINE      ")
#     print("==============================================")
    
#     pipeline_start_time = time.time()
    
#     process_single_pdf_full_pipeline(full_pdf_path)
    
#     pipeline_end_time = time.time()
    
#     print("\n----------------------------------------------")
#     print(f"      PIPELINE RUN FOR '{PDF_TO_PROCESS}' FINISHED      ")
#     print(f"      Total execution time: {round(pipeline_end_time - pipeline_start_time, 2)} seconds.")
#     print("==============================================")