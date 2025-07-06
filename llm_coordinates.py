# Here I created a function that extracts coordinates from given JSONL inputs, using Gemini API's two features:
# 1. Grounding on Google Search
# 2. Structured Output
# I tried to achieve both of these things in a single api call, but it turns out that Gemini doesn't support structured output on tool 
# calling just yet, so I had to set up a faster, lighter gemini model (gemini-2.5-flash-lite-preview-06-17) to simply deserialize the
# tool calling output string into JSON.

import os
import json
import time
from schemas import ListOfLocations
from prompts import extract_location_prompt, structured_response_prompt
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
LLM = "gemini-2.5-flash"
LLM2 = "gemini-2.5-flash-lite-preview-06-17"

grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

config = types.GenerateContentConfig(
    tools=[grounding_tool],
)

def extract_locations_from_projects(project_list):
    s1 = time.time()
    mapping_output = client.models.generate_content(
        model=LLM,
        contents=f"{extract_location_prompt}\n{project_list}",
        config=config
    ).text
    e1 = time.time()
    print(f"DONE WITH INITIAL MAPPING")
    print(f"Time Taken: {round(e1-s1)} seconds.")

    s2 = time.time()
    structured_mapping = client.models.generate_content(
        model=LLM2,
        contents=f"{structured_response_prompt}\n{mapping_output}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ListOfLocations
        )
    ).text
    e2 = time.time()
    print(f"DONE WITH STRUCTURED MAPPING")
    print(f"Time Taken: {round(e2-s2)} seconds.")

    mappings = json.loads(structured_mapping)

    return mappings


if __name__ == "__main__":

    INPUT_NER_FILE = "output/single_report_extraction.jsonl"
    
    print(f"Reading NER results from: {INPUT_NER_FILE}")
    with open(INPUT_NER_FILE, 'r', encoding='utf-8') as f:
        ner_records = [json.loads(line) for line in f]

    unique_records = sorted(list({record["project_name"] for record in ner_records}))
    
    print(f"\n-------------- PREPARING INPUT FOR LLM --------------")
    print(f"Found {len(ner_records)} total mentions in the input file.")
    print(f"Created a de-duplicated list of {len(unique_records)} unique project names.")

    print("\n--- STARTING LLM PROCESSING ---")
    start_time = time.time()
    
    retrieved_locations = extract_locations_from_projects(unique_records)
    
    end_time = time.time()
    print("\n--- LLM PROCESSING COMPLETE ---")
    
    print("\nLLM Output:")
    print(json.dumps(retrieved_locations, indent=2)) 
    print(f"\nTotal Time Taken for LLM calls: {round(end_time - start_time)} seconds.")