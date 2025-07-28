# run_challenge.py

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import argparse
from pathlib import Path
from ingestion_logic import run_ingestion
from analysis_logic import run_analysis
from initialize_model import download_models, MODELS_DIR

def main():
    parser = argparse.ArgumentParser(description="Run a specific document analysis collection.")
    parser.add_argument("collection_name", type=str, help="The name of the collection directory to process (e.g., 'Collection 1').")
    args = parser.parse_args()

    # --- Pre-computation Step: Check for models ---
    if not os.path.exists(MODELS_DIR):
        print("Models directory not found. Running model initializer...")
        download_models()
    else:
        print("Models directory found. Skipping download.")

    collection_path = Path(args.collection_name)
    # UPDATED: Look for 'input.json' as the configuration file
    config_path = collection_path / "input.json"
    input_pdf_dir = collection_path / "PDFs"
    # UPDATED: Save results to 'output.json' to mirror the input file name
    output_file_path = collection_path / "output.json"

    if not config_path.is_file():
        print(f"Error: Configuration file not found at '{config_path.resolve()}'")
        print(f"Please ensure the collection '{args.collection_name}' exists and contains 'input.json'.")
        return
        
    print(f"Loading configuration from: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    challenge_id = config.get("challenge_info", {}).get("challenge_id", "default_challenge")
    documents_to_process = [doc["filename"] for doc in config.get("documents", [])]
    persona_info = config.get("persona", {})
    job_info = config.get("job_to_be_done", {})

    if not all([documents_to_process, persona_info, job_info]):
        print("Error: The JSON config file is missing required keys (documents, persona, job_to_be_done).")
        return

    print(f"\n--- Phase 1: Ingesting Documents from '{input_pdf_dir}' ---")
    in_memory_db = run_ingestion(documents_to_process, input_pdf_dir=str(input_pdf_dir))
    print("--- Ingestion Complete ---")
    
    if not in_memory_db:
        print("Halting pipeline due to ingestion failure.")
        return

    print("\n--- Phase 2: Analyzing Documents for Persona and Job ---")
    run_analysis(
        challenge_id=challenge_id,
        persona_dict=persona_info,
        job_dict=job_info,
        document_list=config.get("documents", []),
        in_memory_db=in_memory_db,
        output_file_path=str(output_file_path)
    )
    print("--- Analysis Complete ---")
    print(f"\nChallenge '{challenge_id}' in collection '{args.collection_name}' finished successfully.")

if __name__ == "__main__":
    main()