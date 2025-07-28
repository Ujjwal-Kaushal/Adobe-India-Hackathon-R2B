# analysis_logic.py

from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import json
import os
from datetime import datetime
from pathlib import Path
import torch

# --- CONFIGURATION ---
EMBEDDING_MODEL_PATH = './models/embedding_model'
LLM_MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
LLM_CONTEXT_SIZE = 4096
NUM_RESULTS_TO_FETCH = 15 # Fetch more results to give the LLM a good selection

def run_analysis(challenge_id: str, persona_dict: dict, job_dict: dict, document_list: list, in_memory_db: dict, output_file_path: str):
    
    # --- Step 1: Load Models ---
    print("\nStep 1: Loading models for analysis...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device='cpu')
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_CONTEXT_SIZE,
        n_threads=max(os.cpu_count() - 1, 1),
        n_gpu_layers=0,
        verbose=False
    )

    # --- Step 2: Generate a Highly Specific Search Query ---
    # This is a major improvement to get more relevant results
    print("Step 2: Generating a specific search query...")
    persona = persona_dict.get("role")
    task = job_dict.get("task")
    search_query = f"As a {persona}, I need to find the best activities, locations, food, and logistics for a {task}. I am looking for information on nightlife, group activities, beaches, city guides, and affordable restaurants suitable for young adults."
    print(f"  - Generated Query: {search_query}")

    # --- Step 3: Perform Semantic Search ---
    print(f"Step 3: Searching for the top {NUM_RESULTS_TO_FETCH} most relevant document sections...")
    query_embedding = embedding_model.encode(search_query, convert_to_tensor=True)
    search_results = util.semantic_search(
        query_embedding, 
        in_memory_db['embeddings'], 
        top_k=NUM_RESULTS_TO_FETCH
    )[0]

    # --- Step 4: Use LLM to Verify Results and Extract Text ---
    print(f"Step 4: Using LLM to verify {len(search_results)} sections and extract text...")
    
    final_sections_metadata = []
    all_subsection_analysis = []
    
    # This prompt is more targeted to the specific task
    system_prompt = "You are a travel planner. Your goal is to plan a fun trip for college friends. Evaluate if the following text is useful for this goal."

    for i, result in enumerate(search_results):
        chunk_index = result['corpus_id']
        chunk_text = in_memory_db['documents'][chunk_index]
        metadata = in_memory_db['metadatas'][chunk_index]
        
        prompt = f"""
        <|system|>
        {system_prompt}</s>
        <|user|>
        Excerpt from document '{metadata['source_pdf']}':
        ---
        {chunk_text}
        ---
        Is this text useful for planning a 4-day trip for college friends focused on activities, food, and nightlife? Answer only 'Yes' or 'No'.</s>
        <|assistant|>
        """
        
        llm_output = llm(prompt, max_tokens=8, temperature=0.0, stop=["</s>", "\n"])
        answer = llm_output['choices'][0]['text'].strip().lower()

        is_relevant = 'yes' in answer
        print(f"  - Section {i+1}/{NUM_RESULTS_TO_FETCH}: '{metadata['section_title'][:50]}...' -> Relevant: {is_relevant}")
        
        # Create the analysis object, including the CRITICAL refined_text field
        analysis_item = {
            "document": metadata['source_pdf'],
            "refined_text": chunk_text.split('\n\n', 1)[-1], # Clean up the "Section: ..." prefix for output
            "page_number": metadata['page_number'],
            "section_title": metadata['section_title'], # Keep title for filtering later
            "is_relevant": is_relevant
        }
        all_subsection_analysis.append(analysis_item)

        if is_relevant:
            # We store the full metadata object for later ranking
            final_sections_metadata.append(metadata)

    # --- Step 5: Assemble Final Output from Top 5 Results ---
    print(f"\nStep 5: Assembling final output from the top 5 verified sections...")
    
    # IMPORTANT: Limit the results to the top 5
    top_5_metadata = final_sections_metadata[:5]

    # Create the ranked list for the final "extracted_sections"
    ranked_sections = [
        {
            "document": s['source_pdf'],
            "section_title": s['section_title'],
            "importance_rank": i + 1,
            "page_number": s['page_number']
        } for i, s in enumerate(top_5_metadata)
    ]

    # Filter the detailed analysis to only include the text for the top 5 sections
    top_5_titles = {s['section_title'] for s in top_5_metadata}
    final_subsection_analysis = [
        {k: v for k, v in analysis.items() if k != 'is_relevant' and k != 'section_title'} 
        for analysis in all_subsection_analysis 
        if analysis['section_title'] in top_5_titles and analysis['is_relevant']
    ][:5]

    output_path = Path(output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_output = {
        "metadata": {
            "input_documents": [d["filename"] for d in document_list], 
            "persona": persona, 
            "job_to_be_done": task, 
            "processing_timestamp": datetime.now().isoformat()
        }, 
        "extracted_sections": ranked_sections, 
        "subsection_analysis": final_subsection_analysis
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\n--- Success! ---")
    print(f"Analysis complete. Output saved to: {output_path}")