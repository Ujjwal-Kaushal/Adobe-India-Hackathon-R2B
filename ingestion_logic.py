# ingestion_logic.py

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from extract_outline import PDFOutlineExtractor

# --- CONFIGURATION ---
MODEL_PATH = './models/embedding_model'
MIN_CHUNK_SIZE = 150 # Minimum number of characters for a chunk to be considered valid

def create_hierarchical_chunks(doc, outline, title, pdf_path):
    """
    Creates structured text chunks from a PDF based on its outline.
    Each chunk is associated with the heading it falls under.
    """
    chunks = []
    # Add the title and intro text (before the first heading) as the first chunk
    first_heading_page = outline[0]['page'] if outline else len(doc)
    intro_text = ""
    for page_num in range(first_heading_page):
        intro_text += doc[page_num].get_text() + " "
    
    if intro_text.strip():
        chunks.append({
            'chunk_text': f"Document Title: {title}\n\n{intro_text.strip()}",
            'metadata': {
                'source_pdf': Path(pdf_path).name,
                'page_number': 1,
                'section_title': title,
                'heading_level': 'H1'
            }
        })

    # Process text under each heading
    for i, heading in enumerate(outline):
        start_page = heading['page'] - 1
        
        # Determine the end page for the current section
        if i + 1 < len(outline):
            next_heading = outline[i+1]
            # If next heading is on the same page, text is only on that page
            if next_heading['page'] == heading['page']:
                end_page = start_page
            else:
                end_page = next_heading['page'] - 2 # Text ends on the page before the next heading
        else:
            end_page = len(doc) - 1 # Last heading, so text goes to the end of the doc

        # Extract text for the current section
        section_text = ""
        for page_num in range(start_page, end_page + 1):
            section_text += doc[page_num].get_text() + " "
        
        # Clean the extracted text
        # A simple way to get text "after" a heading is to find the heading and slice the string
        # This is not perfect but works for many documents
        heading_pos = section_text.lower().find(heading['text'].lower())
        if heading_pos != -1:
            text_after_heading = section_text[heading_pos + len(heading['text']):]
        else:
            text_after_heading = section_text

        # Use a regex to remove the heading text itself from the chunk
        cleaned_text = re.sub(re.escape(heading['text']), '', text_after_heading.strip(), flags=re.IGNORECASE).strip()

        if len(cleaned_text) > MIN_CHUNK_SIZE:
            chunks.append({
                'chunk_text': f"Section: {heading['text']}\n\n{cleaned_text}",
                'metadata': {
                    'source_pdf': Path(pdf_path).name,
                    'page_number': heading['page'],
                    'section_title': heading['text'],
                    'heading_level': heading['level']
                }
            })
            
    return chunks

def process_single_pdf(pdf_path: str):
    """
    Fully processes one PDF: extracts outline, creates chunks.
    This function is designed to be run in a separate process.
    """
    try:
        extractor = PDFOutlineExtractor()
        outline_data = extractor.extract_outline(pdf_path)
        
        if not outline_data or not outline_data.get("outline"):
            print(f"  - Warning: Could not extract a valid outline from {Path(pdf_path).name}. Skipping.")
            return pdf_path, []

        with fitz.open(pdf_path) as doc:
            chunks = create_hierarchical_chunks(doc, outline_data['outline'], outline_data['title'], pdf_path)
        
        return pdf_path, chunks
    except Exception as e:
        print(f"  - ERROR processing {Path(pdf_path).name}: {e}")
        return pdf_path, []

def run_ingestion(document_filenames: list[str], input_pdf_dir: str):
    """
    Processes PDFs from a specific directory, generates embeddings, and returns an in-memory "database".
    """
    pdf_paths = [str(Path(input_pdf_dir) / fn) for fn in document_filenames if (Path(input_pdf_dir) / fn).exists()]
    
    if len(pdf_paths) != len(document_filenames):
        print("Warning: Some documents listed in the config file were not found in the PDFs directory.")

    print(f"Starting parallel processing of {len(pdf_paths)} PDF documents...")
    all_chunks = []
    # Use ProcessPoolExecutor to run PDF processing in parallel for speed
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_pdf, p): p for p in pdf_paths}
        for future in as_completed(futures):
            pdf_path, chunks = future.result()
            if chunks: 
                print(f"  - Created {len(chunks)} chunks from {Path(pdf_path).name}")
                all_chunks.extend(chunks)

    if not all_chunks:
        print("CRITICAL: No processable chunks were created from the provided PDFs. Halting.")
        return None
        
    documents = [chunk['chunk_text'] for chunk in all_chunks]
    metadatas = [chunk['metadata'] for chunk in all_chunks]
    
    print(f"Loading embedding model from: {MODEL_PATH}...")
    # This assumes the models are already downloaded and available at MODEL_PATH
    model = SentenceTransformer(MODEL_PATH)
    
    print(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=64)
    
    print("Ingestion complete. Data is ready in memory.")
    
    return {
        "embeddings": np.array(embeddings),
        "documents": documents,
        "metadatas": metadatas
    }