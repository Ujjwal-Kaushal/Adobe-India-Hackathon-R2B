# initialize_model.py

import os
import shutil
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_HF_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
LLM_HF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

MODELS_DIR = "./models"
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "embedding_model")
LLM_MODEL_PATH = os.path.join(MODELS_DIR, LLM_HF_FILENAME)

def download_models():
    """
    Downloads and saves the sentence transformer and LLM models locally.
    Includes error handling to clean up corrupted downloads automatically.
    """
    print("--- Model Initializer ---")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Models will be stored in: {os.path.abspath(MODELS_DIR)}")

    # 1. Download and save the Sentence Transformer model
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        print(f"\nDownloading embedding model: '{EMBEDDING_MODEL_NAME}'...")
        try:
            # Attempt to download and save the model
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            model.save(EMBEDDING_MODEL_PATH)
            print(f"Embedding model saved successfully to: {EMBEDDING_MODEL_PATH}")
        except Exception as e:
            # This block runs if the download or save fails
            print(f"\n--- ERROR ---")
            print(f"Failed to download the embedding model. The error was: {e}")
            print("This is often caused by a network interruption.")
            
            # IMPORTANT: Clean up the corrupted directory if it was partially created
            if os.path.exists(EMBEDDING_MODEL_PATH):
                print(f"Deleting potentially corrupted directory: {EMBEDDING_MODEL_PATH}")
                shutil.rmtree(EMBEDDING_MODEL_PATH)
                print("Cleanup complete.")
            
            print("\nPlease check your internet connection and try running the script again.")
            # Stop the script to prevent further errors
            raise
    else:
        print(f"\nEmbedding model already exists at: {EMBEDDING_MODEL_PATH}")

    # 2. Download the Llama GGUF model from Hugging Face Hub
    if not os.path.exists(LLM_MODEL_PATH):
        print(f"\nDownloading LLM: '{LLM_HF_FILENAME}' from '{LLM_HF_REPO}'...")
        # Note: hf_hub_download is generally more resilient to interruptions
        hf_hub_download(
            repo_id=LLM_HF_REPO,
            filename=LLM_HF_FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        print(f"LLM saved to: {LLM_MODEL_PATH}")
    else:
        print(f"\nLLM already exists at: {LLM_MODEL_PATH}")

    print("\n--- Model initialization complete! ---")

if __name__ == "__main__":
    download_models()