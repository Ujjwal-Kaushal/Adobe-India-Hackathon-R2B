# 🎯 Adobe India Hackathon – Round 1B Solution

## 📘 Persona-Driven Document Intelligence System

### 🔍 Overview

An intelligent document analysis system designed to **extract, analyze, and rank** relevant sections from PDF documents based on a **persona** and their **job-to-be-done**. The system processes multiple PDFs and provides prioritized insights tailored to the user's role and goal.

### 🏗️ Solution Architecture

The system consists of three primary components:

1. **Ingestion Pipeline** - Extracts structured content (titles, headings, etc.) from PDFs.
2. **Analysis Engine** - Performs semantic and persona-specific relevance analysis.
3. **Output Generator** - Produces a ranked list of key sections with contextual analysis.

### 🌟 Key Features

* 📄 Automatic PDF structure extraction (titles, headings, content)
* 🧠 Semantic understanding of document content
* 🧍 Persona-based relevance scoring
* 🎯 Job-to-be-done focused section selection
* ⚙️ Optimized for CPU-only environments

### 📁 File Structure

```
project-root/
├── Collection 1/              # Sample input/output directory
│   ├── PDFs/                  # Input PDF documents
│   ├── input.json             # Configuration file
│   └── output.json            # Output results
├── extract_outline.py         # PDF outline extraction logic
├── ingestion_logic.py         # PDF processing pipeline
├── analysis_logic.py          # Persona-based analysis engine
├── run_challenge.py          # Main execution script
├── initialize_model.py       # Model setup/download utility
├── requirements.txt          # Python dependencies
└── Dockerfile                # Docker container setup
```

## ⚙️ Installation & Setup

### 🔧 Prerequisites

* Docker
* Python ≥ 3.8

### 🐳 1. Build Docker Image

```bash
docker build -t doc-analyzer .
```

### ▶️ 2. Run the Solution

```bash
docker run --rm -v "$(pwd):/app" doc-analyzer "Collection 1"
```

## 📥 Input / 📤 Output Specification

### ✅ Input Configuration (`input.json`)

```json
{
  "challenge_info": {
    "challenge_id": "unique_identifier",
    "test_case_name": "scenario_name",
    "description": "brief_description"
  },
  "documents": [
    {
      "filename": "document1.pdf",
      "title": "Document Title"
    }
  ],
  "persona": {
    "role": "Persona Role"
  },
  "job_to_be_done": {
    "task": "Specific task description"
  }
}
```

### ✅ Output Format (`output.json`)

```json
{
  "metadata": {
    "input_documents": ["document1.pdf"],
    "persona": "Persona Role",
    "job_to_be_done": "Specific task",
    "processing_timestamp": "ISO-8601 timestamp"
  },
  "extracted_sections": [
    {
      "document": "document1.pdf",
      "section_title": "Relevant Section",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "document1.pdf",
      "refined_text": "Detailed content analysis",
      "page_number": 1
    }
  ]
}
```

## 🧪 Technical Approach

### 📄 Document Processing

* Utilizes **PyMuPDF** for high-speed PDF parsing
* Applies **heuristics** for heading/section detection
* Analyzes **font properties** and layout structure

### 🔍 Content Analysis

* Uses **Sentence-Transformers** for semantic similarity
* Leverages **TinyLLaMA** for contextual embedding
* Computes **persona-job-specific relevance scores**

### ⚡ Performance Optimization

* Multi-file **parallel processing**
* Memory-efficient algorithms
* Works with **CPU-only environments**

## 📏 Constraints Compliance

* ✅ **Model size**: < 1GB (all models)
* ✅ **Execution time**: < 60 seconds
* ✅ **Hardware**: CPU-only (no GPU)
* ✅ **Offline**: No internet required during run-time

## 🧪 Testing Scenarios

The system is validated against multiple document genres with optimized sample counts:

* 📘 **Research Papers/Thesis** (400 samples) - Academic research documents with structured hierarchies
* 📊 **Reports/Documents** (320 samples) - Business and technical reports with formal formatting
* 📚 **E-books/Manuals** (480 samples) - Educational content and instructional materials
* 🎯 **Slides/Presentations** (600 samples) - Presentation decks with visual emphasis
* ⚖️ **Legal Documents** (320 samples) - Legal texts with specialized formatting conventions
* 📝 **Forms/Letters** (200 samples) - Structured forms and correspondence documents

📂 Sample test collections are available in:

* `Collection 1`
* `Collection 2`
* `Collection 3`

## 📄 License

Developed exclusively for **Adobe India Hackathon 2025**. © All rights reserved by the author.
