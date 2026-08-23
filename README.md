# Local RAG Assistant using Microsoft Foundry Local

## Project overview

This project is a simple Retrieval-Augmented Generation (RAG) application built for the Microsoft Summer School Foundry Local project.

The assistant reads local documents from the `knowledge_base` folder, retrieves the most relevant document chunks for a user question, and generates an answer using a local language model through Microsoft Foundry Local.

## What the project demonstrates

- Loading local documents
- Splitting text into chunks
- Creating embeddings
- Finding relevant chunks with similarity search
- Generating grounded answers with a local chat model
- Showing which sources were used

## Technologies used

- Python 3.11+
- Microsoft Foundry Local
- `foundry-local-sdk`
- `pypdf`

## Folder structure

```text
Local-RAG-Assistant/
├── main.py
├── rag_engine.py
├── requirements.txt
├── README.md
├── knowledge_base/
│   ├── foundry_local_notes.txt
│   └── summer_school_project.txt
├── presentation/
│   ├── Local_RAG_Presentation.pptx
│   └── presentation_script.md
└── screenshots/
    └── demo_transcript.txt
```

## Setup

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt
```

On Windows, Microsoft recommends the Windows ML package instead:

```bash
pip install foundry-local-sdk-winml pypdf
```

## Run with Foundry Local

```bash
python main.py
```

The first run may download the local embedding and chat models.

## Quick demo mode

If you only want to test the app structure without downloading models, run:

```bash
python main.py --demo
```

Demo mode does not use the local LLM, but it shows the same document loading and retrieval flow.

## Presentation

The presentation file is included in:

```text
presentation/Local_RAG_Presentation.pptx
```


## Author

Yazan Itani 
Eastern Mediterranean University 
Artificial Intelligence Engineering  
Microsoft Summer School Project
