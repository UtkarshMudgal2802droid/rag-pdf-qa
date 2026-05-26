---
title: PDF Q&A RAG System
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# PDF Question Answering System using RAG

Retrieval-Augmented Generation (RAG) application for PDF documents

## Features
- Extract text from PDF files
- Split text into overlapping chunks
- Create embeddings using SentenceTransformer
- Retrieve relevant chunks using cosine similarity
- Generate answers using QA model
- Detailed process logging

## Technology Stack
- Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)
- QA Model: deepset/roberta-base-squad2
- Similarity: Cosine Similarity (scikit-learn)
- PDF Processing: PyPDF2
- UI: Gradio 4.x

## RAG Pipeline

| Step | Process |
|------|---------|
| 1 | PDF Text Extraction |
| 2 | Text Chunking (500 chars, 50 overlap) |
| 3 | Embedding Creation |
| 4 | Similarity Search |
| 5 | Answer Generation |

## Links
- GitHub: https://github.com/UtkarshMudgal/pdf-qa-rag
- Live Demo: https://huggingface.co/spaces/UtkarshMudgal/pdf-qa-rag

## License
MIT