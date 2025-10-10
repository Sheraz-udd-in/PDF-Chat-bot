# PDF QA Bot

This Streamlit app allows you to upload a PDF and ask questions about its content using **LangChain** and **GeminiPro**.

---

## 🚀 Features
- **PDF Upload**: Upload any PDF document.  
- **Text Chunking**: Splits the PDF into manageable chunks for processing.  
- **Embeddings**: Uses HuggingFace embeddings for semantic search.  
- **Vector Store**: Stores document chunks in a Chroma vector database.  
- **Retrieval QA**: Answers your questions using GeminiPro LLM and retrieves relevant document sources.  

---

## 📦 Requirements
- Python 3.8+  
- Streamlit  
- langchain  
- langchain_community  
- langchain_text_splitters  
- sentence-transformers  
- ChromaDB  
- GeminiPro API key (set as `GOOGLE_API_KEY` environment variable)  

---

## 🛠 Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Set your GeminiPro API key
```bash
export GOOGLE_API_KEY="your_api_key_here"   # Linux / macOS
set GOOGLE_API_KEY=your_api_key_here        # Windows (CMD)
$env:GOOGLE_API_KEY="your_api_key_here"     # Windows (PowerShell)
```
### 3. Run the app
```bash
streamlit run app.py
```
### 4. Upload a PDF and ask questions!

The app will process the PDF, create embeddings, and allow you to query it.

## 📖 Code Overview

Loads and splits PDF into chunks.

Embeds chunks and stores them in Chroma vector DB.

Uses GeminiPro LLM for question answering.

Displays answers and source documents.
