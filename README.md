# PDF QA Bot

This Streamlit app allows you to upload a PDF and ask questions about its content using **LangChain** and **Groq's Llama 3.3 70B**.

---

## 🚀 Features
- **PDF Upload**: Upload any PDF document.  
- **Text Chunking**: Splits the PDF into manageable chunks for processing.  
- **Embeddings**: Uses HuggingFace embeddings for semantic search.  
- **Vector Store**: Stores document chunks in a Chroma vector database.  
- **Retrieval QA**: Answers your questions using Groq's Llama 3.3 70B model and retrieves relevant document sources.  

---

## 📦 Requirements
- Python 3.11+  
- Streamlit  
- langchain  
- langchain-community  
- langchain-core
- langchain-text-splitters  
- langchain-groq
- sentence-transformers  
- ChromaDB  
- pypdf
- Groq API key (enter via the app interface)  

---

## 🛠 Usage

### 1. Install dependencies
```bash
pip install streamlit langchain langchain-community langchain-text-splitters langchain-core langchain-groq sentence-transformers chromadb pypdf
```

### 2. Run the app
```bash
streamlit run new.py
```

### 3. Enter your Groq API key
Get a free API key from [console.groq.com](https://console.groq.com) and enter it in the app interface.

### 4. Upload a PDF and ask questions!

The app will process the PDF, create embeddings, and allow you to query it using AI.

---

## 📖 Code Overview

- Loads and splits PDF into chunks using `RecursiveCharacterTextSplitter`
- Embeds chunks using HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model
- Stores embeddings in Chroma vector database
- Uses Groq's `llama-3.3-70b-versatile` model for question answering
- Retrieves relevant context and generates accurate answers

---

## 🔑 Getting a Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste it into the app

---

## ⚙️ Model Information

**Current Model**: `llama-3.3-70b-versatile`  
**Provider**: Groq  
**Temperature**: 0.2 (for consistent, factual responses)

---

**Note**: Make sure all dependencies are installed before running the app.
