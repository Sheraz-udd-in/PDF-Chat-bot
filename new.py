# app.py
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Prompt template and chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

st.set_page_config(page_title="PDF QA Bot")
st.title("📄 PDF QA Bot")

# --- API key input ---
api_key = st.text_input("Enter your Groq API Key:", type="password", help="Get a key from console.groq.com")
if api_key:
    os.environ["GROQ_API_KEY"] = api_key

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def build_qa_chain(pdf_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Chunk / Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings (local sentence-transformers)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma in-memory vector DB
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)

    # Retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Groq LLM (reads GROQ_API_KEY from env)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # Updated to current model
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Simple RAG: format documents and query into a single prompt
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return retriever, llm

if uploaded_file:
    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ Please enter your Groq API Key above before asking.")
    else:
        # Save uploaded PDF to a temporary file so loaders can access a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            st.info("Indexing document (this may take a few seconds)...")
            retriever, llm = build_qa_chain(tmp_path)

            query = st.text_input("Ask a question about the document:", value="What is this paper about?")
            if st.button("Ask"):
                with st.spinner("Generating answer..."):
                    # Retrieve relevant documents
                    docs = retriever.invoke(query)
                    
                    # Format context
                    context = "\n\n".join(doc.page_content for doc in docs)
                    
                    # Create prompt and get answer
                    prompt = f"""Answer the question based on the following context:

{context}

Question: {query}

Answer:"""
                    
                    answer = llm.invoke(prompt)
                    st.subheader("Answer:")
                    st.write(answer.content)

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            # cleanup temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
