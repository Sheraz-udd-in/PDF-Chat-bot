import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import GeminiPro  # Correct import for GeminiPro
import os
import tempfile

st.set_page_config(page_title="PDF QA Bot")

st.title("📄 PDF QA Bot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(chunks, embedding=embeddings)

    retriever = vectordb.as_retriever()

    llm = GeminiPro(
        model="gemini-pro-2",
        temperature=0,
        max_output_tokens=1024,
        google_api_key=os.getenv("GOOGLE_API_KEY")  # Use a generic env var name
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    query = st.text_input("Ask a question about the document:", value="What is this paper talking about?")

    if st.button("Ask"):
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": query})
            st.subheader("Answer:")
            st.write(result["result"])

            st.subheader("Sources:")
            for doc in result["source_documents"]:
                st.write(doc.metadata.get("source", "No source info"))