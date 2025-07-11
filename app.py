import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import fitz  

# Create ChromaDB client 
client = chromadb.Client()

# Get or create collection
if "second_collection" not in [c.name for c in client.list_collections()]:
    collection = client.create_collection(name="second_collection")
else:
    collection = client.get_collection(name="second_collection")

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# App UI
st.title("Ask Questions from Text & PDF Files")

uploaded_files = st.file_uploader("Upload files (PDF, TXT)", type=["pdf", "txt"], accept_multiple_files=True)

def extract_text_from_pdf(file) -> str:
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Handle uploaded files
if uploaded_files:
    for file in uploaded_files:
        filename = file.name

        # Extract text
        if filename.endswith(".txt"):
            text = file.read().decode("utf-8")
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        else:
            st.warning(f"{filename} is not supported. Skipping.")
            continue

        # Get embedding
        embedding = model.encode(text)

        # Add to ChromaDB
        collection.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[{"filename": filename}],
            ids=[f"doc-{filename}"]
        )
    
    st.success(f" {len(uploaded_files)} files uploaded and indexed.")
else:
    st.info("Please upload one or more PDF or TXT files.")

# Question input
query = st.text_input("Ask something based on your files:")

# If query submitted
if query:
    query_embedding = model.encode(query)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=2)

    st.subheader(" Top Matching Documents:")
for i, doc in enumerate(results["documents"][0]):
    filename = results["metadatas"][0][i]["filename"]
    st.write(f"File: {filename}")
    st.write(doc[:300] + "...")




