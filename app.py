import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import fitz
# creating client object
client = chromadb.PersistentClient(path="./chroma_db")
#creating collection inside our client object
collection = client.create_collection(name="first_collection", get_or_create=True) 
# loading emedding models for our uploaded files
model = SentenceTransformer("all-MiniLM-L6-v2")
st.title("Ask Questions from Text & PDF Files")
upload_files=st.file_uploader("upload files (pdf,txt)",type["pdf","txt"],accept_multiple_files=True)
def extract_text_from_pdf(file) -> str:
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text
if uploaded_files:
    for file in uploaded_files:
        filename = file.name
        if filename.endswith(".txt"):
            text = file.read().decode("utf-8")
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(file)
        else:
            continue  

collection.add(
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[{"filename": filename}],
            ids=[f"doc-{filename}"]
        )
st.success(f"{len(uploaded_files)} files uploaded and indexed.")
query = st.text_input(" Ask something based on your files")
if query:
    query_embedding = model.encode(query)
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=2)
st.subheader(" Top Matching Documents:")
for doc in results["documents"][0]:
        st.markdown(f"> {doc[:300]}...")


