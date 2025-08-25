import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from src.ingest import ingest_files
from src.query import get_answer

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "vectorstore/chroma"

os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# -------------------------------
# Load Documents
# -------------------------------
def load_documents(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs = loader.load()
        documents.extend(docs)
    return documents

# -------------------------------
# Split Documents into Chunks
# -------------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    return chunks

# -------------------------------
# Create Embeddings & Store in ChromaDB
# -------------------------------
def create_embeddings_and_store(chunks):
    print("🔹 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create Chroma vector store
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    vectordb.persist()
    print(f"✅ Embeddings created and stored in {CHROMA_PERSIST_DIR}!")

def ingest_files(file_paths):
    """
    Ingests a list of file paths (PDF, TXT, DOCX), splits, embeds, and stores in ChromaDB.
    """
    documents = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs = loader.load()
        documents.extend(docs)
    if not documents:
        return 0, 0
    chunks = split_documents(documents)
    if chunks:
        create_embeddings_and_store(chunks)
    return len(documents), len(chunks)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    print(f"📄 Loaded {len(docs)} documents")
    
    chunks = split_documents(docs)
    print(f"🔹 Split into {len(chunks)} chunks")
    
    if len(chunks) > 0:
        create_embeddings_and_store(chunks)
    else:
        print("⚠️ No chunks created. Check documents in 'data/' folder.")

# After file upload in Streamlit:
if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    num_docs, num_chunks = ingest_files(file_paths)
    st.success(f"Ingested {num_docs} documents, split into {num_chunks} chunks.")
    answer, sources = get_answer(query)
