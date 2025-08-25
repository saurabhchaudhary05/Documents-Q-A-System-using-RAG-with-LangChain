import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# -------------------------------
# Config
# -------------------------------
CHROMA_PERSIST_DIR = "vectorstore/chroma"
MODEL_NAME = "tiiuae/falcon-7b-instruct"  # HuggingFace instruct model
TOP_K = 3  # Number of relevant chunks to retrieve
MAX_TOKENS = 200
TEMPERATURE = 0.0

# -------------------------------
# Load Vector Store
# -------------------------------
vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=None)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

# -------------------------------
# Load HuggingFace LLM Pipeline
# -------------------------------
generator = pipeline(
    task="text-generation",
    model=MODEL_NAME,
    max_new_tokens=MAX_TOKENS,
    temperature=TEMPERATURE
)
llm = HuggingFacePipeline(pipeline=generator)

# -------------------------------
# Setup Retrieval QA Chain
# -------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# -------------------------------
# Helper Function for Streamlit
# -------------------------------
def get_answer(query: str):
    """
    Returns the answer and sources for a given query using RAG pipeline.
    """
    result = qa_chain(query)
    answer = result['result']
    sources = []
    for doc in result.get('source_documents', []):
        sources.append({
            "title": doc.metadata.get('source', 'Unknown'),
            "snippet": doc.page_content[:200]  # Show first 200 chars as snippet
        })
    return answer, sources

# -------------------------------
# User Query Loop (Optional CLI)
# -------------------------------
if __name__ == "__main__":
    print("🤖 RAG QA System (type 'exit' to quit)\n")
    while True:
        query = input("Enter your question: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        result = qa_chain(query)
        answer = result['result']
        print("\n💬 Answer:\n", answer)
        
        # Optional: Show source documents
        sources = result['source_documents']
        if sources:
            print("\n📄 Sources:")
            for doc in sources:
                print(f"- {doc.metadata.get('source', 'Unknown')}")
