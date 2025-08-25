import os
import streamlit as st
from src.ingest import ingest_files
from src.query import get_answer

st.set_page_config(page_title="RAG QA", page_icon="🤖", layout="wide")

# Sidebar: Project info and settings
st.sidebar.title("RAG QA Project")
st.sidebar.markdown("Retrieval-Augmented Generation QA System")
st.sidebar.markdown("---")
st.sidebar.header("Settings")
answer_style = st.sidebar.selectbox("Answer Style", ["Short", "Long"])

# File uploader for document ingestion
uploaded_files = st.sidebar.file_uploader("Upload Documents (PDF, TXT, DOCX)", accept_multiple_files=True)
if uploaded_files:
    file_paths = []
    os.makedirs("data", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    with st.spinner("Ingesting documents..."):
        num_docs, num_chunks = ingest_files(file_paths)
    st.sidebar.success(f"Ingested {num_docs} docs, {num_chunks} chunks.")

st.sidebar.markdown("---")
st.sidebar.subheader("Query History")
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if st.session_state.query_history:
    for q in reversed(st.session_state.query_history[-10:]):
        st.sidebar.markdown(f"- {q}")
else:
    st.sidebar.caption("No queries yet.")

# Session state for chat history and likes
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "likes" not in st.session_state:
    st.session_state.likes = []

st.title("🤖 RAG QA System")
st.markdown(f"**Questions asked this session:** {len(st.session_state.chat_history)}")

# Input area
with st.container():
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Ask a question about your documents:")
    with col2:
        ask = st.button("Get Answer", use_container_width=True)

# Handle answer generation (real backend)
if ask:
    if query.strip() == "":
        st.warning("Please enter a question first!")
    else:
        with st.spinner("Generating answer..."):
            answer, sources = get_answer(query)
        st.session_state.chat_history.append(
            {"question": query, "answer": answer, "sources": sources, "feedback": ""}
        )
        st.session_state.likes.append(0)
        # Maintain last 10 queries in query_history
        st.session_state.query_history.append(query)
        if len(st.session_state.query_history) > 10:
            st.session_state.query_history = st.session_state.query_history[-10:]

# Display chat history as a conversation
if st.session_state.chat_history:
    st.subheader("Conversation")
    for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
        idx = len(st.session_state.chat_history) - i
        with st.container():
            st.markdown(f"**🧑‍💻 Q{i}:** {chat['question']}")
            st.success(chat["answer"])
            with st.expander("📄 Sources", expanded=False):
                if isinstance(chat["sources"], list) and chat["sources"]:
                    for src in chat["sources"]:
                        st.markdown(f"- **{src.get('title', 'Source')}**: {src.get('snippet', '')}")
                else:
                    st.markdown("No sources returned.")
            cols = st.columns([1, 4])
            with cols[0]:
                if st.button(f"👍 {st.session_state.likes[idx]}", key=f"like_{idx}"):
                    st.session_state.likes[idx] += 1
            with cols[1]:
                feedback = st.text_area("Feedback (optional):", value=chat["feedback"], key=f"fb_{idx}")
                st.session_state.chat_history[idx]["feedback"] = feedback
            st.markdown("---")

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.likes = []
    st.session_state.query_history = []
    st.info("Chat history cleared.")

st.info("created by ❤️ saurabh chaudhary")
