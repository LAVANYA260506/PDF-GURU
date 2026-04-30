import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import os
import time
from llm import generate_answer

# ---------------- CONFIG ----------------
PERSIST_DIRECTORY = "./chromadb"
COLLECTION_NAME = "pdf-database"

# ---------------- CACHED MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# ---------------- PDF PROCESSING ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(raw_text)

# ---------------- EMBEDDING ----------------
def embed_query(text):
    return model.encode(text).tolist()

def embed_chunks(chunks):
    return model.encode(chunks, batch_size=32).tolist()

# ---------------- DB ----------------
@st.cache_resource
def get_collection():
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def store_chunks(chunks):
    collection = get_collection()
    embeddings = embed_chunks(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks
    )
    return len(chunks)

def retrieve(query):
    collection = get_collection()

    if collection.count() == 0:
        return ""

    query_vec = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=5,
        include=["documents"]
    )

    docs = results.get("documents", [[]])[0]
    return "\n\n".join(docs)

# ---------------- STREAMING RESPONSE ----------------
def stream_response(text):
    lines = text.split("\n")
    placeholder = st.empty()
    full_text = ""

    for line in lines:
        words = line.split(" ")
        line_text = ""

        for word in words:
            line_text += word + " "
            placeholder.markdown(f"🧘‍♂️ {full_text + line_text}")
            time.sleep(0.03)

        full_text += line_text + "\n\n"
        placeholder.markdown(f"🧘‍♂️ {full_text}")
        time.sleep(0.1)  # pause between sections

    return full_text

# ---------------- MAIN APP ----------------
def main():
    st.set_page_config(page_title="PDF GURU", page_icon="📚", layout="wide")

    # --- session ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # auto-detect DB
    collection = get_collection()
    vector_ready = collection.count() > 0

    # --- header ---
    st.markdown(
        """
        <h1 style='text-align: center;'>📚 PDF GURU</h1>
        <p style='text-align: center;'>Learn from your documents like a student with a guru 🧠</p>
        """,
        unsafe_allow_html=True
    )

    # --- chat history ---
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- input ---
    user_query = st.chat_input("Ask your question...")

    if user_query:
        # user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # assistant
        with st.chat_message("assistant"):
            with st.spinner("GURU is thinking... 🧘"):

                if not vector_ready:
                    response = (
                        "It seems no documents have been loaded yet. "
                        "Please upload your PDFs, and then we can explore the concepts together."
                    )
                    st.markdown(f"🧘‍♂️ {response}")

                else:
                    context = retrieve(user_query)

                    # chat memory
                    history = ""
                    for msg in st.session_state.chat_history[-6:]:
                        role = "User" if msg["role"] == "user" else "Guru"
                        history += f"{role}: {msg['content']}\n"

                    final_prompt = f"""
                    Conversation:
                    {history}

                    Context:
                    {context}

                    Question:
                    {user_query}
                    """

                    answer = generate_answer(user_query, final_prompt)

                    # 🔥 streaming effect
                    response = stream_response(answer)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"🧘‍♂️ {response}"}
        )

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.subheader("📂 Your Documents")

        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)

        if st.button("⚡ Ingest & Prepare"):
            if not pdf_docs:
                st.warning("Upload at least one PDF.")
            else:
                with st.spinner("Processing knowledge..."):
                    raw_text = get_pdf_text(pdf_docs)
                    chunks = get_chunks(raw_text)
                    count = store_chunks(chunks)

                st.success(f"✅ {count} chunks stored successfully!")

        if vector_ready:
            st.success("📚 Knowledge base loaded!")

        st.markdown("---")

        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()