"""
Simple RAG (Retrieval-Augmented Generation) Pipeline — Streamlit App
----------------------------------------------------------------------
This app uses a PDF as the knowledge source and answers questions
based on the content of that PDF.

Steps:
1. Load PDF
2. Extract text
3. Split text into chunks
4. Generate embeddings (HuggingFace - local, free)
5. Store embeddings in a Vector Database (Chroma)
6. Take a question from the user
7. Retrieve relevant information
8. Generate answer using an LLM (HuggingFace Router API)
9. Test with multiple questions (chat history)

SETUP:
1. pip install -r requirements.txt
2. Create a .env file in this folder with:
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   (Get a token from: https://huggingface.co/settings/tokens)
3. Run: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from openai import OpenAI


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Assistant")
st.caption("Upload a PDF and ask questions about it — the AI will answer using the document's content.")


# =========================================================
# LOAD ENVIRONMENT VARIABLE (HuggingFace Token)
# =========================================================
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

with st.sidebar:
    st.header("⚙️ Settings")

    # If token is not found in .env, ask for it in the sidebar
    if not HF_TOKEN:
        HF_TOKEN = st.text_input("HuggingFace API Token", type="password")
        st.caption("Create a token at: huggingface.co/settings/tokens")

    model_name = st.text_input("Model Name", value="openai/gpt-oss-120b")
    chunk_size = st.slider("Chunk Size", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 100, step=10)
    top_k = st.slider("Number of chunks to retrieve (k)", 1, 10, 4)


# =========================================================
# SESSION STATE
# =========================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# STEP 1-5: UPLOAD -> LOAD -> SPLIT -> EMBED -> STORE
# =========================================================
uploaded_file = st.file_uploader("Upload your PDF here", type=["pdf"])

if uploaded_file is not None:
    if st.button("🔍 Process PDF", type="primary"):

        if not HF_TOKEN:
            st.error("Please enter your HuggingFace API Token first (in the sidebar).")
            st.stop()

        with st.spinner("Processing PDF..."):

            # Save the uploaded PDF temporarily to disk (loader needs a file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # ---- Step 1 & 2: Load PDF + Extract text ----
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # ---- Step 3: Split into chunks ----
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(docs)

            # ---- Step 4: Generate embeddings ----
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # ---- Step 5: Store in Vector Database (Chroma) ----
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory="chroma_db"
            )
            st.session_state.vectorstore = vectorstore

            os.remove(tmp_path)

        st.success(f"✅ PDF processed! {len(docs)} pages -> {len(chunks)} chunks created.")


st.markdown("---")


# =========================================================
# STEP 6-9: QUESTION -> RETRIEVE -> GENERATE ANSWER
# =========================================================
st.subheader("💬 Ask a question about the PDF")

if st.session_state.vectorstore is None:
    st.info("Please upload a PDF above and click 'Process PDF' first.")
else:
    user_question = st.text_input("Type your question:", placeholder="e.g. What is the main topic of this document?")

    if st.button("Get Answer") and user_question:

        if not HF_TOKEN:
            st.error("Please enter your HuggingFace API Token first (in the sidebar).")
            st.stop()

        with st.spinner("Generating answer..."):

            # ---- Step 6: Retriever (MMR search) ----
            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": top_k * 2, "lambda_mult": 0.5}
            )

            # ---- Step 7: Retrieve relevant chunks ----
            relevant_docs = retriever.invoke(user_question)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            # ---- Step 8: Generate answer using LLM (HuggingFace Router API) ----
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=HF_TOKEN
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant.\n"
                        "Answer the user's question using ONLY the provided context.\n"
                        "If the answer is not present in the context, say:\n"
                        '"I could not find the answer in the document."\n'
                        "Do not use outside knowledge.\n"
                        "Keep the answer clear and concise."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n\n{context}\n\nQuestion:\n\n{user_question}"
                }
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.1
            )

            answer = response.choices[0].message.content

        # ---- Step 9: Store + display (supports testing multiple questions) ----
        st.session_state.chat_history.append((user_question, answer))

        st.markdown("### 🧠 Answer")
        st.write(answer)

        with st.expander("📚 Source chunks (retrieved from the PDF)"):
            for i, doc in enumerate(relevant_docs, 1):
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Chunk {i} (Page {page}):**")
                st.write(doc.page_content[:400] + "...")


# =========================================================
# CHAT HISTORY (for testing with multiple questions)
# =========================================================
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🕑 Previous Questions & Answers")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
        st.markdown("---")