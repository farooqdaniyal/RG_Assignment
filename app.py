
import os
import tempfile
import shutil
import uuid

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF RAG Assistant")
st.caption("Upload a PDF and ask questions based on its content.")

load_dotenv()

# Read token from Streamlit Cloud Secrets or local .env
try:
    HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
except Exception:
    HF_TOKEN = ""

HF_TOKEN = (
    HF_TOKEN
    or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    or os.getenv("HF_TOKEN", "")
).strip()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    if not HF_TOKEN:
        HF_TOKEN = st.text_input(
            "Hugging Face API Token",
            type="password",
        ).strip()

    model_name = st.text_input(
        "Model Name",
        value="openai/gpt-oss-120b:fastest",
    ).strip()

    chunk_size = st.slider(
        "Chunk Size", 200, 2000, 1000, step=100
    )
    chunk_overlap = st.slider(
        "Chunk Overlap", 0, 500, 100, step=10
    )
    top_k = st.slider(
        "Chunks to retrieve", 1, 10, 4
    )


# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None


# ---------------- EMBEDDING MODEL ----------------
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ---------------- PDF UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)

if uploaded_file:
    st.caption(f"Selected file: {uploaded_file.name}")

    if st.button("🔍 Process PDF", type="primary"):
        if not uploaded_file.getvalue():
            st.error("The uploaded PDF is empty.")
            st.stop()

        temp_path = None

        try:
            with st.spinner("Extracting text and creating embeddings..."):
                # Use a unique collection for each processed PDF
                file_id = str(uuid.uuid4()).replace("-", "")
                uploaded_bytes = uploaded_file.getvalue()

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as temp_file:
                    temp_file.write(uploaded_bytes)
                    temp_path = temp_file.name

                docs = PyPDFLoader(temp_path).load()

                if not docs or not any(
                    doc.page_content.strip() for doc in docs
                ):
                    st.error(
                        "No readable text found. "
                        "This may be a scanned PDF requiring OCR."
                    )
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=min(chunk_overlap, chunk_size - 1),
                )
                chunks = splitter.split_documents(docs)

                if not chunks:
                    st.error("No text chunks could be created.")
                    st.stop()

                embeddings = get_embedding_model()

                # Keep the vector store in memory for this session.
                # Each PDF gets its own unique collection.
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=f"pdf_{file_id}",
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.processed_file_id = file_id
                st.session_state.chat_history = []

            st.success(
                f"PDF processed successfully: "
                f"{len(docs)} pages and {len(chunks)} chunks."
            )

        except Exception as e:
            st.error(f"PDF processing failed: {type(e).__name__}: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


st.divider()


# ---------------- ASK QUESTIONS ----------------
st.subheader("💬 Ask a question about your PDF")

if st.session_state.vectorstore is None:
    st.info("Upload a PDF and click Process PDF first.")

else:
    question = st.text_input(
        "Your question",
        placeholder="What is the main topic of this document?",
        key="question_input",
    )

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")

        elif not HF_TOKEN:
            st.error(
                "Hugging Face token is missing. "
                "Add HUGGINGFACEHUB_API_TOKEN in Streamlit Secrets."
            )

        elif not model_name:
            st.error("Please enter a model name.")

        else:
            try:
                with st.spinner("Searching the PDF and generating answer..."):
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": top_k,
                            "fetch_k": max(top_k * 2, top_k),
                            "lambda_mult": 0.5,
                        },
                    )

                    relevant_docs = retriever.invoke(question)

                    if not relevant_docs:
                        st.warning("No relevant text was found in the PDF.")
                        st.stop()

                    context_parts = []

                    for doc in relevant_docs:
                        page = doc.metadata.get("page")
                        page_label = (
                            str(page + 1)
                            if isinstance(page, int)
                            else "Unknown"
                        )
                        context_parts.append(
                            f"[PDF page {page_label}]\n{doc.page_content}"
                        )

                    context = "\n\n".join(context_parts)

                    client = OpenAI(
                        base_url="https://router.huggingface.co/v1",
                        api_key=HF_TOKEN,
                        timeout=60.0,
                        max_retries=2,
                    )

                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a PDF question-answering assistant. "
                                    "Answer using only the supplied PDF context. "
                                    "If the answer is not supported by the context, "
                                    "say: I could not find the answer in the document. "
                                    "Do not invent facts. Cite page numbers when possible."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"PDF context:\n{context}\n\n"
                                    f"Question: {question}"
                                ),
                            },
                        ],
                        max_tokens=512,
                        temperature=0.1,
                    )

                    answer = response.choices[0].message.content

                    if not answer:
                        answer = "The model returned an empty answer."

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer,
                        "sources": [
                            {
                                "page": doc.metadata.get("page"),
                                "text": doc.page_content,
                            }
                            for doc in relevant_docs
                        ],
                    }
                )

            except RateLimitError:
                st.error(
                    "Hugging Face rate limit or credits issue. "
                    "Check your Inference Providers usage and credits."
                )

            except APIStatusError as e:
                st.error(f"Hugging Face API returned HTTP {e.status_code}.")

                if e.status_code == 401:
                    st.warning(
                        "Token invalid or not accepted. "
                        "Create a valid token with Inference Providers permission."
                    )
                elif e.status_code == 403:
                    st.warning(
                        "Access denied. Check token permissions and model/provider access."
                    )
                elif e.status_code == 402:
                    st.warning(
                        "Provider credits or payment may be required."
                    )
                elif e.status_code == 404:
                    st.warning(
                        "Model or endpoint not found. Check the model name."
                    )
                elif e.status_code == 429:
                    st.warning(
                        "Too many requests. Wait and try again."
                    )
                else:
                    st.warning(
                        "Check the model provider and Hugging Face Inference settings."
                    )

                # Avoid showing raw response data that may contain sensitive details.
                st.caption(
                    "For detailed diagnostics, check Streamlit Cloud logs. "
                    "Never share your API token."
                )

            except APIConnectionError:
                st.error(
                    "Could not connect to Hugging Face Router. "
                    "Check your internet connection or try again later."
                )

            except Exception as e:
                st.error(
                    f"Unexpected error: {type(e).__name__}: {e}"
                )


# ---------------- DISPLAY CHAT HISTORY ----------------
if st.session_state.chat_history:
    st.divider()
    st.subheader("🕑 Previous Questions & Answers")

    for item in reversed(st.session_state.chat_history):
        with st.expander(item["question"], expanded=True):
            st.markdown("**Answer**")
            st.write(item["answer"])

            with st.expander("📚 Retrieved source chunks"):
                for i, source in enumerate(item["sources"], start=1):
                    page = source["page"]
                    page_label = (
                        str(page + 1)
                        if isinstance(page, int)
                        else "Unknown"
                    )
                    st.markdown(f"**Source {i} — PDF page {page_label}**")
                    st.write(source["text"][:1200])
