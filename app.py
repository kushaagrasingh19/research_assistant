import os
import pathlib
import streamlit as st
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- Config ---
DATA_DIR = pathlib.Path("data")
INDEX_DIR = pathlib.Path("faiss_index")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

st.set_page_config(page_title="LLM Research Assistant", page_icon="📚", layout="wide")
st.title(" Research Assistant ")

# --- API key guard ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.info("No OPENAI_API_KEY found. Paste it to test this session.")
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

# --- Helpers ---
def load_docs_from_paths(paths: List[pathlib.Path]):
    docs = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
        elif p.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(p)).load())
    return docs

def build_or_update_index(files: List[bytes], names: List[str]):
    DATA_DIR.mkdir(exist_ok=True)
    saved = []
    for b, name in zip(files, names):
        dest = DATA_DIR / name
        with open(dest, "wb") as f:
            f.write(b)
        saved.append(dest)

    raw_docs = load_docs_from_paths(saved)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings()
    if INDEX_DIR.exists():
        vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings)

    vs.save_local(str(INDEX_DIR))
    return len(chunks)

def load_index():
    if not INDEX_DIR.exists():
        return None
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

# --- Sidebar: upload & manage index ---
with st.sidebar:
    st.header("📥 Add Papers")
    uploads = st.file_uploader(
        "Upload PDFs / .txt / .md", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Build / Update Index", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Upload at least one file.")
        else:
            n = build_or_update_index([u.getvalue() for u in uploads], [u.name for u in uploads])
            st.success(f"Indexed/updated ✅  ({n} chunks)")

    st.markdown("---")
    if st.button("Clear Index (danger)"):
        if INDEX_DIR.exists():
            for p in INDEX_DIR.glob("*"):
                p.unlink()
            INDEX_DIR.rmdir()
        st.success("Cleared index.")

# --- Main tabs ---
tab_ask, tab_summarize = st.tabs(["🔎 Ask Questions (with citations)", "📝 Summarize a Paper"])

with tab_ask:
    st.subheader("Query across your indexed papers")
    vs = load_index()
    if not vs:
        st.info("No index yet. Upload files and click **Build / Update Index** in the sidebar.")
    else:
        k = st.slider("Top-K passages", 2, 10, 4)
        q = st.text_input("Your question")
        if st.button("Answer", type="primary") and q:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff",
                retriever=vs.as_retriever(search_kwargs={"k": k}),
                return_source_documents=True
            )
            res = qa({"query": q})
            st.markdown("### Answer")
            st.write(res["result"])

            st.markdown("### Sources")
            seen = set()
            for i, d in enumerate(res["source_documents"], start=1):
                src = d.metadata.get("source", "unknown")
                page = d.metadata.get("page")
                key = (src, page)
                if key in seen:
                    continue
                seen.add(key)
                fname = pathlib.Path(src).name
                st.write(f"{i}. {fname}" + (f" — page {page+1}" if page is not None else ""))

with tab_summarize:
    st.subheader("Summarize one local PDF")
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        st.info("Upload a PDF first (sidebar).")
    else:
        choice = st.selectbox("Choose PDF", [p.name for p in pdfs])
        if st.button("Summarize"):
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            pages = PyPDFLoader(str(DATA_DIR / choice)).load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            chunks = splitter.split_documents(pages)

            st.write("Summarizing…")
            partials = []
            for ch in chunks:
                prompt = (
                    "Summarize this chunk focusing on Problem, Method, Results, Limitations.\n\n"
                    f"{ch.page_content}"
                )
                partials.append(llm.invoke(prompt).content)

            final = llm.invoke(
                "Combine the chunk summaries into a concise paper summary. "
                "Structure as: Background, Method, Results, Limitations, Key Takeaways.\n\n"
                + "\n\n---\n\n".join(partials)
            ).content
            st.markdown("### Summary")
            st.write(final)
