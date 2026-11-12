import io
import os
import re
import time
import wave

import numpy as np
import streamlit as st
import whisper
from dotenv import load_dotenv
from gtts import gTTS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "voice-rag"
PATH = r"C:\Users\yash\Downloads\AI_Engineering_Interview_Questions.pdf"
END_WORDS = {"bye", "quit", "exit", "goodbye"}

st.set_page_config(page_title="Voice RAG Assistant", page_icon="🎙️", layout="wide")

st.markdown(
    """
<style>
.stApp {background: radial-gradient(circle at 15% 0%, #1e1b4b 0%, #0f172a 45%, #020617 100%);}
.hero {padding: 1.4rem 1.8rem; border-radius: 20px; margin-bottom: 1rem;
  background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
  box-shadow: 0 10px 40px rgba(99,102,241,.35);}
.hero h1 {color: #fff; margin: 0; font-size: 2rem;}
.hero p {color: #e0e7ff; margin: .3rem 0 0 0;}
[data-testid="stChatMessage"] {background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px; padding: .8rem 1rem; backdrop-filter: blur(6px);}
.chip {display:inline-block; padding: .25rem .7rem; border-radius: 999px; margin: .15rem 0;
  background: rgba(99,102,241,.18); border: 1px solid rgba(129,140,248,.4); color: #c7d2fe; font-size: .8rem;}
section[data-testid="stSidebar"] {background: rgba(15,23,42,.9); border-right: 1px solid rgba(255,255,255,.06);}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models and index... (first run takes a minute)")
def load_backend():
    stt = whisper.load_model("base")
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-20b")

    ext = os.path.splitext(PATH)[-1].lower()
    loader = {".pdf": PyPDFLoader, ".csv": CSVLoader, ".docx": Docx2txtLoader}[ext](PATH)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250).split_documents(
        loader.load()
    )

    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME, dimension=768, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        vs = PineconeVectorStore.from_documents(chunks, emb, index_name=INDEX_NAME)
    else:
        vs = PineconeVectorStore(index_name=INDEX_NAME, embedding=emb)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 5
    ens = EnsembleRetriever(
        retrievers=[bm25, vs.as_retriever(search_kwargs={"k": 5})], weights=[0.5, 0.5]
    )
    retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=ens)
    return stt, llm, retriever, len(chunks)


stt, llm, retriever, n_chunks = load_backend()

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a voice RAG assistant. Answer using the context only. "
                   "If the answer is not in the context, say \"I don't know\". "
                   "Keep answers short and easy to say aloud."),
        ("human", "question: {text}\n\ncontext: {context}"),
    ]
)


def transcribe(wav_bytes: bytes) -> str:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return stt.transcribe(data, fp16=False)["text"].strip()


def tts(text: str) -> bytes:
    buf = io.BytesIO()
    gTTS(text=re.sub(r"[*#<]", "", text), lang="en", tld="co.in").write_to_fp(buf)
    return buf.getvalue()


def ask(question: str, speak: bool):
    if set(re.findall(r"\w+", question.lower())) & END_WORDS:
        reply, sources = "Goodbye! 👋 Ask me anything whenever you're back.", []
    else:
        docs = retriever.invoke(question)
        sources = [d.page_content for d in docs]
        reply = (PROMPT | llm).invoke({"text": question, "context": "\n\n".join(sources)}).content
    audio = tts(reply) if speak else None
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply, "sources": sources, "audio": audio},
    ]
    st.session_state.autoplay = True


# ---------- state ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("mic", 0)

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    speak = st.toggle("🔊 Speak answers", value=True)
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("### 📚 Knowledge base")
    st.markdown(f"<span class='chip'>📄 {os.path.basename(PATH)}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='chip'>🧩 {n_chunks} chunks</span>", unsafe_allow_html=True)
    st.markdown("<span class='chip'>🔎 BM25 + Dense hybrid</span>", unsafe_allow_html=True)
    st.markdown("<span class='chip'>🗣️ Whisper + gTTS</span>", unsafe_allow_html=True)
    st.caption("Tip: say or type 'bye' to end the conversation.")

# ---------- header ----------
st.markdown(
    "<div class='hero'><h1>🎙️ Voice RAG Assistant</h1>"
    "<p>Ask by voice or text. Answers come only from your document.</p></div>",
    unsafe_allow_html=True,
)

chat_box = st.container()

# ---------- inputs ----------
c1, c2 = st.columns([1, 2])
with c1:
    audio_file = st.audio_input("🎙️ Tap to speak", key=f"mic_{st.session_state.mic}")
text_q = st.chat_input("...or type your question here")

if audio_file is not None:
    with st.spinner("🎧 Listening and thinking..."):
        q = transcribe(audio_file.getvalue())
        if len(q) >= 3:
            ask(q, speak)
    st.session_state.mic += 1  # resets the mic widget
    st.rerun()
elif text_q:
    with st.spinner("🤔 Thinking..."):
        ask(text_q, speak)

# ---------- render chat ----------
autoplay = st.session_state.pop("autoplay", False)
with chat_box:
    if not st.session_state.messages:
        st.info("👋 Tap the mic and ask a question about your document to get started.")
    last = len(st.session_state.messages) - 1
    for i, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("audio"):
                    st.audio(m["audio"], format="audio/mp3", autoplay=autoplay and i == last)
                if m.get("sources"):
                    with st.expander(f"📎 Sources ({len(m['sources'])})"):
                        for s in m["sources"]:
                            st.caption(s[:350] + ("..." if len(s) > 350 else ""))