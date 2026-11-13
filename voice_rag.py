import os
import re
import time
from io import BytesIO

import numpy as np
import pygame
import sounddevice as sd
import whisper
from dotenv import load_dotenv
from gtts import gTTS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph, add_messages
from pinecone import Pinecone, ServerlessSpec
from typing_extensions import Annotated, Any, List, TypedDict

load_dotenv()

INDEX_NAME = "voice-rag"
END_WORDS = {"bye", "quit", "exit", "stop", "goodbye"}
PATH = r"C:\Users\yash\Downloads\AI_Engineering_Interview_Questions.pdf"

pygame.mixer.init()  # required before using pygame.mixer.music
stt_model = whisper.load_model("base")

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
)


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    docs: Any
    chunks: Any
    embedding: Any  # holds the retriever object
    context: Any
    text: Any


def documents(state: State):
    ext = os.path.splitext(PATH)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(PATH)
    elif ext == ".csv":
        loader = CSVLoader(PATH)
    elif ext == ".docx":
        loader = Docx2txtLoader(PATH)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return {"docs": loader.load()}


def split_docs(state: State):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    return {"chunks": splitter.split_documents(state["docs"])}


def Embedding(state: State):
    chunks = state["chunks"]
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        # first run only: upload chunks
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            index_name=INDEX_NAME,
        )
    else:
        # already indexed: reuse it, no duplicate uploads
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embedding_model,
        )

    dense = vectorstore.as_retriever(search_kwargs={"k": 5})
    bm25 = BM25Retriever.from_documents(documents=chunks)
    bm25.k = 5

    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
    multi_query = MultiQueryRetriever.from_llm(llm=llm, retriever=ensemble)
    return {"embedding": multi_query}


def into_text(state: State):
    print("Listening 🎙️ 5 sec")
    sr, duration = 16000, 5
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    audio = audio.flatten()

    # skip near-silent audio so Whisper doesn't hallucinate text
    if np.abs(audio).max() < 0.01:
        print("YOU: (silence)")
        return {"text": ""}

    result = stt_model.transcribe(audio, fp16=False)
    text = result["text"].strip()
    print("YOU:", text)
    return {"text": text}


def routing(state: State):
    text = state["text"].strip().lower()
    if len(text) < 3:
        return "listen"
    words = set(re.findall(r"\w+", text))
    if words & END_WORDS:
        return END
    return "retriever"


def retriever(state: State):
    docs = state["embedding"].invoke(state["text"])
    return {"context": [d.page_content for d in docs]}


def Answer(state: State):
    context = "\n\n".join(state["context"])
    text = state["text"].strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a voice RAG assistant. Answer the user's question using "
                "the context only. Do not use your own knowledge. If the answer "
                "is not in the context, say \"I don't know\". Keep answers short "
                "and easy to say aloud.",
            ),
            ("human", "question: {text}\n\ncontext: {context}"),
        ]
    )
    response = (prompt | llm).invoke({"text": text, "context": context})
    return {
        "messages": [
            HumanMessage(content=text),
            AIMessage(content=response.content),
        ]
    }


def speak(state: State):
    reply = state["messages"][-1].content
    reply = re.sub(r"[*#<]", "", reply)
    print("BOT:", reply)

    buf = BytesIO()
    gTTS(text=reply, lang="en", tld="co.in").write_to_fp(buf)
    buf.seek(0)

    pygame.mixer.music.load(buf, "mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()
    buf.close()
    return {}


q = StateGraph(State)
q.add_node("doc", documents)
q.add_node("chunk", split_docs)
q.add_node("embedding", Embedding)
q.add_node("listen", into_text)
q.add_node("retriever", retriever)
q.add_node("answer", Answer)
q.add_node("speak", speak)

q.add_edge(START, "doc")
q.add_edge("doc", "chunk")
q.add_edge("chunk", "embedding")
q.add_edge("embedding", "listen")
q.add_conditional_edges("listen", routing)  # -> listen / retriever / END
q.add_edge("retriever", "answer")
q.add_edge("answer", "speak")
q.add_edge("speak", "listen")
graph = q.compile()

if __name__ == "__main__":
    response=graph.invoke({"messages": []}, config={"recursion_limit": 200})
    print(response)