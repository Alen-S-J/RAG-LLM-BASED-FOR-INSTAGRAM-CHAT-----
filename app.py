# filename: app.py

import os
import pickle
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FAISS_DIR = "storage/faiss/faiss_index"
DOCS_FILE = "storage/faiss/docs.pkl"

# FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []

# DuckDuckGo Web Search
def simple_web_search(query: str) -> str:
    url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if not response.ok:
        return "Web search failed."
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.select(".result__a")[:5]:
        title = result.get_text(strip=True)
        href = result.get("href")
        if href:
            results.append(f"- {title}: {href}")
    return "\n".join(results) if results else "No relevant results found."

# Load FAISS + Docs
def load_instagram_vectorstore():
    if not os.path.exists(FAISS_DIR):
        return None, []
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = []
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "rb") as f:
            docs = pickle.load(f)
    return vectorstore, docs

# Cosine similarity reranker
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embedding_rerank(query: str, documents: List[str], embeddings_model, top_n: int = 3):
    query_embedding = embeddings_model.embed_query(query)
    doc_embeddings = embeddings_model.embed_documents(documents)
    scored = [(doc, cosine_similarity(np.array(query_embedding), np.array(doc_emb)))
              for doc, doc_emb in zip(documents, doc_embeddings)]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

# GPT-based reranker
def gpt_rerank(query: str, documents: List[str], client, top_n: int = 3):
    scored = []
    for doc in documents:
        prompt = f"""
You are a smart reranker.

Given this user query: "{query}"
And this document: "{doc}"

Score how relevant this document is to the query on a scale from 1 (not relevant) to 10 (very relevant). Only return a number.
"""
        response = client.invoke(prompt).content.strip()
        try:
            score = float(response)
            scored.append((doc, score))
        except:
            continue
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

# Main QA Function
def answer_with_context(query: str, history: List[Dict[str, str]]) -> str:
    vectorstore, docs = load_instagram_vectorstore()
    context = ""

    if vectorstore:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=6)
        retrieved_texts = [doc.page_content for doc, _ in docs_with_scores]

        RERANK_MODE = "embedding"  # "embedding", "gpt", or "none"
        if RERANK_MODE == "embedding":
            reranked = embedding_rerank(query, retrieved_texts, OpenAIEmbeddings(), top_n=3)
        elif RERANK_MODE == "gpt":
            reranked = gpt_rerank(query, retrieved_texts, ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1), top_n=3)
        else:
            reranked = [(doc, 0.0) for doc in retrieved_texts[:3]]

        context = "\n\n".join([doc for doc, score in reranked])
    else:
        context = "No Instagram data available."

    web_result = simple_web_search(query)
    chat_history = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history[-3:]])

    prompt_template = PromptTemplate(
        input_variables=["chat_history", "instagram_context", "web_context", "question"],
        template="""
You are a helpful assistant combining Instagram personal data and live web info.

Conversation History:
{chat_history}

Instagram Context:
{instagram_context}

Web Search Results:
{web_context}

Current Question: {question}

Answer:"""
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key=OPENAI_API_KEY
    )

    final_prompt = prompt_template.format(
        chat_history=chat_history,
        instagram_context=context,
        web_context=web_result,
        question=query
    )

    return llm.invoke(final_prompt).content

# Root health check
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Instagram RAG API is running."}

# Chat endpoint
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        answer = answer_with_context(request.question, request.history)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
