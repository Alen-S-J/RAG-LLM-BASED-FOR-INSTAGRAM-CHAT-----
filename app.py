# filename: app_streamlit.py

import os
import pickle
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict
from textwrap import shorten
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

FAISS_DIR = "storage/faiss/faiss_index"
DOCS_FILE = "storage/faiss/docs.pkl"

# ---------------------------
# Helper Functions
# ---------------------------

def simple_web_search(query: str):
    """DuckDuckGo search returning top 5 results."""
    url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if not response.ok:
        return "Web search failed.", []
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.select(".result__a")[:5]:
        title = result.get_text(strip=True)
        href = result.get("href")
        if href:
            results.append({"title": title, "url": href})
    display_str = "\n".join([f"- {r['title']}: {r['url']}" for r in results])
    return display_str if results else "No relevant results found.", results

def load_instagram_vectorstore():
    """Load FAISS vectorstore and docs."""
    if not os.path.exists(FAISS_DIR):
        return None, []
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
    docs = []
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "rb") as f:
            docs = pickle.load(f)
    return vectorstore, docs

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embedding_rerank(query: str, documents: List[str], embeddings_model, top_n: int = 3):
    """Rerank retrieved docs using embeddings."""
    query_embedding = embeddings_model.embed_query(query)
    doc_embeddings = embeddings_model.embed_documents(documents)
    scored = [(doc, cosine_similarity(np.array(query_embedding), np.array(doc_emb)))
              for doc, doc_emb in zip(documents, doc_embeddings)]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

def summarize_chunks(chunks: List[str], max_chars=300):
    """Shorten chunks to prevent dumping raw text."""
    return [shorten(c.strip().replace("\n", " "), width=max_chars, placeholder="...") for c in chunks if c.strip()]

# ---------------------------
# Main QA Function
# ---------------------------

def answer_with_context(query: str, history: List[Dict[str, str]]):
    vectorstore, _ = load_instagram_vectorstore()
    insta_context = ""
    insta_links = []

    if vectorstore:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=6)
        retrieved_texts = [doc.page_content for doc, _ in docs_with_scores]
        insta_links = [str(doc.metadata.get("url") or "No URL") for doc, _ in docs_with_scores]

        summarized_texts = summarize_chunks(retrieved_texts)
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        reranked = embedding_rerank(query, summarized_texts, embeddings_model, top_n=3)
        insta_context = "\n".join([doc for doc, _ in reranked])
    else:
        insta_context = "No Instagram data available."

    web_result_str, web_links = simple_web_search(query)

    # Limit web context to ~30% of Instagram context length
    max_web_chars = int(len(insta_context) * 0.43)  # adjusted to ~30% ratio
    web_context_trimmed = web_result_str[:max_web_chars]

    chat_history = "\n".join([f"User: {h['user']}\nBot: {h['bot']}" for h in history[-3:]])

    prompt_template = PromptTemplate(
        input_variables=["chat_history", "instagram_context", "instagram_links", "web_context", "web_links", "question"],
        template="""
You are a helpful assistant combining **Instagram personal data (70% weight)** and **live web search data (30% weight)**.

Rules:
- Prioritize Instagram context first in answers.
- Use web search to fill missing details.
- Mention video/page names naturally, with links in parentheses.
- Keep tone conversational, avoid raw dumps.

Conversation History:
{chat_history}

=== Instagram Context (90%) ===
{instagram_context}

Instagram Links:
{instagram_links}

=== Web Context (10%) ===
{web_context}

Web Links:
{web_links}

Current Question: {question}

Answer:
"""
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=512, api_key=OPENAI_API_KEY)

    final_prompt = prompt_template.format(
        chat_history=chat_history,
        instagram_context=insta_context,
        instagram_links="\n".join(insta_links),
        web_context=web_context_trimmed,
        web_links="\n".join([f"{w['title']}: {w['url']}" for w in web_links]),
        question=query
    )

    answer = llm.invoke(final_prompt).content
    return answer, web_links

# ---------------------------
# Streamlit Interface
# ---------------------------

st.set_page_config(page_title="Instagram + Web RAG", page_icon="📱", layout="centered")
st.title("📱 Instagram + Web RAG Assistant")
st.markdown("Ask a question and get a **70% Instagram / 30% web search** blended answer.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type your question...")

if user_input:
    answer, web_links = answer_with_context(user_input, st.session_state.history)
    st.session_state.history.append({"user": user_input, "bot": answer})

for chat in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])

if st.session_state.history:
    st.divider()
    st.subheader("🌐 Web Sources")
    for w in web_links:
        st.markdown(f"- [{w['title']}]({w['url']})")
