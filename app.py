import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

# Load environment variables
load_dotenv()

# === CONFIG ===
FAISS_BASE_PATH = "storage/faiss"
EMBEDDER = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LLM - GPT-3.5 Turbo
LLM = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=512,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# DuckDuckGo search results tool (returns titles + links)
duckduckgo_results = DuckDuckGoSearchResults()

# === FUNCTIONS ===
def get_latest_user_id():
    """Return the most recently updated user_id from FAISS_BASE_PATH."""
    if not os.path.exists(FAISS_BASE_PATH):
        return None
    
    user_dirs = [
        os.path.join(FAISS_BASE_PATH, d)
        for d in os.listdir(FAISS_BASE_PATH)
        if os.path.isdir(os.path.join(FAISS_BASE_PATH, d))
    ]
    
    if not user_dirs:
        return None
    
    # Sort by modification time (latest first)
    latest_dir = max(user_dirs, key=os.path.getmtime)
    return os.path.basename(latest_dir)

def load_faiss_index(user_id):
    """Load FAISS index for a given user."""
    index_path = os.path.join(FAISS_BASE_PATH, user_id, "faiss_index")
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, EMBEDDER, allow_dangerous_deserialization=True)
    return None

def web_search_with_links(query: str):
    """Perform a DuckDuckGo search and return results with links."""
    results = duckduckgo_results.run(query)
    search_links = []
    for r in results:
        if isinstance(r, dict) and "link" in r:
            search_links.append(f"{r['title']}: {r['link']}")
    return "\n".join(search_links) if search_links else "No links found."

def main():
    print("CLI Chat with Auto-FAISS Selection + DuckDuckGo Search (GPT-3.5 Turbo + Links)")

    user_id = get_latest_user_id()
    if user_id:
        print(f"Auto-selected latest user ID: {user_id}")
    else:
        print("No FAISS user data found. Using only web search.")

    faiss_db = load_faiss_index(user_id) if user_id else None
    chat_history = []

    if faiss_db:
        print(f"Loaded FAISS index for {user_id}")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=LLM,
            retriever=faiss_db.as_retriever(),
            memory=memory
        )
    else:
        print("No FAISS index loaded. All queries will use web search.")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if faiss_db:
            response = qa_chain.run(query)
            if not response.strip():  # If FAISS gives empty, fallback to search
                print("Using DuckDuckGo for live search...")
                links = web_search_with_links(query)
                response = f"{LLM.predict(query)}\n\n🔗 Links:\n{links}"
        else:
            links = web_search_with_links(query)
            response = f"{LLM.predict(query)}\n\n🔗 Links:\n{links}"

        chat_history.append({"user": query, "bot": response})
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
