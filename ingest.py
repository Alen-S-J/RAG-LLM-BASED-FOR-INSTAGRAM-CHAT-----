import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import whisper  
from crewai.tools import tool
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from post_comments_1 import extract_post_comments
from personal_information import extract_personal_information
from instagram_friend_map import extract_friend_map
from ads_about_meta import extract_ads_info
from reels import extract_reels
from posts_1 import extract_posts
from profile_photos import extract_profile_photos

load_dotenv()

# Paths for multiple users
BASE_DIRS = [
    
    Path("data/Demo-20250808T144143Z-1-001"),
    Path("data/Demo2")
]

# File types
AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".mp4", ".aac"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]

# Offline embedding model
embedding_model = OpenAIEmbeddings(model_name="text-embedding-3-small")


class IndexBuildResult(BaseModel):
    success: bool
    index_path: str
    total_chunks: int
    data_types: list
    message: str


@tool("Audio Processing Tool")
def process_audio_file(file_path: str) -> Dict[str, Any]:
    """Transcribes audio/video files with Whisper and returns transcript."""
    try:
        model = whisper.load_model("base")  # Change to "small" for better accuracy
        result = model.transcribe(file_path)
        transcript = result["text"].strip()

        txt_path = Path(file_path).with_suffix(".txt")
        txt_path.write_text(transcript, encoding="utf-8")

        return {
            "success": True,
            "text": transcript,
            "metadata": {
                "source": str(file_path),
                "type": "audio_transcript",
                "filename": Path(file_path).name
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool("File Collection Tool Per User")
def collect_user_files(base_dir: str) -> Dict[str, Any]:
    """Collects all relevant Instagram data files for a single user."""
    base_path = Path(base_dir)
    supported_files = []

    for root, _, files in os.walk(base_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in AUDIO_EXTENSIONS or ext in IMAGE_EXTENSIONS or ext in [".json", ".srt", ".txt"]:
                supported_files.append(str(Path(root) / file))

    return {"files": supported_files, "count": len(supported_files)}


@tool("JSON Processing Tool")
def process_json_file(file_path: str) -> Dict[str, Any]:
    """Processes Instagram JSON files using specialized extraction functions."""
    rel_path = str(file_path).replace("\\", "/")
    docs = []

    try:
        if rel_path.endswith("post_comments_1.json"):
            for c in extract_post_comments(file_path):
                docs.append({"text": c['comment'], "metadata": {"source": rel_path, "type": "comment"}})
        elif rel_path.endswith("personal_information.json"):
            info = extract_personal_information(file_path)
            docs.append({"text": json.dumps(info), "metadata": {"source": rel_path, "type": "personal_info"}})
        elif rel_path.endswith("instagram_friend_map.json"):
            info = extract_friend_map(file_path)
            docs.append({"text": json.dumps(info), "metadata": {"source": rel_path, "type": "friend_map"}})
        elif rel_path.endswith("ads_about_meta.json"):
            info = extract_ads_info(file_path)
            docs.append({"text": json.dumps(info), "metadata": {"source": rel_path, "type": "ads_info"}})
        elif rel_path.endswith("reels.json"):
            for r in extract_reels(file_path):
                docs.append({"text": r.get('title', ''), "metadata": {"source": rel_path, "type": "reel"}})
        elif rel_path.endswith("posts_1.json"):
            for p in extract_posts(file_path):
                docs.append({"text": p.get('title', ''), "metadata": {"source": rel_path, "type": "post"}})
        elif rel_path.endswith("profile_photos.json"):
            for photo in extract_profile_photos(file_path):
                docs.append({"text": photo.get('uri', ''), "metadata": {"source": rel_path, "type": "profile_photo"}})
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs.append({"text": json.dumps(data), "metadata": {"source": rel_path, "type": "json_data"}})
    except Exception as e:
        return {"success": False, "error": str(e)}

    return {"success": True, "documents": docs}


@tool("Text Chunking Tool")
def chunk_documents(documents_json: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
    """Splits text documents into smaller chunks for better vector search."""
    try:
        documents = json.loads(documents_json)
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-3.5-turbo", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunked = []
        for doc in documents:
            chunks = splitter.split_text(doc["text"])
            for i, chunk in enumerate(chunks):
                meta = doc["metadata"].copy()
                meta["chunk_id"] = i
                chunked.append({"text": chunk, "metadata": meta})
        return {"success": True, "chunked_documents": chunked}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool("FAISS Index Builder Per User")
def build_faiss_index_user(chunked_documents_json: str, user_id: str) -> Dict[str, Any]:
    """Builds and saves FAISS index for a specific user."""
    try:
        docs = json.loads(chunked_documents_json)
        texts = [d["text"] for d in docs]
        metas = [d["metadata"] for d in docs]

        db = FAISS.from_texts(texts, embedding_model, metadatas=metas)

        index_dir = Path(f"storage/faiss/{user_id}/faiss_index")
        index_dir.mkdir(parents=True, exist_ok=True)
        db.save_local(str(index_dir))

        with open(Path(f"storage/faiss/{user_id}/docs.pkl"), "wb") as f:
            pickle.dump(docs, f)

        return {"success": True, "index_path": str(index_dir), "total_chunks": len(docs)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def process_user(user_dir: Path):
    """Processes one user's Instagram data into a FAISS index."""
    user_id = user_dir.name
    print(f"\n📂 Processing user: {user_id}")
    files_info = collect_user_files.run(str(user_dir))
    documents = []

    for file_path in files_info["files"]:
        ext = Path(file_path).suffix.lower()
        if ext in AUDIO_EXTENSIONS:
            res = process_audio_file.run(file_path)
            if res["success"]:
                documents.append({"text": res["text"], "metadata": res["metadata"]})
        elif ext == ".srt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = re.sub(r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> .*", "", f.read()).strip()
            documents.append({"text": content, "metadata": {"source": file_path, "type": "srt"}})
        elif ext == ".json":
            res = process_json_file.run(file_path)
            if res["success"]:
                documents.extend(res["documents"])
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({"text": f.read(), "metadata": {"source": file_path, "type": "txt"}})

    chunked = chunk_documents.run(json.dumps(documents))
    if chunked["success"]:
        build_faiss_index_user.run(json.dumps(chunked["chunked_documents"]), user_id)


def main():
    for user_dir in BASE_DIRS:
        if user_dir.exists():
            process_user(user_dir)


if __name__ == "__main__":
    main()
