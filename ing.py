# ingest.py
from pathlib import Path
from transcript import VideoTranscriptExtractor
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import pickle
from dotenv import load_dotenv

load_dotenv()

# Constants – match app.py
FAISS_DB_PATH = Path("storage/faiss/faiss_index")
DOCS_FILE_PATH = Path("storage/faiss/docs.pkl")
VIDEO_BASE_DIR = Path("data")
TRANSCRIPT_BASE_DIR = Path("video_transcripts")

def get_video_directories():
    """Find all subdirectories in VIDEO_BASE_DIR."""
    return [p for p in VIDEO_BASE_DIR.iterdir() if p.is_dir()]

def build_faiss_index():
    """Build FAISS index from transcript .txt files."""
    docs = []

    # Include all transcript directories and ALL_TRANSCRIPTS_COMBINED.txt
    transcript_paths = [p for p in TRANSCRIPT_BASE_DIR.iterdir() if p.is_dir()]
    combined_file = TRANSCRIPT_BASE_DIR / "ALL_TRANSCRIPTS_COMBINED.txt"
    if combined_file.exists():
        transcript_paths.append(combined_file)

    for path in transcript_paths:
        if path.is_dir():
            txt_files = [f for f in path.rglob("*.txt") if "transcription_summary.txt" not in f.name]
        elif path.is_file() and path.suffix.lower() == ".txt":
            txt_files = [path] if "transcription_summary.txt" not in path.name else []
        else:
            txt_files = []

        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = splitter.split_text(content)

                for i, chunk in enumerate(chunks):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": str(txt_file),
                            "url": None,  # Placeholder if you want to add a video URL later
                            "chunk": i
                        }
                    ))
            except Exception as e:
                print(f"⚠ Error reading {txt_file}: {e}")

    if not docs:
        print("⚠ No transcript files found for FAISS indexing.")
        return

    print(f"🔍 Embedding {len(docs)} transcript chunks using OpenAI...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("📦 Creating FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Ensure directory exists
    FAISS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving FAISS index to: {FAISS_DB_PATH}")
    vectorstore.save_local(str(FAISS_DB_PATH))

    print(f"💾 Saving docs metadata to: {DOCS_FILE_PATH}")
    with open(DOCS_FILE_PATH, "wb") as f:
        pickle.dump(docs, f)

    print("✅ FAISS indexing complete.")

def main():
    video_dirs = get_video_directories()
    if not video_dirs:
        print(f"⚠ No video directories found in {VIDEO_BASE_DIR}")
        return

    extractor = VideoTranscriptExtractor(video_dirs)

    print("🎥 Starting video transcription...")
    total_processed = extractor.process_all_directories()

    if total_processed > 0:
        extractor.create_combined_transcript_file()
        extractor.create_summary_report()
        print(f"✅ Processed {total_processed} video(s).")

    print("\n📂 Building FAISS index from transcripts...")
    build_faiss_index()

if __name__ == "__main__":
    main()
