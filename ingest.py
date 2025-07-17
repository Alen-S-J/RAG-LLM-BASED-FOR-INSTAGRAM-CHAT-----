import os
import re
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from pydantic import BaseModel, Field

# Existing imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

# Import your existing extraction functions
from post_comments_1 import extract_post_comments
from personal_information import extract_personal_information
from instagram_friend_map import extract_friend_map
from ads_about_meta import extract_ads_info
from reels import extract_reels
from posts_1 import extract_posts
from profile_photos import extract_profile_photos

load_dotenv()

BASE_DIRS = [
    Path("data/instagram-days010601-2025-07-09-GyNYqGsQ"),
    Path("data/instagram-days010602-2025-07-11-cdZXWvWZ"),
    Path("data/instagram-days010603-2025-07-11-5uV1CjIS")
]

SRT_DIRS = [base / "media" / "reels" / "202507" for base in BASE_DIRS]
POSTS_DIRS = [base / "your_instagram_activity" / "media" / "posts" / "202507" for base in BASE_DIRS]
PHOTOS_DIRS = [base / "your_instagram_activity" / "media" / "profile" / "202507" for base in BASE_DIRS]

FAISS_DIR = Path("storage/faiss/faiss_index")
DOCS_FILE = Path("storage/faiss/docs.pkl")
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]

# Pydantic models for structured data
class ProcessedDocument(BaseModel):
    text: str
    metadata: Dict[str, Any]

class ProcessingResult(BaseModel):
    documents: List[ProcessedDocument]
    file_count: int
    document_count: int
    chunk_count: int

class IndexBuildResult(BaseModel):
    success: bool
    index_path: str
    total_chunks: int
    data_types: List[str]
    message: str

# Custom Tools using the @tool decorator
@tool("File Collection Tool")
def collect_instagram_files(base_dir: str) -> Dict[str, Any]:
    """Collects all supported Instagram data files from the base directory"""
    base_path = Path(base_dir)
    supported_files = []
    
    for root, _, files in os.walk(base_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext == ".srt" or ext in IMAGE_EXTENSIONS or ext == ".json":
                supported_files.append(str(Path(root) / file))
    
    # Add specific directories
    for srt_dir in SRT_DIRS:
        if srt_dir.exists():
            supported_files.extend([str(p) for p in srt_dir.glob("*.srt")])

    for post_dir in POSTS_DIRS:
        if post_dir.exists():
            supported_files.extend([str(p) for p in post_dir.glob("*") if p.suffix.lower() == ".json"])

    for photo_dir in PHOTOS_DIRS:
        if photo_dir.exists():
            supported_files.extend([str(p) for p in photo_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])

    
    return {
        "files": supported_files,
        "count": len(supported_files),
        "types": list(set(Path(f).suffix.lower() for f in supported_files))
    }

@tool("SRT Processing Tool")
def process_srt_file(file_path: str) -> Dict[str, Any]:
    """Processes SRT subtitle files and extracts clean text content"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Clean SRT content
        cleaned_content = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n', '', content)
        cleaned_content = re.sub(r'\d+\n', '', cleaned_content)
        cleaned_content = re.sub(r'\n+', '\n', cleaned_content).strip()
        
        return {
            "success": True,
            "text": cleaned_content,
            "metadata": {
                "source": file_path,
                "type": "srt",
                "filename": Path(file_path).name
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }

@tool("JSON Processing Tool")
def process_json_file(file_path: str) -> Dict[str, Any]:
    """Processes Instagram JSON files using specialized extraction functions"""
    rel_path = str(file_path).replace("\\", "/")
    docs = []
    
    try:
        if rel_path.endswith("post_comments_1.json"):
            for c in extract_post_comments(file_path):
                docs.append({
                    "text": f"Instagram Comment: {c['comment']}",
                    "metadata": {
                        "source": rel_path,
                        "media_owner": c.get("media_owner", ""),
                        "timestamp": c.get("timestamp", ""),
                        "type": "comment"
                    }
                })
        
        elif rel_path.endswith("personal_information.json"):
            info = extract_personal_information(file_path)
            docs.append({
                "text": f"Personal Information: {json.dumps(info)}",
                "metadata": {
                    "source": rel_path,
                    "type": "personal_info"
                }
            })
        
        elif rel_path.endswith("instagram_friend_map.json"):
            info = extract_friend_map(file_path)
            docs.append({
                "text": f"Friend Map: {json.dumps(info)}",
                "metadata": {
                    "source": rel_path,
                    "type": "friend_map"
                }
            })
        
        
        elif rel_path.endswith("ads_about_meta.json"):
            info = extract_ads_info(file_path)
            docs.append({
                "text": f"Ads Info: {json.dumps(info)}",
                "metadata": {
                    "source": rel_path,
                    "fbid": info.get("fbid", ""),
                    "type": "ads_info"
                }
            })
        
        elif rel_path.endswith("reels.json"):
            for r in extract_reels(file_path):
                docs.append({
                    "text": f"Instagram Reel: {r.get('title', 'No title')}",
                    "metadata": {
                        "source": rel_path,
                        "uri": r.get("uri", ""),
                        "creation_timestamp": r.get("creation_timestamp", ""),
                        "subtitle_uri": r.get("subtitle_uri", ""),
                        "type": "reel"
                    }
                })
        
        elif rel_path.endswith("posts_1.json"):
            for p in extract_posts(file_path):
                docs.append({
                    "text": f"Instagram Post: {p.get('title', 'No title')}",
                    "metadata": {
                        "source": rel_path,
                        "creation_timestamp": p.get("creation_timestamp", ""),
                        "type": "post"
                    }
                })
        
        elif rel_path.endswith("profile_photos.json"):
            for photo in extract_profile_photos(file_path):
                docs.append({
                    "text": f"Profile Photo: {photo.get('uri', 'No URI')}",
                    "metadata": {
                        "source": rel_path,
                        "creation_timestamp": photo.get("creation_timestamp", ""),
                        "type": "profile_photo"
                    }
                })
        
        else:
            # Generic JSON file processing
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs.append({
                "text": f"JSON Data: {json.dumps(data)}",
                "metadata": {
                    "source": rel_path,
                    "type": "json_data"
                }
            })
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }
    
    return {
        "success": True,
        "documents": docs,
        "count": len(docs)
    }

@tool("Text Chunking Tool")
def chunk_documents(documents_json: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
    """Splits documents into smaller chunks for better retrieval performance"""
    try:
        documents = json.loads(documents_json)
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o-mini",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        chunked_docs = []
        
        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                
                chunked_docs.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })
        
        return {
            "success": True,
            "chunked_documents": chunked_docs,
            "original_count": len(documents),
            "chunk_count": len(chunked_docs)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to chunk documents: {str(e)}"
        }

@tool("FAISS Index Builder")
def build_faiss_index(chunked_documents_json: str) -> Dict[str, Any]:
    """Builds and saves FAISS vector index from processed documents"""
    try:
        chunked_documents = json.loads(chunked_documents_json)
        
        # Extract texts and metadata
        texts = [doc["text"] for doc in chunked_documents]
        metadatas = [doc["metadata"] for doc in chunked_documents]
        
        # Build FAISS index
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        
        # Save index and documents
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        db.save_local(str(FAISS_DIR))
        
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(chunked_documents, f)
        
        data_types = list(set(m['type'] for m in metadatas))
        
        return {
            "success": True,
            "index_path": str(FAISS_DIR),
            "total_chunks": len(chunked_documents),
            "data_types": data_types,
            "message": f"Successfully indexed {len(chunked_documents)} chunks from {len(data_types)} different data types"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to build FAISS index: {str(e)}"
        }

# CrewAI System
class InstagramDataProcessorCrew:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.setup_agents()
        self.setup_tasks()
    
    def setup_agents(self):
        """Initialize specialized agents for different processing tasks"""
        
        # File Discovery Agent
        self.file_agent = Agent(
            role="File Discovery Specialist",
            goal="Discover and catalog all Instagram data files for processing",
            backstory="You are an expert at finding and organizing Instagram data files. "
                     "You understand the structure of Instagram data exports and can identify "
                     "all relevant files including JSON data, SRT subtitles, and media files.",
            tools=[collect_instagram_files],
            verbose=True
        )
        
        # Content Processing Agent
        self.content_agent = Agent(
            role="Content Processing Expert",
            goal="Extract and process content from various Instagram data file types",
            backstory="You specialize in processing different types of Instagram data files. "
                     "You can handle SRT files, JSON data, and media files, extracting "
                     "meaningful content while preserving important metadata.",
            tools=[process_srt_file, process_json_file],
            verbose=True
        )
        
        # Text Chunking Agent
        self.chunking_agent = Agent(
            role="Text Chunking Specialist",
            goal="Optimize text documents for vector search by creating appropriate chunks",
            backstory="You are an expert at preparing text for vector search. You understand "
                     "how to split large documents into meaningful chunks that preserve context "
                     "while optimizing for retrieval performance.",
            tools=[chunk_documents],
            verbose=True
        )
        
        # Vector Index Agent
        self.index_agent = Agent(
            role="Vector Index Engineer",
            goal="Build and optimize FAISS vector indexes for efficient similarity search",
            backstory="You are a specialist in building high-performance vector indexes. "
                     "You understand how to create FAISS indexes that provide fast and "
                     "accurate similarity search capabilities for Instagram data.",
            tools=[build_faiss_index],
            verbose=True
        )
    
    def setup_tasks(self):
        """Define the sequence of tasks for processing Instagram data"""
        
        # Task 1: File Discovery
        self.file_discovery_task = Task(
            description=f"Discover all Instagram data files in the directory: {BASE_DIRS}. "
                       f"Use the file collection tool to identify JSON files, SRT subtitle files, and media files. "
                       f"Provide a comprehensive catalog of all files to be processed.",
            agent=self.file_agent,
            expected_output="A complete list of all Instagram data files with their types and paths in JSON format"
        )
        
        # Task 2: Content Processing
        self.content_processing_task = Task(
            description="Process all discovered files and extract meaningful content. "
                       "For each file, use the appropriate processing tool based on file type. "
                       "For JSON files, use the JSON processing tool. "
                       "For SRT files, use the SRT processing tool. "
                       "Ensure all content is properly formatted with metadata.",
            agent=self.content_agent,
            expected_output="A collection of processed documents with extracted content and metadata in JSON format"
        )
        
        # Task 3: Text Chunking
        self.chunking_task = Task(
            description="Split the processed documents into optimal chunks for vector search. "
                       "Use the chunking tool to ensure chunks preserve context while being appropriately sized. "
                       "Maintain metadata continuity across chunks.",
            agent=self.chunking_agent,
            expected_output="A collection of document chunks optimized for vector search in JSON format"
        )
        
        # Task 4: Index Building
        self.index_building_task = Task(
            description="Build a FAISS vector index from the chunked documents. "
                       "Use the FAISS index builder tool to create embeddings for all text chunks. "
                       "Save the index and document metadata for future retrieval.",
            agent=self.index_agent,
            expected_output="A complete FAISS index with saved metadata and performance metrics"
        )
    
    def run_processing(self) -> IndexBuildResult:
        """Execute the complete Instagram data processing pipeline"""
        
        # Create the crew
        crew = Crew(
            agents=[self.file_agent, self.content_agent, self.chunking_agent, self.index_agent],
            tasks=[self.file_discovery_task, self.content_processing_task, self.chunking_task, self.index_building_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute the crew
        print("🚀 Starting Instagram Data Processing Crew...")
        result = crew.kickoff()
        
        # Parse the final result
        try:
            if isinstance(result, dict):
                return IndexBuildResult(**result)
            else:
                # If result is a string, try to parse it or create a basic result
                return IndexBuildResult(
                    success=True,
                    index_path=str(FAISS_DIR),
                    total_chunks=0,
                    data_types=[],
                    message=str(result)
                )
        except Exception as e:
            return IndexBuildResult(
                success=False,
                index_path="",
                total_chunks=0,
                data_types=[],
                message=f"Processing completed but result parsing failed: {str(e)}"
            )

def main():
    """Main function to run the Instagram data processing crew"""
    print("🤖 Instagram Data Processor with CrewAI")
    print("=" * 50)
    
    # Initialize the crew
    processor = InstagramDataProcessorCrew(chunk_size=500, chunk_overlap=50)
    
    # Run the processing pipeline
    result = processor.run_processing()
    
    # Display results
    print("\n" + "=" * 50)
    print("📊 Processing Results:")
    print(f"Success: {result.success}")
    print(f"Index Path: {result.index_path}")
    print(f"Total Chunks: {result.total_chunks}")
    print(f"Data Types: {', '.join(result.data_types)}")
    print(f"Message: {result.message}")
    
    if result.success:
        print("\n✅ Instagram data processing completed successfully!")
        print(f"🔍 FAISS index saved to: {result.index_path}")
        print(f"📁 Document metadata saved to: {DOCS_FILE}")
    else:
        print("\n❌ Processing failed. Check the logs for details.")

if __name__ == "__main__":
    main()