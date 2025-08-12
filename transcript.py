import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import subprocess
import tempfile
import time

# Import required libraries for audio processing and speech recognition
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Some libraries not installed. Install with:")
    print("pip install SpeechRecognition pydub openai-whisper")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoTranscriptExtractor:
    """Extract transcripts from video files and save as .txt files"""
    
    def __init__(self, base_dirs: List[Path]):
        self.base_dirs = base_dirs
        self.output_dir = Path("video_transcripts")
        self.temp_dir = Path("temp_audio")
        
        # Supported video formats
        self.video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        
        # Speech recognition methods
        self.recognition_methods = ['whisper', 'google', 'sphinx']
        self.preferred_method = 'whisper' if WHISPER_AVAILABLE else 'google'
        
        # Initialize Whisper model if available
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            try:
                logger.info("Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {e}")
                self.preferred_method = 'google'
    
    def setup_directories(self):
        """Create output and temporary directories"""
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        logger.info(f"Directories created: {self.output_dir}, {self.temp_dir}")
    
    def find_video_files(self, directory: Path) -> List[Path]:
        """Find all video files in a directory"""
        video_files = []
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return video_files
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.video_formats:
                video_files.append(file_path)
        
        logger.info(f"Found {len(video_files)} video files in {directory}")
        return video_files
    
    def extract_audio_from_video(self, video_path: Path) -> Optional[Path]:
        """Extract audio from video file using ffmpeg"""
        try:
            # Create temporary audio file path
            audio_filename = f"{video_path.stem}_audio.wav"
            audio_path = self.temp_dir / audio_filename
            
            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz sample rate for better compatibility
                '-ac', '1',      # Convert to mono
                '-y',            # Overwrite output file
                str(audio_path)
            ]
            
            logger.info(f"Extracting audio from {video_path.name}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and audio_path.exists():
                logger.info(f"Audio extracted successfully: {audio_path}")
                return audio_path
            else:
                logger.error(f"FFmpeg failed for {video_path}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Audio extraction timeout for {video_path}")
            return None
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {e}")
            return None
    
    def transcribe_with_whisper(self, audio_path: Path) -> str:
        """Transcribe audio using Whisper"""
        try:
            if not self.whisper_model:
                return ""
            
            logger.info(f"Transcribing with Whisper: {audio_path.name}")
            result = self.whisper_model.transcribe(str(audio_path))
            return result["text"].strip()
            
        except Exception as e:
            logger.error(f"Whisper transcription failed for {audio_path}: {e}")
            return ""
    
    def transcribe_with_google(self, audio_path: Path) -> str:
        """Transcribe audio using Google Speech Recognition"""
        try:
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(str(audio_path)) as source:
                logger.info(f"Loading audio file: {audio_path.name}")
                audio_data = recognizer.record(source)
            
            logger.info("Transcribing with Google Speech Recognition...")
            text = recognizer.recognize_google(audio_data)
            return text.strip()
            
        except sr.UnknownValueError:
            logger.warning(f"Google could not understand audio: {audio_path}")
            return ""
        except sr.RequestError as e:
            logger.error(f"Google API error for {audio_path}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Google transcription failed for {audio_path}: {e}")
            return ""
    
    def transcribe_with_sphinx(self, audio_path: Path) -> str:
        """Transcribe audio using CMU Sphinx (offline)"""
        try:
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(str(audio_path)) as source:
                audio_data = recognizer.record(source)
            
            logger.info("Transcribing with CMU Sphinx...")
            text = recognizer.recognize_sphinx(audio_data)
            return text.strip()
            
        except sr.UnknownValueError:
            logger.warning(f"Sphinx could not understand audio: {audio_path}")
            return ""
        except sr.RequestError as e:
            logger.error(f"Sphinx error for {audio_path}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Sphinx transcription failed for {audio_path}: {e}")
            return ""
    
    def split_audio_for_long_files(self, audio_path: Path, chunk_length_ms: int = 60000) -> List[Path]:
        """Split long audio files into chunks for better processing"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            chunks = []
            
            # If audio is shorter than chunk length, return original
            if len(audio) <= chunk_length_ms:
                return [audio_path]
            
            logger.info(f"Splitting audio into chunks: {audio_path.name}")
            
            for i, chunk_start in enumerate(range(0, len(audio), chunk_length_ms)):
                chunk_end = min(chunk_start + chunk_length_ms, len(audio))
                chunk = audio[chunk_start:chunk_end]
                
                chunk_path = self.temp_dir / f"{audio_path.stem}_chunk_{i:03d}.wav"
                chunk.export(str(chunk_path), format="wav")
                chunks.append(chunk_path)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting audio {audio_path}: {e}")
            return [audio_path]
    
    def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio file using the best available method"""
        # For long files, split into chunks
        audio_chunks = self.split_audio_for_long_files(audio_path)
        transcripts = []
        
        for chunk_path in audio_chunks:
            chunk_transcript = ""
            
            # Try different transcription methods
            if self.preferred_method == 'whisper' and self.whisper_model:
                chunk_transcript = self.transcribe_with_whisper(chunk_path)
            
            if not chunk_transcript and self.preferred_method != 'whisper':
                if self.preferred_method == 'google':
                    chunk_transcript = self.transcribe_with_google(chunk_path)
                elif self.preferred_method == 'sphinx':
                    chunk_transcript = self.transcribe_with_sphinx(chunk_path)
            
            # Fallback to other methods if primary method fails
            if not chunk_transcript:
                logger.info(f"Primary method failed, trying fallbacks for {chunk_path.name}")
                for method in self.recognition_methods:
                    if method != self.preferred_method and not chunk_transcript:
                        if method == 'whisper' and self.whisper_model:
                            chunk_transcript = self.transcribe_with_whisper(chunk_path)
                        elif method == 'google':
                            chunk_transcript = self.transcribe_with_google(chunk_path)
                        elif method == 'sphinx':
                            chunk_transcript = self.transcribe_with_sphinx(chunk_path)
            
            if chunk_transcript:
                transcripts.append(chunk_transcript)
            
            # Clean up chunk files if they were created
            if chunk_path != audio_path:
                try:
                    chunk_path.unlink()
                except:
                    pass
        
        return ' '.join(transcripts).strip()
    
    def process_video_file(self, video_path: Path) -> str:
        """Process a single video file and return transcript"""
        logger.info(f"Processing video: {video_path}")
        
        # Extract audio from video
        audio_path = self.extract_audio_from_video(video_path)
        if not audio_path:
            return ""
        
        try:
            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            return transcript
            
        finally:
            # Clean up temporary audio file
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                    logger.debug(f"Cleaned up temporary file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up {audio_path}: {e}")
    
    def generate_output_filename(self, video_file: Path, base_dir: Path) -> str:
        """Generate output filename for transcript"""
        try:
            rel_path = video_file.relative_to(base_dir)
            output_name = str(rel_path).replace(os.sep, '_').replace('/', '_')
            output_name = Path(output_name).stem + '_transcript.txt'
            return output_name
        except ValueError:
            return video_file.stem + '_transcript.txt'
    
    def process_directory(self, base_dir: Path):
        """Process all video files in a base directory"""
        logger.info(f"Processing directory: {base_dir}")
        
        # Create subdirectory for this base directory
        dir_name = base_dir.name
        output_subdir = self.output_dir / dir_name
        output_subdir.mkdir(exist_ok=True)
        
        video_files = self.find_video_files(base_dir)
        
        if not video_files:
            logger.info(f"No video files found in {base_dir}")
            return 0
        
        processed_count = 0
        failed_files = []
        
        for video_file in video_files:
            try:
                logger.info(f"Processing {video_file.name} ({processed_count + 1}/{len(video_files)})")
                
                # Extract transcript
                transcript = self.process_video_file(video_file)
                
                if transcript:
                    # Generate output filename
                    output_filename = self.generate_output_filename(video_file, base_dir)
                    output_path = output_subdir / output_filename
                    
                    # Write transcript to file
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(f"Video: {video_file.name}\n")
                        f.write(f"Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Method: {self.preferred_method}\n")
                        f.write("-" * 50 + "\n\n")
                        f.write(transcript)
                    
                    logger.info(f"Transcript saved: {output_path}")
                    processed_count += 1
                else:
                    logger.warning(f"No transcript generated for: {video_file}")
                    failed_files.append(video_file.name)
                    
            except Exception as e:
                logger.error(f"Failed to process {video_file}: {e}")
                failed_files.append(video_file.name)
        
        logger.info(f"Processed {processed_count}/{len(video_files)} files from {base_dir}")
        if failed_files:
            logger.warning(f"Failed files: {failed_files}")
            
        return processed_count
    
    def process_all_directories(self):
        """Process all base directories"""
        self.setup_directories()
        
        # Check dependencies
        if not self.check_dependencies():
            return 0
        
        total_processed = 0
        for base_dir in self.base_dirs:
            count = self.process_directory(base_dir)
            total_processed += count
        
        # Cleanup temp directory
        self.cleanup_temp_files()
        
        logger.info(f"Total video files processed: {total_processed}")
        return total_processed
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        # Check for ffmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("FFmpeg not found. Please install FFmpeg.")
                return False
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg.")
            print("Install FFmpeg from: https://ffmpeg.org/download.html")
            return False
        
        # Check for speech recognition libraries
        if not WHISPER_AVAILABLE:
            logger.warning("Speech recognition libraries not fully available")
            print("For better results, install: pip install SpeechRecognition pydub openai-whisper")
        
        return True
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for temp_file in self.temp_dir.glob('*'):
                temp_file.unlink()
            self.temp_dir.rmdir()
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")
    
    def create_combined_transcript_file(self):
        """Create one master .txt file containing all individual transcripts"""
        combined_path = self.output_dir / "ALL_TRANSCRIPTS_COMBINED.txt"
        
        logger.info("Creating combined transcript file...")
        
        with open(combined_path, 'w', encoding='utf-8') as combined_file:
            # Write header
            combined_file.write("COMBINED VIDEO TRANSCRIPTS\n")
            combined_file.write("=" * 80 + "\n")
            combined_file.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            combined_file.write(f"Transcription method: {self.preferred_method}\n")
            combined_file.write("=" * 80 + "\n\n")
            
            total_transcripts = 0
            
            # Process each base directory
            for base_dir in self.base_dirs:
                dir_name = base_dir.name
                output_subdir = self.output_dir / dir_name
                
                if not output_subdir.exists():
                    continue
                
                # Write directory header
                combined_file.write(f"\n{'#' * 60}\n")
                combined_file.write(f"DIRECTORY: {base_dir}\n")
                combined_file.write(f"{'#' * 60}\n\n")
                
                # Get all transcript files in this directory
                txt_files = sorted(output_subdir.glob("*_transcript.txt"))
                
                for txt_file in txt_files:
                    try:
                        # Read the transcript file
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        
                        if content:
                            # Write separator and file info
                            combined_file.write(f"\n{'-' * 50}\n")
                            combined_file.write(f"TRANSCRIPT FILE: {txt_file.name}\n")
                            combined_file.write(f"SOURCE DIRECTORY: {dir_name}\n")
                            combined_file.write(f"FULL PATH: {txt_file}\n")
                            combined_file.write(f"{'-' * 50}\n\n")
                            
                            # Write the actual transcript content
                            combined_file.write(content)
                            combined_file.write("\n\n")
                            
                            total_transcripts += 1
                            logger.debug(f"Added to combined file: {txt_file.name}")
                    
                    except Exception as e:
                        logger.error(f"Error reading {txt_file}: {e}")
                        combined_file.write(f"\n[ERROR: Could not read {txt_file.name}: {e}]\n\n")
            
            # Write footer
            combined_file.write(f"\n{'=' * 80}\n")
            combined_file.write(f"SUMMARY: Combined {total_transcripts} transcript files\n")
            combined_file.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            combined_file.write(f"{'=' * 80}\n")
        
        logger.info(f"Combined transcript file created: {combined_path}")
        logger.info(f"Total transcripts combined: {total_transcripts}")
        return combined_path, total_transcripts
    
    def create_summary_report(self):
        """Create a summary report of processed files"""
        summary_path = self.output_dir / "transcription_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Video Transcription Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Transcription method used: {self.preferred_method}\n")
            f.write(f"Whisper available: {WHISPER_AVAILABLE}\n\n")
            
            for base_dir in self.base_dirs:
                dir_name = base_dir.name
                output_subdir = self.output_dir / dir_name
                
                f.write(f"Directory: {base_dir}\n")
                f.write(f"Output: {output_subdir}\n")
                
                if output_subdir.exists():
                    txt_files = list(output_subdir.glob("*.txt"))
                    f.write(f"Transcripts generated: {len(txt_files)}\n")
                    
                    for txt_file in txt_files:
                        f.write(f"  - {txt_file.name}\n")
                else:
                    f.write("No transcripts generated\n")
                
                f.write("\n")
        
        logger.info(f"Summary report created: {summary_path}")


def main():
    """Main function to run the video transcript extractor"""
    
    # Define your base directories with video files
    BASE_DIRS = [
        Path("data/Demo-20250808T144143Z-1-001"),
        Path("data/Demo2")
    ]
    
    # Create extractor instance
    extractor = VideoTranscriptExtractor(BASE_DIRS)
    
    try:
        print("Video Transcript Extractor")
        print("=" * 30)
        print(f"Processing directories: {[str(d) for d in BASE_DIRS]}")
        print(f"Output directory: {extractor.output_dir}")
        print(f"Preferred transcription method: {extractor.preferred_method}")
        print()
        
        # Process all directories
        total_files = extractor.process_all_directories()
        
        # Create combined transcript file
        if total_files > 0:
            combined_path, combined_count = extractor.create_combined_transcript_file()
            print(f"Combined transcript created: {combined_path}")
            print(f"Total transcripts combined: {combined_count}")
        
        # Create summary report
        extractor.create_summary_report()
        
        print(f"\nProcessing complete!")
        print(f"Total video files processed: {total_files}")
        print(f"Transcripts saved to: {extractor.output_dir}")
        
        # Display output structure
        if extractor.output_dir.exists():
            print(f"\nGenerated transcripts:")
            for item in extractor.output_dir.rglob('*.txt'):
                rel_path = item.relative_to(extractor.output_dir)
                print(f"  {rel_path}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        extractor.cleanup_temp_files()
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        print(f"Error: {e}")
        extractor.cleanup_temp_files()


if __name__ == "__main__":
    main()