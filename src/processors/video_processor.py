"""
Video processing functionality
"""
import os
import tempfile
import asyncio
import subprocess
from ..utils.smart_chunking import process_extracted_text
from ..utils.cpu_config import run_cpu_task


def extract_video_metadata(video_content, file_name):
    """
    Extract basic metadata from video file.
    
    Args:
        video_content (bytes): Video file content
        file_name (str): Name of the file
        
    Returns:
        dict: Video metadata
    """
    try:
        # For now, return basic info
        # In future, can use ffprobe or similar tools
        metadata = {
            "file_name": file_name,
            "file_size": len(video_content),
            "file_type": "video",
            "format": file_name.split('.')[-1].lower() if '.' in file_name else "unknown"
        }
        
        return metadata
        
    except Exception as e:
        raise Exception(f"Failed to extract video metadata: {str(e)}")


def extract_video_frames(video_content, file_name, max_frames=10):
    """
    Extract frames from video for analysis.
    
    Args:
        video_content (bytes): Video file content
        file_name (str): Name of the file
        max_frames (int): Maximum number of frames to extract
        
    Returns:
        list: List of frame information
    """
    try:
        # Placeholder for frame extraction
        # In future implementation, you could:
        # 1. Use OpenCV to extract frames
        # 2. Use ffmpeg to extract frames
        # 3. Analyze frames with vision models
        
        frames_info = []
        for i in range(min(max_frames, 5)):  # Simulate extracting 5 frames
            frames_info.append({
                "frame_number": i + 1,
                "timestamp": f"00:00:{i:02d}",
                "description": f"Frame {i + 1} from {file_name}"
            })
        
        return frames_info
        
    except Exception as e:
        raise Exception(f"Failed to extract video frames: {str(e)}")


def _extract_audio_cpu_intensive(video_content, file_name):
    """
    CPU-intensive audio extraction function for multiprocessing.
    
    Args:
        video_content (bytes): Video file content
        file_name (str): Name of the file
        
    Returns:
        tuple: (audio_file_path, audio_info) or (None, error_message)
    """
    import tempfile
    import subprocess
    import os
    
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=f".{file_name.split('.')[-1]}", delete=False) as temp_video:
            temp_video.write(video_content)
            temp_video_path = temp_video.name
        
        # Create temp audio file path
        temp_audio_path = temp_video_path.rsplit('.', 1)[0] + '_audio.wav'
        
        # Extract audio using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', temp_video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            temp_audio_path
        ]
        
        # Run ffmpeg command
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Clean up video temp file
        try:
            os.unlink(temp_video_path)
        except:
            pass
        
        if result.returncode == 0:
            # Get audio file info
            audio_info = {
                "status": "success",
                "audio_file": temp_audio_path,
                "format": "wav",
                "sample_rate": "16000",
                "channels": "1",
                "extracted_from": file_name
            }
            
            # Check if audio file exists and has content
            if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                return temp_audio_path, audio_info
            else:
                return None, "Audio file was created but is empty"
        else:
            error_msg = f"FFmpeg failed: {result.stderr}"
            return None, error_msg
            
    except subprocess.TimeoutExpired:
        return None, "Audio extraction timed out (file too large or processing too slow)"
    except FileNotFoundError:
        return None, "FFmpeg not found. Please install FFmpeg: 'brew install ffmpeg' or 'apt-get install ffmpeg'"
    except Exception as e:
        return None, f"Failed to extract audio: {str(e)}"
    finally:
        # Clean up temp video file if it still exists
        try:
            if 'temp_video_path' in locals():
                os.unlink(temp_video_path)
        except:
            pass


async def extract_audio_from_video(video_content, file_name):
    """
    Extract audio from video file using ffmpeg with multiprocessing.
    
    Args:
        video_content (bytes): Video file content
        file_name (str): Name of the file
        
    Returns:
        tuple: (audio_file_path, audio_info) or (None, error_message)
    """
    return await run_cpu_task(_extract_audio_cpu_intensive, video_content, file_name)


def transcribe_audio_with_assemblyai(audio_path, file_name):
    """
    Transcribe audio using AssemblyAI.
    
    Args:
        audio_path (str): Path to the audio file
        file_name (str): Original filename
        
    Returns:
        dict: Transcription results or error message
    """
    try:
        import assemblyai as aai
        from ..config.config import ASSEMBLYAI_API_KEY
        
        # Set AssemblyAI API key
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        
        # Configure transcription settings
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.universal,
            language_detection=True,  # Auto-detect language
            punctuate=True,  # Add punctuation
            format_text=True  # Format text properly
        )
        
        # Create transcriber and transcribe the audio file
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_path)
        
        # Check transcription status
        if transcript.status == "error":
            return {
                "success": False,
                "error": f"AssemblyAI transcription failed: {transcript.error}",
                "transcript": "",
                "duration": "Unknown",
                "language": "Unknown"
            }
        
        # Extract transcript information
        transcript_text = transcript.text or ""
        duration = getattr(transcript, 'audio_duration', 'Unknown')
        language = getattr(transcript, 'language_code', 'Unknown')
        confidence = getattr(transcript, 'confidence', 'Unknown')
        
        return {
            "success": True,
            "transcript": transcript_text,
            "duration": duration,
            "language": language,
            "confidence": confidence,
            "model": "universal",
            "service": "AssemblyAI"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"AssemblyAI transcription failed: {str(e)}",
            "transcript": "",
            "duration": "Unknown",
            "language": "Unknown"
        }


def extract_video_audio_transcript(video_content, file_name):
    """
    Extract audio from video and transcribe using AssemblyAI.
    
    Args:
        video_content (bytes): Video file content
        file_name (str): Name of the file
        
    Returns:
        str: Transcribed text or error information
    """
    try:
        # Extract audio from video
        audio_path, audio_info = extract_audio_from_video(video_content, file_name)
        
        if audio_path:
            # Audio extracted successfully, now transcribe
            transcription_result = transcribe_audio_with_assemblyai(audio_path, file_name)
            
            if transcription_result["success"]:
                # Transcription successful
                transcript = f"""
[AUDIO TRANSCRIBED FROM {file_name}]

✅ Audio extraction & transcription successful!
🎵 Audio format: {audio_info.get('format', 'wav')}
📊 Sample rate: {audio_info.get('sample_rate', '16000')} Hz
🔊 Channels: {audio_info.get('channels', '1')} (mono)
🗣️  Language: {transcription_result.get('language', 'Unknown')}
⏱️  Duration: {transcription_result.get('duration', 'Unknown')} seconds
🎯 Confidence: {transcription_result.get('confidence', 'Unknown')}
🤖 Model: {transcription_result.get('model', 'universal')} ({transcription_result.get('service', 'AssemblyAI')})

📝 TRANSCRIPT:
{transcription_result['transcript']}
"""
                
                # Clean up the temporary audio file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                    
                return transcript.strip()
            else:
                # Transcription failed
                transcript = f"""
[AUDIO EXTRACTION SUCCESSFUL, TRANSCRIPTION FAILED FROM {file_name}]

✅ Audio extraction successful!
❌ Transcription error: {transcription_result.get('error', 'Unknown error')}

📁 Audio file: {os.path.basename(audio_path)}
🎵 Format: {audio_info.get('format', 'wav')}
📊 Sample rate: {audio_info.get('sample_rate', '16000')} Hz

💡 Possible issues:
1. AssemblyAI API key not set or invalid
2. Audio file too large or corrupted
3. No speech content in audio
4. Network connectivity issues
5. Insufficient AssemblyAI credits
"""
                
                # Clean up the temporary audio file
                try:
                    os.unlink(audio_path)
                except:
                    pass
                    
                return transcript.strip()
        else:
            # Audio extraction failed
            error_msg = audio_info if isinstance(audio_info, str) else "Unknown error"
            transcript = f"""
[AUDIO EXTRACTION FAILED FROM {file_name}]

❌ Error: {error_msg}

📝 Fallback: Using video metadata only
🎬 Video format: {file_name.split('.')[-1].upper()}
⚠️  No audio content available for transcription

💡 To fix this issue:
1. Install FFmpeg: 'brew install ffmpeg' (macOS) or 'apt-get install ffmpeg' (Linux)
2. Ensure video file has audio track
3. Check video file is not corrupted
"""
            return transcript.strip()
        
    except Exception as e:
        return f"[AUDIO PROCESSING ERROR] Failed to process audio from {file_name}: {str(e)}"


async def process_video(video_content, file_name, chunk_strategy="semantic"):
    """
    Process video file: extract audio → transcribe → save to .txt → return path for common processing.
    
    Args:
        video_content (bytes): Video file content
        file_name (str): Name of the file
        chunk_strategy (str): Chunking strategy - "semantic", "hierarchical", "markdown", "simple"
        
    Returns:
        str: Path to saved transcript text file for common processing pipeline
    """
    try:
        # Extract metadata (for logging)
        metadata = extract_video_metadata(video_content, file_name)
        print(f"🎥 Video processing: {file_name} ({metadata.get('format', 'unknown')} format)")
        
        # Extract and transcribe audio to get transcript text
        transcript = extract_video_audio_transcript(video_content, file_name)
        
        if not transcript.strip():
            raise Exception("No transcript content extracted from video")
        
        print(f"📊 Video transcript extracted: {len(transcript)} characters")
        
        # Save transcript to text file
        from ..utils.smart_chunking import save_extracted_text
        text_filepath = save_extracted_text(transcript, file_name, "video")
        
        if not text_filepath:
            raise Exception("Failed to save transcript text file")
        
        return text_filepath
        
    except Exception as e:
        raise Exception(f"Failed to process video: {str(e)}")


def get_supported_video_formats():
    """
    Get list of supported video formats.
    
    Returns:
        list: List of supported video file extensions
    """
    return ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'm4v']


def is_video_file(file_extension):
    """
    Check if file extension is a supported video format.
    
    Args:
        file_extension (str): File extension
        
    Returns:
        bool: True if supported video format
    """
    return file_extension.lower() in get_supported_video_formats()
