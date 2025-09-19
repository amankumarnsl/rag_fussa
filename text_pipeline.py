"""
Common text processing pipeline for all file types after text extraction
"""
from smart_chunking import load_text_file, smart_chunk_text


def process_text_file_to_chunks(text_filepath: str, filename: str, file_type: str, chunk_strategy: str = "semantic"):
    """
    Common pipeline: Load text file → Smart chunking → Return chunks for Pinecone.
    
    This is the shared pipeline used by both PDF and video processing after text extraction.
    
    Args:
        text_filepath (str): Path to the extracted text file
        filename (str): Original filename 
        file_type (str): Original file type (pdf, video, image)
        chunk_strategy (str): Chunking strategy to use
        
    Returns:
        list: List of smart chunks ready for Pinecone storage
    """
    try:
        print(f"📄 Starting common text processing pipeline for {filename}")
        
        # Step 1: Load text from saved file
        text_content = load_text_file(text_filepath)
        
        if not text_content.strip():
            raise Exception(f"No content found in text file: {text_filepath}")
        
        print(f"📊 Loaded text: {len(text_content)} characters, {len(text_content.split())} words")
        
        # Step 2: Perform smart semantic chunking
        chunks = smart_chunk_text(text_content, filename, chunk_strategy)
        
        if not chunks:
            raise Exception("Smart chunking failed to produce any chunks")
        
        # Step 3: Add common metadata to all chunks
        for chunk in chunks:
            chunk["metadata"].update({
                "original_file_type": file_type,
                "text_file_path": text_filepath,
                "chunking_strategy": chunk_strategy,
                "processing_pipeline": "common_text_pipeline"
            })
        
        print(f"✅ Common text processing complete: {len(chunks)} chunks ready for Pinecone")
        return chunks
        
    except Exception as e:
        print(f"❌ Common text processing failed: {str(e)}")
        raise Exception(f"Text processing pipeline failed: {str(e)}")


def get_text_file_stats(text_filepath: str):
    """Get statistics about the text file."""
    try:
        content = load_text_file(text_filepath)
        
        stats = {
            "file_path": text_filepath,
            "total_characters": len(content),
            "total_words": len(content.split()),
            "total_lines": len(content.split('\n')),
            "total_paragraphs": len([p for p in content.split('\n\n') if p.strip()]),
            "exists": True
        }
        
        return stats
        
    except Exception as e:
        return {
            "file_path": text_filepath,
            "error": str(e),
            "exists": False
        }
