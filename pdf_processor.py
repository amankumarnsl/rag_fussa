"""
PDF processing functionality
"""
import io
import PyPDF2
from chunking import smart_chunk


def extract_pdf_text(pdf_content):
    """
    Extract text from PDF content.
    
    Args:
        pdf_content (bytes): PDF file content
        
    Returns:
        str: Extracted text from PDF
    """
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text + "\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def process_pdf(pdf_content, file_name, chunk_strategy="words"):
    """
    Process PDF file and return chunks with metadata.
    
    Args:
        pdf_content (bytes): PDF file content
        file_name (str): Name of the file
        chunk_strategy (str): Chunking strategy to use
        
    Returns:
        list: List of text chunks with metadata
    """
    try:
        # Extract text from PDF
        text = extract_pdf_text(pdf_content)
        
        if not text.strip():
            raise Exception("No text content found in PDF")
        
        # Create chunks
        chunks = smart_chunk(text, chunk_size=1000, overlap=200, strategy=chunk_strategy)
        
        # Add metadata to chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "content": chunk,
                "metadata": {
                    "file_name": file_name,
                    "file_type": "pdf",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "word_count": len(chunk.split())
                }
            })
        
        return processed_chunks
        
    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")


def get_pdf_info(pdf_content):
    """
    Get basic information about the PDF.
    
    Args:
        pdf_content (bytes): PDF file content
        
    Returns:
        dict: PDF information
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        info = {
            "total_pages": len(pdf_reader.pages),
            "has_text": False,
            "metadata": {}
        }
        
        # Check if PDF has extractable text
        for page in pdf_reader.pages:
            if page.extract_text().strip():
                info["has_text"] = True
                break
        
        # Get PDF metadata if available
        if pdf_reader.metadata:
            info["metadata"] = {
                "title": pdf_reader.metadata.get("/Title", ""),
                "author": pdf_reader.metadata.get("/Author", ""),
                "subject": pdf_reader.metadata.get("/Subject", ""),
                "creator": pdf_reader.metadata.get("/Creator", "")
            }
        
        return info
        
    except Exception as e:
        return {"error": f"Failed to get PDF info: {str(e)}"}
