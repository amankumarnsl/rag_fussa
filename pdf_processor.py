"""
PDF processing functionality
"""
import io
import fitz  # PyMuPDF
from smart_chunking import process_extracted_text


def extract_pdf_text(pdf_content):
    """
    Extract text from PDF content using PyMuPDF.
    
    Args:
        pdf_content (bytes): PDF file content
        
    Returns:
        str: Extracted text from PDF
    """
    try:
        text = ""
        
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text + "\n"
        
        pdf_document.close()
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def process_pdf(pdf_content, file_name, chunk_strategy="semantic"):
    """
    Process PDF file: extract text → save to .txt → return path for common processing.
    
    Args:
        pdf_content (bytes): PDF file content
        file_name (str): Name of the file
        chunk_strategy (str): Chunking strategy - "semantic", "hierarchical", "markdown", "simple"
        
    Returns:
        str: Path to saved text file for common processing pipeline
    """
    try:
        # Extract text from PDF
        text = extract_pdf_text(pdf_content)
        
        if not text.strip():
            raise Exception("No text content found in PDF")
        
        print(f"📄 PDF text extracted: {len(text)} characters")
        
        # Save extracted text to file
        from smart_chunking import save_extracted_text
        text_filepath = save_extracted_text(text, file_name, "pdf")
        
        if not text_filepath:
            raise Exception("Failed to save extracted text file")
        
        return text_filepath
        
    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")


def get_pdf_info(pdf_content):
    """
    Get basic information about the PDF using PyMuPDF.
    
    Args:
        pdf_content (bytes): PDF file content
        
    Returns:
        dict: PDF information
    """
    
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        info = {
            "total_pages": pdf_document.page_count,
            "has_text": False,
            "metadata": {}
        }
        
        # Check if PDF has extractable text
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            if page.get_text().strip():
                info["has_text"] = True
                break
        
        # Get PDF metadata if available
        metadata = pdf_document.metadata
        if metadata:
            info["metadata"] = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", "")
            }
        
        pdf_document.close()
        return info
        
    except Exception as e:
        return {"error": f"Failed to get PDF info: {str(e)}"}
