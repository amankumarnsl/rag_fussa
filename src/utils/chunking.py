"""
Chunking strategies for different file types
"""

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks.
    
    Args:
        text (str): Text to split
        chunk_size (int): Maximum number of words per chunk
        overlap (int): Number of words to overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def chunk_by_sentences(text, max_sentences=10):
    """
    Split text into chunks by sentences.
    
    Args:
        text (str): Text to split
        max_sentences (int): Maximum sentences per chunk
    
    Returns:
        list: List of text chunks
    """
    import re
    
    # Simple sentence splitting (can be improved)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk_sentences = sentences[i:i + max_sentences]
        chunk = ". ".join(chunk_sentences) + "."
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def chunk_by_paragraphs(text):
    """
    Split text into chunks by paragraphs.
    
    Args:
        text (str): Text to split
    
    Returns:
        list: List of text chunks (paragraphs)
    """
    paragraphs = text.split('\n\n')
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks


def smart_chunk(text, chunk_size=1000, overlap=200, strategy="words"):
    """
    Smart chunking that chooses the best strategy.
    
    Args:
        text (str): Text to split
        chunk_size (int): Target chunk size
        overlap (int): Overlap between chunks
        strategy (str): Chunking strategy - "words", "sentences", "paragraphs"
    
    Returns:
        list: List of text chunks
    """
    if strategy == "sentences":
        return chunk_by_sentences(text, max_sentences=chunk_size//50)  # Rough estimate
    elif strategy == "paragraphs":
        return chunk_by_paragraphs(text)
    else:
        return split_text_into_chunks(text, chunk_size, overlap)
