"""
Smart chunking using LangChain 0.3 for semantic and hierarchical text splitting
"""
import os
from typing import List, Dict, Any
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_openai import OpenAIEmbeddings
from ..config.config import OPENAI_API_KEY


def save_extracted_text(content: str, filename: str, file_type: str) -> str:
    """
    Save extracted text to a .txt file.
    
    Args:
        content (str): Extracted text content
        filename (str): Original filename (e.g., sample_data.pdf)
        file_type (str): Type of file (pdf, video, image)
        
    Returns:
        str: Path to saved text file
    """
    try:
        # Create data_extracted_visualize/extracted_texts directory
        extracted_dir = os.path.join("data_extraction_visualize", "extracted_texts")
        os.makedirs(extracted_dir, exist_ok=True)
        
        # Generate text filename
        base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
        text_filename = f"{base_name}.txt"
        text_filepath = os.path.join(extracted_dir, text_filename)
        
        # Save content to text file
        with open(text_filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Extracted from {filename} ({file_type.upper()})\n\n")
            f.write(content)
        
        print(f"📄 Saved extracted text: {text_filepath}")
        return text_filepath
        
    except Exception as e:
        print(f"❌ Failed to save extracted text: {str(e)}")
        return None


def load_text_file(filepath: str) -> str:
    """
    Load text content from file.
    
    Args:
        filepath (str): Path to text file
        
    Returns:
        str: Text content
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Failed to load text file {filepath}: {str(e)}")
        return ""


def semantic_chunking(text: str, filename: str) -> List[Dict[str, Any]]:
    """
    Perform TRUE semantic chunking using paragraph-level analysis with text-embedding-3-small.
    
    Args:
        text (str): Text to chunk
        filename (str): Original filename
        
    Returns:
        list: List of semantically coherent chunks with metadata
    """
    try:
        import openai
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        print(f"🧠 Starting paragraph-level semantic chunking for {filename}")
        
        # Step 1: Split text into paragraphs (more robust than sentences)
        paragraphs = split_into_paragraphs(text)
        if len(paragraphs) < 2:
            print("⚠️  Too few paragraphs for semantic analysis, using hierarchical chunking")
            return hierarchical_chunking(text, filename)
        
        print(f"📝 Split into {len(paragraphs)} paragraphs")
        
        # Step 2: Generate embeddings for each paragraph using text-embedding-3-small
        print("🔍 Generating paragraph embeddings...")
        paragraph_embeddings = get_paragraph_embeddings(paragraphs)
        
        if not paragraph_embeddings:
            print("❌ Failed to generate embeddings, falling back to hierarchical")
            return hierarchical_chunking(text, filename)
        
        # Step 3: Find semantic boundaries using similarity analysis
        print("🎯 Analyzing semantic boundaries between paragraphs...")
        boundaries = find_paragraph_semantic_boundaries(paragraph_embeddings, paragraphs)
        
        # Step 4: Create chunks based on semantic boundaries
        print("📦 Creating semantic chunks from paragraph groups...")
        chunks = create_paragraph_semantic_chunks(paragraphs, boundaries, filename)
        
        # Step 5: Optimize chunks for size and coherence
        final_chunks = optimize_semantic_chunks(chunks, filename)
        
        print(f"✅ Paragraph-level semantic chunking complete: {len(final_chunks)} coherent chunks created")
        return final_chunks
        
    except Exception as e:
        print(f"❌ Semantic chunking failed: {str(e)}")
        return hierarchical_chunking(text, filename)  # Fallback


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs - more robust than sentence splitting.
    
    Args:
        text (str): Text to split
        
    Returns:
        list: List of paragraphs
    """
    import re
    
    # Preserve original paragraph structure
    # Split on double newlines (standard paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean and filter paragraphs
    cleaned_paragraphs = []
    for para in paragraphs:
        # Clean whitespace but preserve structure
        para = re.sub(r'\n+', ' ', para)  # Replace newlines with spaces
        para = re.sub(r'\s+', ' ', para)  # Normalize whitespace
        para = para.strip()
        
        # Filter meaningful paragraphs
        if len(para) > 20 and len(para.split()) >= 5:  # Minimum viable paragraph
            cleaned_paragraphs.append(para)
    
    return cleaned_paragraphs


def get_paragraph_embeddings(paragraphs: List[str]) -> List[List[float]]:
    """Generate embeddings for paragraphs using text-embedding-3-small."""
    try:
        import openai
        from ..config.config import OPENAI_API_KEY
        
        openai.api_key = OPENAI_API_KEY
        
        # Process in batches to avoid rate limits
        batch_size = 20  # Smaller batches for paragraphs (longer text)
        all_embeddings = []
        
        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i:i + batch_size]
            
            response = openai.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            print(f"   📊 Processed {min(i + batch_size, len(paragraphs))}/{len(paragraphs)} paragraphs")
        
        return all_embeddings
        
    except Exception as e:
        print(f"❌ Failed to generate paragraph embeddings: {str(e)}")
        return []


def safe_cosine_similarity(a, b) -> float:
    """Calculate cosine similarity with numerical stability."""
    try:
        import numpy as np
        
        # Normalize vectors to prevent overflow
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        # Handle zero vectors
        if a_norm == 0 or b_norm == 0:
            return 0.0
        
        # Calculate cosine similarity safely
        dot_product = np.dot(a, b)
        similarity = dot_product / (a_norm * b_norm)
        
        # Clamp to valid range [-1, 1] to handle numerical errors
        return np.clip(similarity, -1.0, 1.0)
        
    except Exception as e:
        print(f"⚠️  Cosine similarity calculation error: {str(e)}")
        return 0.0


def find_paragraph_semantic_boundaries(embeddings: List[List[float]], paragraphs: List[str]) -> List[int]:
    """Find semantic boundaries by analyzing paragraph embedding similarities."""
    try:
        import numpy as np
        
        if len(embeddings) < 2:
            return [0, len(paragraphs)]
        
        # Convert to numpy array and ensure proper dtype
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Calculate cosine similarities between consecutive paragraphs
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = safe_cosine_similarity(embeddings_array[i], embeddings_array[i + 1])
            similarities.append(sim)
        
        print(f"   📊 Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
        
        # Find boundaries using adaptive threshold
        boundaries = [0]  # Always start with first paragraph
        
        if len(similarities) >= 2:
            # Use percentile-based threshold for more robust boundary detection
            similarity_percentile_25 = np.percentile(similarities, 25)
            threshold = similarity_percentile_25  # Bottom 25% of similarities
            
            print(f"   🎯 Using similarity threshold: {threshold:.3f}")
            
            # Find significant drops in similarity (topic changes)
            for i, sim in enumerate(similarities):
                if sim < threshold:
                    # Ensure minimum distance between boundaries
                    if len(boundaries) == 1 or i - boundaries[-1] >= 1:  # At least 1 paragraph per chunk
                        boundaries.append(i + 1)
                        print(f"   📍 Boundary found at paragraph {i + 1} (similarity: {sim:.3f})")
        
        # Always include the end
        if boundaries[-1] != len(paragraphs):
            boundaries.append(len(paragraphs))
        
        print(f"   🎯 Final boundaries: {boundaries}")
        print(f"   📦 Will create {len(boundaries) - 1} semantic chunks")
        return boundaries
        
    except Exception as e:
        print(f"❌ Failed to find paragraph semantic boundaries: {str(e)}")
        return [0, len(paragraphs)]


def create_paragraph_semantic_chunks(paragraphs: List[str], boundaries: List[int], filename: str) -> List[Dict[str, Any]]:
    """Create chunks based on paragraph semantic boundaries."""
    chunks = []
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        # Combine paragraphs in this semantic segment
        chunk_paragraphs = paragraphs[start_idx:end_idx]
        chunk_content = "\n\n".join(chunk_paragraphs)  # Preserve paragraph structure
        
        chunks.append({
            "content": chunk_content,
            "metadata": {
                "filename": filename,
                "chunk_type": "semantic_paragraph",
                "chunk_index": i,
                "paragraph_start": start_idx,
                "paragraph_end": end_idx,
                "paragraph_count": len(chunk_paragraphs),
                "word_count": len(chunk_content.split()),
                "char_count": len(chunk_content),
                "avg_paragraph_length": sum(len(p.split()) for p in chunk_paragraphs) / len(chunk_paragraphs)
            }
        })
    
    return chunks


def optimize_semantic_chunks(chunks: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
    """Optimize semantic chunks for size and coherence."""
    if not chunks:
        return chunks
    
    print(f"🔧 Optimizing {len(chunks)} semantic chunks...")
    
    processed_chunks = []
    min_words = 100  # Larger minimum for paragraphs
    max_words = 1500  # Reasonable maximum for coherent reading
    
    i = 0
    while i < len(chunks):
        current_chunk = chunks[i]
        word_count = current_chunk["metadata"]["word_count"]
        
        print(f"   📊 Chunk {i}: {word_count} words")
        
        # If chunk is too small, try to merge with next
        if word_count < min_words and i < len(chunks) - 1:
            next_chunk = chunks[i + 1]
            combined_words = word_count + next_chunk["metadata"]["word_count"]
            
            if combined_words <= max_words:
                print(f"   🔗 Merging chunks {i} and {i+1} ({word_count} + {next_chunk['metadata']['word_count']} = {combined_words} words)")
                
                # Merge chunks while preserving paragraph structure
                merged_content = current_chunk["content"] + "\n\n" + next_chunk["content"]
                merged_chunk = {
                    "content": merged_content,
                    "metadata": {
                        "filename": filename,
                        "chunk_type": "semantic_merged",
                        "chunk_index": len(processed_chunks),
                        "paragraph_start": current_chunk["metadata"]["paragraph_start"],
                        "paragraph_end": next_chunk["metadata"]["paragraph_end"],
                        "paragraph_count": current_chunk["metadata"]["paragraph_count"] + next_chunk["metadata"]["paragraph_count"],
                        "word_count": len(merged_content.split()),
                        "char_count": len(merged_content),
                        "merged_from": f"{i},{i + 1}"  # Convert list to string
                    }
                }
                processed_chunks.append(merged_chunk)
                i += 2  # Skip next chunk as it's merged
                continue
        
        # If chunk is too large, split it intelligently
        if word_count > max_words:
            print(f"   ✂️  Splitting large chunk {i} ({word_count} words)")
            large_chunks = split_large_semantic_chunk(current_chunk, max_words, filename)
            for j, sub_chunk in enumerate(large_chunks):
                sub_chunk["metadata"]["chunk_index"] = len(processed_chunks) + j
            processed_chunks.extend(large_chunks)
        else:
            # Keep chunk as is, update index
            current_chunk["metadata"]["chunk_index"] = len(processed_chunks)
            processed_chunks.append(current_chunk)
        
        i += 1
    
    # Update total chunks count
    for chunk in processed_chunks:
        chunk["metadata"]["total_chunks"] = len(processed_chunks)
    
    print(f"   ✅ Optimization complete: {len(processed_chunks)} final chunks")
    return processed_chunks


def split_large_semantic_chunk(chunk: Dict[str, Any], max_words: int, filename: str) -> List[Dict[str, Any]]:
    """Split a large semantic chunk while preserving paragraph structure."""
    content = chunk["content"]
    
    # Split by paragraphs first
    paragraphs = content.split('\n\n')
    
    sub_chunks = []
    current_chunk_paras = []
    current_word_count = 0
    
    for para in paragraphs:
        para_words = len(para.split())
        
        # If adding this paragraph would exceed limit, start new chunk
        if current_word_count + para_words > max_words and current_chunk_paras:
            # Save current chunk
            chunk_content = "\n\n".join(current_chunk_paras)
            sub_chunks.append({
                "content": chunk_content,
                "metadata": {
                    "filename": filename,
                    "chunk_type": "semantic_split",
                    "chunk_index": -1,  # Will be updated
                    "sub_chunk_index": len(sub_chunks),
                    "word_count": len(chunk_content.split()),
                    "char_count": len(chunk_content),
                    "paragraph_count": len(current_chunk_paras),
                    "split_from_large": True
                }
            })
            
            # Start new chunk
            current_chunk_paras = [para]
            current_word_count = para_words
        else:
            current_chunk_paras.append(para)
            current_word_count += para_words
    
    # Add remaining paragraphs as final chunk
    if current_chunk_paras:
        chunk_content = "\n\n".join(current_chunk_paras)
        sub_chunks.append({
            "content": chunk_content,
            "metadata": {
                "filename": filename,
                "chunk_type": "semantic_split",
                "chunk_index": -1,  # Will be updated
                "sub_chunk_index": len(sub_chunks),
                "word_count": len(chunk_content.split()),
                "char_count": len(chunk_content),
                "paragraph_count": len(current_chunk_paras),
                "split_from_large": True
            }
        })
    
    return sub_chunks


def hierarchical_split_large_chunk(chunk: Dict[str, Any], max_words: int) -> List[Dict[str, Any]]:
    """Split a large chunk using hierarchical method."""
    content = chunk["content"]
    filename = chunk["metadata"]["filename"]
    
    # Use hierarchical splitter for large chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_words * 4,  # Approximate character count
        chunk_overlap=100,
        length_function=len,
        separators=[". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    
    split_texts = text_splitter.split_text(content)
    
    sub_chunks = []
    for i, text in enumerate(split_texts):
        sub_chunk = {
            "content": text,
            "metadata": {
                "filename": filename,
                "chunk_type": "semantic_split",
                "chunk_index": -1,  # Will be updated later
                "sub_chunk_index": i,
                "word_count": len(text.split()),
                "char_count": len(text),
                "parent_chunk": True
            }
        }
        sub_chunks.append(sub_chunk)
    
    return sub_chunks


def hierarchical_chunking(text: str, filename: str) -> List[Dict[str, Any]]:
    """
    Perform hierarchical chunking using RecursiveCharacterTextSplitter.
    
    Args:
        text (str): Text to chunk
        filename (str): Original filename
        
    Returns:
        list: List of hierarchical chunks with metadata
    """
    try:
        # Create hierarchical text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\n\n",  # Triple newlines (major sections)
                "\n\n",    # Double newlines (paragraphs)
                "\n",      # Single newlines
                ". ",      # Sentences
                "! ",      # Exclamations
                "? ",      # Questions
                "; ",      # Semicolons
                ", ",      # Commas
                " ",       # Spaces
                ""         # Characters
            ]
        )
        
        # Split text hierarchically
        docs = text_splitter.split_text(text)
        
        # Convert to chunks with metadata
        chunks = []
        for i, doc in enumerate(docs):
            chunks.append({
                "content": doc,
                "metadata": {
                    "filename": filename,
                    "chunk_type": "hierarchical",
                    "chunk_index": i,
                    "total_chunks": len(docs),
                    "word_count": len(doc.split()),
                    "char_count": len(doc)
                }
            })
        
        print(f"🏗️  Hierarchical chunking: {len(chunks)} chunks created")
        return chunks
        
    except Exception as e:
        print(f"❌ Hierarchical chunking failed: {str(e)}")
        return simple_chunking(text, filename)  # Final fallback


def markdown_aware_chunking(text: str, filename: str) -> List[Dict[str, Any]]:
    """
    Perform markdown-aware chunking for structured documents.
    
    Args:
        text (str): Text to chunk (should be markdown)
        filename (str): Original filename
        
    Returns:
        list: List of markdown-aware chunks with metadata
    """
    try:
        # Define markdown headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"), 
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        # Create markdown header splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        
        # Split by headers first
        md_header_splits = markdown_splitter.split_text(text)
        
        # Further split large sections
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        
        # Convert to chunks with metadata
        chunks = []
        chunk_index = 0
        
        for doc in md_header_splits:
            # Split large sections further
            sub_docs = text_splitter.split_text(doc.page_content)
            
            for sub_doc in sub_docs:
                chunks.append({
                    "content": sub_doc,
                    "metadata": {
                        "filename": filename,
                        "chunk_type": "markdown_aware",
                        "chunk_index": chunk_index,
                        "word_count": len(sub_doc.split()),
                        "char_count": len(sub_doc),
                        "headers": doc.metadata
                    }
                })
                chunk_index += 1
        
        # Update total chunks
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        print(f"📝 Markdown-aware chunking: {len(chunks)} chunks created")
        return chunks
        
    except Exception as e:
        print(f"❌ Markdown chunking failed: {str(e)}")
        return hierarchical_chunking(text, filename)  # Fallback


def simple_chunking(text: str, filename: str) -> List[Dict[str, Any]]:
    """
    Simple chunking fallback method.
    
    Args:
        text (str): Text to chunk
        filename (str): Original filename
        
    Returns:
        list: List of simple chunks with metadata
    """
    try:
        words = text.split()
        chunk_size = 1000
        overlap = 200
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "filename": filename,
                        "chunk_type": "simple",
                        "chunk_index": len(chunks),
                        "total_chunks": 0,  # Will be updated
                        "word_count": len(chunk_words),
                        "char_count": len(chunk_text)
                    }
                })
        
        # Update total chunks
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        print(f"📄 Simple chunking: {len(chunks)} chunks created")
        return chunks
        
    except Exception as e:
        print(f"❌ Simple chunking failed: {str(e)}")
        return []


def smart_chunk_text(text: str, filename: str, strategy: str = "semantic") -> List[Dict[str, Any]]:
    """
    Main function to perform smart chunking using LangChain.
    
    Args:
        text (str): Text to chunk
        filename (str): Original filename
        strategy (str): Chunking strategy - "semantic", "hierarchical", "markdown", "simple"
        
    Returns:
        list: List of chunks with metadata
    """
    print(f"🧠 Starting smart chunking for {filename} using {strategy} strategy")
    print(f"📊 Text length: {len(text)} characters, {len(text.split())} words")
    
    if strategy == "semantic":
        return semantic_chunking(text, filename)
    elif strategy == "hierarchical":
        return hierarchical_chunking(text, filename)
    elif strategy == "markdown":
        return markdown_aware_chunking(text, filename)
    elif strategy == "simple":
        return simple_chunking(text, filename)
    else:
        print(f"⚠️  Unknown strategy '{strategy}', using semantic chunking")
        return semantic_chunking(text, filename)


def process_extracted_text(content: str, filename: str, file_type: str, chunking_strategy: str = "semantic") -> List[Dict[str, Any]]:
    """
    Complete pipeline: save extracted text → smart chunking.
    
    Args:
        content (str): Extracted text content
        filename (str): Original filename
        file_type (str): File type (pdf, video, image)
        chunking_strategy (str): Chunking strategy to use
        
    Returns:
        list: List of smart chunks ready for Pinecone
    """
    try:
        # Save extracted text to file
        text_filepath = save_extracted_text(content, filename, file_type)
        
        if not text_filepath:
            print("❌ Failed to save extracted text, using in-memory chunking")
        
        # Perform smart chunking
        chunks = smart_chunk_text(content, filename, chunking_strategy)
        
        # Add file type and extraction info to metadata
        for chunk in chunks:
            chunk["metadata"].update({
                "original_file_type": file_type,
                "text_file_path": text_filepath,
                "chunking_strategy": chunking_strategy
            })
        
        print(f"✅ Smart chunking complete: {len(chunks)} chunks ready for Pinecone")
        return chunks
        
    except Exception as e:
        print(f"❌ Smart chunking pipeline failed: {str(e)}")
        return []
