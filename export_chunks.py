"""
Export all chunks from Pinecone indexes to structured JSON files for visualization
"""
import os
import json
from pinecone import Pinecone
from config import *

def create_chunk_structure():
    """Export all chunks from Pinecone to organized JSON files"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create main chunk_data directory
    chunk_data_dir = "chunk_data"
    os.makedirs(chunk_data_dir, exist_ok=True)
    
    # Create subdirectories
    rag_pdfs_dir = os.path.join(chunk_data_dir, "rag_pdfs")
    rag_videos_dir = os.path.join(chunk_data_dir, "rag_videos")
    rag_images_dir = os.path.join(chunk_data_dir, "rag_images")
    
    os.makedirs(rag_pdfs_dir, exist_ok=True)
    os.makedirs(rag_videos_dir, exist_ok=True)
    os.makedirs(rag_images_dir, exist_ok=True)
    
    # Index configurations
    indexes_config = [
        {
            "name": PINECONE_PDF_INDEX,
            "type": "pdf",
            "directory": rag_pdfs_dir
        },
        {
            "name": PINECONE_VIDEO_INDEX,
            "type": "video", 
            "directory": rag_videos_dir
        },
        {
            "name": PINECONE_IMAGE_INDEX,
            "type": "image",
            "directory": rag_images_dir
        }
    ]
    
    total_files_exported = 0
    total_chunks_exported = 0
    
    print("🚀 Starting chunk export process...")
    print("=" * 50)
    
    for config in indexes_config:
        index_name = config["name"]
        file_type = config["type"]
        output_dir = config["directory"]
        
        try:
            print(f"\n📊 Processing {file_type.upper()} index: {index_name}")
            
            # Get index
            index = pc.Index(index_name)
            
            # Get index stats to find all namespaces
            stats = index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            
            if not namespaces:
                print(f"   ⚠️  No namespaces found in {index_name}")
                continue
            
            print(f"   📁 Found {len(namespaces)} files in index")
            
            # Process each namespace (file)
            for namespace_name, namespace_stats in namespaces.items():
                print(f"   📄 Processing: {namespace_name}")
                
                # Get all vectors from this namespace
                chunks_data = []
                
                try:
                    # Query all vectors in namespace (using dummy vector)
                    query_response = index.query(
                        vector=[0.0] * 1536,  # Dummy vector
                        top_k=10000,  # Get all vectors
                        include_metadata=True,
                        include_values=False,  # Don't include embeddings (too large)
                        namespace=namespace_name
                    )
                    
                    # Process each chunk
                    for i, match in enumerate(query_response.matches):
                        chunk_data = {
                            "chunk_index": i,
                            "vector_id": match.id,
                            "similarity_score": float(match.score) if hasattr(match, 'score') else None,
                            "content": match.metadata.get("content", ""),
                            "metadata": {
                                "s3_url": match.metadata.get("s3_url", ""),
                                "filename": match.metadata.get("filename", namespace_name),
                                "file_type": match.metadata.get("file_type", file_type),
                                "chunk_index": match.metadata.get("chunk_index", i),
                                "total_chunks": match.metadata.get("total_chunks", len(query_response.matches)),
                                "word_count": match.metadata.get("word_count", 0),
                                **{k: v for k, v in match.metadata.items() 
                                   if k not in ["content", "s3_url", "filename", "file_type", "chunk_index", "total_chunks", "word_count"]}
                            }
                        }
                        chunks_data.append(chunk_data)
                    
                    # Sort chunks by chunk_index for proper order
                    chunks_data.sort(key=lambda x: x["metadata"]["chunk_index"])
                    
                    # Create JSON file for this file/namespace
                    filename_without_ext = namespace_name.rsplit('.', 1)[0] if '.' in namespace_name else namespace_name
                    json_filename = f"{filename_without_ext}.json"
                    json_filepath = os.path.join(output_dir, json_filename)
                    
                    # Prepare final structure
                    file_data = {
                        "file_info": {
                            "filename": namespace_name,
                            "file_type": file_type,
                            "total_chunks": len(chunks_data),
                            "namespace": namespace_name,
                            "s3_url": chunks_data[0]["metadata"]["s3_url"] if chunks_data else "",
                            "exported_at": __import__('datetime').datetime.now().isoformat()
                        },
                        "chunks": chunks_data
                    }
                    
                    # Write JSON file
                    with open(json_filepath, 'w', encoding='utf-8') as f:
                        json.dump(file_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"      ✅ Exported {len(chunks_data)} chunks to {json_filename}")
                    total_files_exported += 1
                    total_chunks_exported += len(chunks_data)
                    
                except Exception as e:
                    print(f"      ❌ Error processing {namespace_name}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"   ❌ Error accessing index {index_name}: {str(e)}")
            continue
    
    print("\n" + "=" * 50)
    print("🎉 Export completed!")
    print(f"📁 Total files exported: {total_files_exported}")
    print(f"📄 Total chunks exported: {total_chunks_exported}")
    print(f"📂 Output directory: {os.path.abspath(chunk_data_dir)}")
    print("\n📋 Directory structure:")
    print(f"   {chunk_data_dir}/")
    print(f"   ├── rag_pdfs/")
    print(f"   ├── rag_videos/")
    print(f"   └── rag_images/")


def show_chunk_structure():
    """Display the current chunk_data directory structure"""
    chunk_data_dir = "chunk_data"
    
    if not os.path.exists(chunk_data_dir):
        print("❌ chunk_data directory doesn't exist. Run export first.")
        return
    
    print("📁 Current chunk_data structure:")
    print("=" * 40)
    
    for root, dirs, files in os.walk(chunk_data_dir):
        level = root.replace(chunk_data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        sub_indent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        chunk_count = len(data.get('chunks', []))
                        print(f"{sub_indent}{file} ({chunk_count} chunks)")
                except:
                    print(f"{sub_indent}{file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show_chunk_structure()
    else:
        create_chunk_structure()
