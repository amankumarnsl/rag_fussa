"""
Setup script to create Pinecone indexes for the RAG system
"""
from pinecone import Pinecone, ServerlessSpec

# Handle both relative and absolute imports
try:
    from ..config.config import *
except ImportError:
    # If running as standalone script, add parent directory to path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.config.config import *

def setup_pinecone_indexes():
    """Create the required Pinecone indexes if they don't exist"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Define the indexes we need
    indexes_to_create = [
        {"name": PINECONE_PDF_INDEX, "description": "PDF documents"},
        {"name": PINECONE_VIDEO_INDEX, "description": "Video files"},
        {"name": PINECONE_IMAGE_INDEX, "description": "Image files"}
    ]
    
    # Get existing indexes
    existing_indexes = [index.name for index in pc.list_indexes()]
    print(f"📋 Existing indexes: {existing_indexes}")
    
    for index_config in indexes_to_create:
        index_name = index_config["name"]
        description = index_config["description"]
        
        if index_name not in existing_indexes:
            print(f"🔨 Creating index: {index_name} ({description})")
            try:
                pc.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI ada-002 embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"✅ Successfully created index: {index_name}")
            except Exception as e:
                print(f"❌ Failed to create index {index_name}: {str(e)}")
        else:
            print(f"✅ Index already exists: {index_name}")
    
    print(f"\n🎉 Pinecone setup complete!")
    print(f"📊 Final indexes: {[index.name for index in pc.list_indexes()]}")


if __name__ == "__main__":
    setup_pinecone_indexes()
