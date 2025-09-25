import os
from dotenv import load_dotenv

load_dotenv()

# Simple configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_PDF_INDEX = os.getenv("PINECONE_PDF_INDEX", "rag-pdfs")
PINECONE_VIDEO_INDEX = os.getenv("PINECONE_VIDEO_INDEX", "rag-videos")  
PINECONE_IMAGE_INDEX = os.getenv("PINECONE_IMAGE_INDEX", "rag-images")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "./temp_uploads")
