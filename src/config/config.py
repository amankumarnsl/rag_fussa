import os
from dotenv import load_dotenv

load_dotenv()

# Simple configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
# Single namespace index (optimized)
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-single-namespace")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "./temp_uploads")

# Conversation title generation configuration
CONVERSATION_TITLE_MESSAGE_NUMBER = int(os.getenv("CONVERSATION_TITLE_MESSAGE_NUMBER", "3"))
INCLUDE_CHUNKS_IN_RESPONSE = os.getenv("INCLUDE_CHUNKS_IN_RESPONSE", "false").lower() == "true"
