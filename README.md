# RAG FUSSA API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, designed to handle document processing, conversation management, and intelligent query responses.

## 🚀 Features

### Core Functionality
- **Document Processing**: PDF, Video, and Image content extraction and indexing
- **RAG System**: Intelligent question answering with retrieved context
- **Conversation Management**: Context-aware multi-turn conversations
- **Health Monitoring**: Comprehensive health checks for all dependencies
- **Async Processing**: Non-blocking document processing pipeline

### Smart Query Processing
- **Query Classification**: Automatically distinguishes between general conversation and knowledge questions
- **Context Retrieval**: Semantic search across multiple document types
- **Conversation Continuity**: Maintains context across conversation turns
- **Response Generation**: OpenAI-powered intelligent responses

## 🛠️ Technology Stack

- **FastAPI**: Modern, fast web framework
- **OpenAI**: GPT-4o-mini for text generation and embeddings
- **Pinecone**: Vector database for semantic search
- **AWS S3**: Document storage and retrieval
- **Pydantic**: Data validation and serialization
- **Tenacity**: Retry logic and error handling

## 📁 Project Structure

```
rag_fussa/
├── src/
│   ├── config/
│   │   ├── config.py          # Configuration management
│   │   └── schemas.py         # Pydantic models
│   ├── processors/
│   │   ├── pdf_processor.py   # PDF text extraction
│   │   ├── video_processor.py # Video transcription
│   │   └── image_processor.py # Image OCR processing
│   ├── utils/
│   │   ├── health_checks.py   # Health monitoring
│   │   ├── error_handling.py  # Error management
│   │   ├── chunking.py        # Text chunking utilities
│   │   ├── smart_chunking.py  # Intelligent text segmentation
│   │   ├── text_pipeline.py   # Text processing pipeline
│   │   ├── setup_pinecone.py  # Pinecone initialization
│   │   ├── export_chunks.py   # Data export utilities
│   │   └── upload_to_s3.py    # S3 upload functionality
│   └── main.py               # Main FastAPI application
├── docs/
│   ├── README.md             # Detailed documentation
│   └── RAG_FUSSA_API.postman_collection.json
├── logs/                     # Debug logs (with .gitkeep)
├── data_extraction_visualize/ # Processing data
├── data_to_upload_to_S3/     # Source documents
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables
```

## 🔧 Installation

### Prerequisites
- Python 3.12+
- OpenAI API key
- Pinecone API key
- AWS credentials (for S3)
- Backend API endpoint

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd rag_fussa
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp src/config/env_template.txt .env
# Edit .env with your API keys and configuration
```

5. **Set up Pinecone**
```bash
python src/utils/setup_pinecone.py
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET=your_s3_bucket_name
AWS_REGION=your_aws_region

# Backend Configuration
BACKEND_BASE_URL=your_backend_host
BACKEND_PORT=your_backend_port
BACKEND_ENDPOINT_PATH=/your/endpoint/path

# Debug Configuration
DEBUG_PRINT=false  # Set to true for console debug output

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Debug Mode

Set `DEBUG_PRINT=true` in your `.env` file to enable:
- Console debug output with timestamps
- Automatic debug file logging with datetime filenames
- Detailed request/response tracking

Debug files are saved as: `logs/YYYY-MM-DD_HH-MM-SS_debug.log`

## 🚀 Usage

### Start the Server

```bash
# Development mode with auto-reload
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Health Checks
- `GET /health` - Overall system health
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe

#### Document Processing
- `POST /ai-service/internal/process-document-data` - Process documents
- `GET /ai-service/internal/task-status/{task_id}` - Check processing status

#### Query Interface
- `POST /ai-service/internal/ask-question` - Ask questions with RAG

### Example API Calls

#### Ask a Question (New Conversation)
```json
POST /ai-service/internal/ask-question
{
  "question": "What is artificial intelligence?",
  "conversationId": null,
  "type": "TEXT",
  "conversationHistory": []
}
```

#### Continue a Conversation
```json
POST /ai-service/internal/ask-question
{
  "question": "Tell me more about that",
  "conversationId": "resp_abc123...",
  "type": "TEXT",
  "conversationHistory": [
    {
      "timestamp": "2025-01-01T10:00:00Z",
      "question": "What is AI?",
      "answer": "AI is the simulation of human intelligence..."
    }
  ]
}
```

#### Process a Document
```json
POST /ai-service/internal/process-document-data
{
  "uuid": "unique-document-id",
  "url": "s3://bucket/document.pdf",
  "type": "PDF",
  "trainingStatus": "PENDING"
}
```

## 🔍 Query Processing Flow

1. **Request Reception**: API receives query with conversation context
2. **Query Rephrasing**: Context-aware query enhancement for better retrieval
3. **Content Retrieval**: Semantic search across Pinecone vector database
4. **Query Classification**: Determines if it's general conversation or knowledge question
5. **Response Generation**: OpenAI-powered answer with retrieved context
6. **Context Update**: Updates conversation history for future turns

## 📊 Health Monitoring

The API provides comprehensive health checks:

- **OpenAI**: API connectivity and response times
- **Pinecone**: Vector database status and performance
- **AWS S3**: Storage service availability
- **Backend API**: External service connectivity
- **File System**: Local storage and permissions

## 🛡️ Error Handling

- **Retry Logic**: Automatic retries with exponential backoff
- **Circuit Breaker**: Prevents cascade failures
- **Graceful Degradation**: Fallback responses when services fail
- **Comprehensive Logging**: Detailed error tracking and debugging

## 🧪 Testing

Use the provided Postman collection:
```bash
docs/RAG_FUSSA_API.postman_collection.json
```

Import this collection into Postman for easy API testing.

## 📝 Debugging

### Debug Mode Features
- **Console Output**: Real-time debug information
- **File Logging**: Timestamped debug files in `logs/` directory
- **Request Tracking**: Full request/response lifecycle logging
- **Performance Metrics**: Response times and processing durations

### Debug Output Example
```
🚀 Starting RAG FUSSA API
📝 Debug file: logs/2025-10-01_11-30-45_debug.log
Processing AI query
  conversation_id: null
  question: What is AI?
Using rephrased query
  original_query: What is AI?
  rephrased_query: What is artificial intelligence?
Content retrieved
  total_results: 5
Query classified
  query_type: KNOWLEDGE_QUESTION
```

## 🔄 Conversation Management

The API supports both new and continuing conversations:

- **New Conversation**: `conversationId: null` or `conversationId: ""`
- **Continuing Conversation**: `conversationId: "resp_abc123..."`
- **Context Preservation**: Full conversation history maintained
- **Thread Continuity**: OpenAI thread IDs for seamless context

## 📈 Performance

- **Async Processing**: Non-blocking document processing
- **Vector Search**: Optimized semantic retrieval
- **Caching**: Intelligent response caching
- **Rate Limiting**: Built-in protection against abuse

## 🚀 Production Deployment

1. **Environment Setup**: Configure all required environment variables
2. **Security**: Ensure proper API key management
3. **Monitoring**: Enable health checks and logging
4. **Scaling**: Configure load balancing and horizontal scaling
5. **Backup**: Set up data backup and recovery procedures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
- Check the health endpoints for system status
- Review debug logs for detailed error information
- Consult the Postman collection for API examples
- Contact the development team for assistance

---

**RAG FUSSA API** - Intelligent Document Processing and Conversation Management
