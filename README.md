# RAG FUSSA API

A production-ready RAG (Retrieval-Augmented Generation) API built with FastAPI, supporting PDF, video, and image document processing with intelligent chunking and vector storage.

## 🚀 Features

- **Multi-format Document Processing**: PDF, video, and image support
- **Intelligent Chunking**: Semantic, hierarchical, and markdown-aware text splitting
- **Vector Storage**: Pinecone integration for scalable similarity search
- **Conversational AI**: Context-aware chat with conversation history
- **Production Ready**: Health checks, structured logging, error handling, and retry logic
- **Cloud Integration**: AWS S3 for file storage, OpenAI for embeddings and chat

## 📋 Prerequisites

- Python 3.12+
- OpenAI API key
- Pinecone account and API key
- AWS S3 bucket and credentials
- (Optional) AssemblyAI API key for video transcription

## 🛠️ Installation

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

## ⚙️ Configuration

Create a `.env` file with the following variables:

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here

# Pinecone Configuration
PINECONE_PDF_INDEX=rag-pdfs
PINECONE_VIDEO_INDEX=rag-videos
PINECONE_IMAGE_INDEX=rag-images

# Logging & Environment
LOG_LEVEL=INFO
ENVIRONMENT=development
```

## 🏃‍♂️ Running the Application

### Prerequisites
- Redis server running on port 6379
- All API keys configured in `.env` file

### Development (Easy Start)
```bash
# Start Redis (if not already running)
docker run -d -p 6379:6379 redis:7-alpine

# Start everything with one command
./start_dev.sh
```

### Development (Manual Start)
```bash
# Terminal 1: Start Celery worker
python start_worker.py

# Terminal 2: Start FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production with Docker
```bash
docker-compose up -d
```

### Production (Manual)
```bash
# Start Celery worker
python start_worker.py

# Start FastAPI server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📚 API Endpoints

### Health Checks
- `GET /health` - Comprehensive health check with all dependencies
- `GET /health/live` - Simple liveness check for Kubernetes
- `GET /health/ready` - Readiness check for load balancers

### Document Processing
- `POST /ai-service/internal/process-document-data` - Start async document processing
- `GET /ai-service/internal/task-status/{task_id}` - Check processing status
- `POST /unprocess-document-data` - Remove documents from index

### Query & Chat
- `POST /fetch_rag` - Retrieve relevant content without AI processing
- `POST /ai-service/internal/ask-question` - Conversational AI with RAG

## 🔍 API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│  Celery Worker  │────│  Redis Broker   │
│                 │    │  (Async Tasks)  │    │  & Result Store │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  Document       │              │
         │              │  Processing     │              │
         │              │  Pipeline       │              │
         │              └─────────────────┘              │
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────│  Smart Chunking │──────────────┘
                        │  & Embeddings   │
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │  OpenAI API     │
                        │  (Embeddings &  │
                        │   Chat)         │
                        └─────────────────┘
```

## ⚡ Async Processing

The system now uses **Celery + Redis** for asynchronous document processing:

### Benefits
- **Non-blocking API**: Document processing doesn't block the API
- **Scalable**: Multiple workers can process documents in parallel
- **Progress Tracking**: Real-time status updates and progress monitoring
- **Resilient**: Failed tasks can be retried automatically
- **Resource Efficient**: Better memory and CPU utilization

### Processing Flow
1. **API Request**: Client submits document for processing
2. **Immediate Response**: API returns task ID immediately
3. **Background Processing**: Celery worker processes document asynchronously
4. **Progress Updates**: Client can check status using task ID
5. **Completion**: Final result available when processing completes

### Task Queues
- **document_processing**: Heavy document processing tasks
- **embeddings**: Embedding generation tasks
- **default**: General purpose tasks

## 📊 Monitoring & Logging

### Structured Logging
- **Development**: Colored console output with request tracking
- **Production**: JSON formatted logs for log aggregation
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Health Monitoring
- Dependency health checks (OpenAI, Pinecone, S3, Backend)
- Performance metrics and request tracking
- Circuit breakers for service resilience

### Log Files
- `logs/rag_fussa.log` - Application logs
- `logs/rag_fussa_errors.log` - Error logs only

### Task Monitoring
```bash
# Monitor active tasks
python monitor_tasks.py

# Check specific task status
python monitor_tasks.py <task_id>
```

## 🔧 Production Deployment

### Docker (Recommended)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production
```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
```

### Kubernetes Health Checks
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
```

## 🛡️ Security Features

- **CORS Configuration**: Restrictive origin policies
- **Input Validation**: Pydantic schemas for all endpoints
- **Error Handling**: Secure error responses without sensitive data
- **API Key Management**: Environment-based configuration
- **Request Tracking**: Correlation IDs for debugging

## 📈 Performance Features

- **Retry Logic**: Exponential backoff for transient failures
- **Circuit Breakers**: Prevent cascade failures
- **Async Processing**: Non-blocking I/O operations
- **Request Correlation**: Track requests across services
- **Performance Metrics**: Response time monitoring

## 🧪 Testing

```bash
# Test health endpoints
curl http://localhost:8000/health
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready

# Test document processing
curl -X POST http://localhost:8000/ai-service/internal/process-document-data \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "uuid": "123", "url": "s3://bucket/file.pdf", "type": "PDF", "trainingStatus": "pending"}'
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated and dependencies are installed
2. **API Key Errors**: Verify all required environment variables are set
3. **Pinecone Connection**: Check network connectivity and API key validity
4. **S3 Access**: Verify AWS credentials and bucket permissions

### Logs Location
- Development: Console output
- Production: `logs/` directory

### Health Check Failures
Check individual service health at `/health` endpoint for detailed dependency status.

## 📞 Support

For issues and questions:
- Check the logs first
- Review health check status
- Verify environment configuration
- [Add your support contact information]