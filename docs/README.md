# RAG FUSSA API

A simple RAG system built with FastAPI that processes files from AWS S3 and stores them in Pinecone with custom file names.

## Features

- **Multi-Format Processing**: Supports PDF, video, and image files
- **PDF Processing**: Extracts text from PDF files with page information
- **Video Processing**: Analyzes video content, extracts frames, and transcribes audio
- **Image Processing**: Analyzes visual content and extracts text using OCR
- **Organized Storage**: Separate Pinecone indexes for each file type (PDFs, videos, images)
- **File Isolation**: Each file stored in its own namespace within the appropriate index
- **AWS S3 Integration**: Direct file processing from S3 URLs with automatic filename extraction
- **Smart Organization**: Files tracked by S3 URL for easy management

## API Endpoints

- `POST /train` - Process S3 file and save to appropriate Pinecone index
- `POST /untrain` - Remove file from Pinecone using S3 URL
- `POST /retrain` - Placeholder for future use
- `POST /fetch_rag` - Placeholder for future use

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag_fussa

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the template:

```bash
cp env_template.txt .env
```

Edit `.env` with your actual values:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration (if using Pinecone)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here

# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_id_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key_here
AWS_REGION=us-east-1
AWS_S3_BUCKET=your_s3_bucket_name_here

# Vector Database Type
VECTOR_DB_TYPE=pinecone  # Options: pinecone, chroma, faiss
```

### 3. Run the Application

```bash
# Development mode
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Usage Examples

### Training with different file types

**PDF file:**
```bash
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "s3_url": "https://my-bucket.s3.amazonaws.com/documents/sample.pdf"
     }'
```

**Video file:**
```bash
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "s3_url": "https://my-bucket.s3.amazonaws.com/videos/presentation.mp4"
     }'
```

**Image file:**
```bash
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "s3_url": "https://my-bucket.s3.amazonaws.com/images/diagram.png"
     }'
```

### Removing a file

```bash
curl -X POST "http://localhost:8000/untrain" \
     -H "Content-Type: application/json" \
     -d '{
       "s3_url": "https://my-bucket.s3.amazonaws.com/documents/sample.pdf"
     }'
```

## Architecture

```
rag_fussa/
├── main.py           # FastAPI application with all API logic
├── config.py         # Environment configuration
├── schemas.py        # Pydantic request/response models
├── chunking.py       # Text chunking strategies
├── pdf_processor.py  # PDF file processing
├── video_processor.py # Video file processing
├── image_processor.py # Image file processing
├── requirements.txt  # Python dependencies
├── env_template.txt  # Environment variables template
└── README.md         # Documentation
```

## Supported File Types

### PDF Files
- **Extension**: `.pdf`
- **Processing**: Text extraction with page information
- **Features**: Full text search, metadata extraction

### Video Files
- **Extensions**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`, `.m4v`
- **Processing**: Frame analysis and audio transcription
- **Features**: Visual content analysis, speech-to-text (placeholder)

### Image Files
- **Extensions**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.webp`, `.svg`
- **Processing**: Visual analysis and OCR text extraction
- **Features**: Content description, text recognition (placeholder)

## Configuration Options

### Vector Database Options

**Pinecone** (Recommended for production):
```env
VECTOR_DB_TYPE=pinecone
PINECONE_API_KEY=your_key
PINECONE_ENVIRONMENT=your_env
PINECONE_INDEX_NAME=your_index
```

**ChromaDB** (Good for development):
```env
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### File Processing Settings

```env
CHUNK_SIZE=1000              # Text chunk size for embeddings
CHUNK_OVERLAP=200           # Overlap between chunks
MAX_FILE_SIZE=100MB         # Maximum file size
TEMP_UPLOAD_DIR=./temp_uploads
```

## Supported File Types

- **Documents**: PDF
- **Videos**: MP4, AVI, MOV
- **Images**: JPG, JPEG, PNG, GIF, BMP, TIFF

## Error Handling

The API provides comprehensive error handling with appropriate HTTP status codes:

- `400` - Bad Request (invalid input)
- `404` - Not Found (file not found)
- `500` - Internal Server Error (processing failures)

## Logging

Structured logging is configured with different levels:
- `DEBUG` - Detailed debugging information
- `INFO` - General information (default)
- `WARNING` - Warning messages
- `ERROR` - Error messages

Configure log level in `.env`:
```env
LOG_LEVEL=INFO
```

## Production Deployment

### Docker Deployment (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production

```env
DEBUG=False
LOG_LEVEL=INFO
```

## Development

### Adding New File Types

1. Add the file type to `FileType` enum in `app/models/schemas.py`
2. Implement processing logic in `app/services/file_processor.py`
3. Update the supported file types configuration

### Adding New Vector Stores

1. Implement the `VectorStoreInterface` in `app/services/vector_store.py`
2. Add the new store to the `get_vector_store()` factory function
3. Update configuration options

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Check your `.env` file configuration
2. **S3 Access Denied**: Verify AWS credentials and bucket permissions
3. **Vector Store Connection**: Ensure Pinecone/ChromaDB is properly configured
4. **File Processing Errors**: Check file format and size limits

### Health Check

Use the health endpoint to verify system status:
```bash
curl http://localhost:8000/api/v1/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the API documentation at `/docs`
- Review the logs for error details
- Ensure all required environment variables are set
