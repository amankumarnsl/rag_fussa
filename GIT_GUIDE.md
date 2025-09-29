# Git Repository Guide for RAG FUSSA

## ✅ Files to Commit (Production Ready)

### Core Application
- `src/` - All source code including new utilities
- `requirements.txt` - Python dependencies
- `README.md` - Comprehensive documentation
- `src/config/env_template.txt` - Environment template (NO SECRETS)

### Infrastructure & Deployment
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Local development setup
- `.gitignore` - Comprehensive ignore rules
- `.dockerignore` - Docker build optimization

### Documentation
- `docs/` - API documentation (if any)
- `*.md` - Markdown documentation files

## ❌ Files NEVER to Commit

### Sensitive Data
- `.env` - Environment variables with API keys
- `config/secrets.py` - Secret configurations
- `*.key`, `*.pem` - Private keys and certificates

### Runtime Files
- `venv/` - Python virtual environment
- `__pycache__/` - Python bytecode cache
- `*.pyc` - Compiled Python files
- `logs/` - Log files
- `*.log` - Any log files

### Data & Processing
- `data_extraction_visualize/` - Large data files
- `data_to_upload_to_S3/` - Upload data
- `temp_uploads/` - Temporary files
- `processed_data/` - Processed content
- `models/` - ML model files
- `embeddings_cache/` - Cached embeddings

### IDE & OS Files
- `.vscode/`, `.idea/` - IDE configurations
- `.DS_Store` - macOS system files
- `Thumbs.db` - Windows thumbnails
- `*.swp`, `*.swo` - Vim temporary files

### Development
- `tests/` - Test files (if separate)
- `.pytest_cache/` - Test cache
- `coverage/` - Test coverage reports
- `*.prof` - Profiling files

## 🔒 Security Checklist

Before committing, ensure:

1. **No API Keys**: Check for OpenAI, Pinecone, AWS keys
2. **No Secrets**: Verify no passwords or tokens
3. **No Personal Data**: Remove any user data or PII
4. **No Large Files**: Keep repository under 100MB
5. **Environment Template**: Only commit template, not actual .env

## 📊 Repository Size Optimization

### What Makes It Efficient:
- ✅ Source code only (~2-5MB)
- ✅ Documentation and configs (~1MB)
- ✅ No binary files or data
- ✅ Proper .gitignore rules

### What to Avoid:
- ❌ Virtual environments (venv/) - Can be 500MB+
- ❌ Data files - Can be GBs
- ❌ Log files - Accumulate over time
- ❌ Model files - Often 100MB+
- ❌ Cache directories - Temporary files

## 🚀 Deployment Strategy

### Development
```bash
git clone <repo>
cp src/config/env_template.txt .env
# Edit .env with your keys
pip install -r requirements.txt
uvicorn src.main:app --reload
```

### Production
```bash
# Docker
docker build -t rag-fussa-api .
docker run -p 8000:8000 --env-file .env rag-fussa-api

# Or docker-compose
docker-compose up -d
```

## 📋 Commit Message Template

```
feat: Add production-ready logging and health checks

- Implement structured logging with loguru
- Add comprehensive health check endpoints
- Add error handling with retry logic and circuit breakers
- Add Docker configuration for deployment
- Update documentation and .gitignore

Production ready features:
✅ Health monitoring
✅ Structured logging  
✅ Error handling
✅ Security improvements
✅ Docker support
```

## 🔍 Pre-Commit Checklist

Before every commit:

- [ ] No `.env` file committed
- [ ] No API keys in code
- [ ] No large files (>10MB)
- [ ] No binary files (except Docker images)
- [ ] No personal/sensitive data
- [ ] `.gitignore` is comprehensive
- [ ] Documentation is updated
- [ ] Code is tested locally

## 📈 Benefits of This Setup

1. **Fast Cloning**: Repository stays under 10MB
2. **Secure**: No secrets or sensitive data
3. **Portable**: Works on any machine with Docker
4. **Maintainable**: Clear separation of code and data
5. **Scalable**: Easy to deploy to cloud platforms
6. **Collaborative**: Safe for team development
