"""
Health check utilities for RAG FUSSA API
"""
import time
import asyncio
from typing import Dict, Any, Optional
from enum import Enum
from .logging_config import get_logger

logger = get_logger("health_checks")

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class HealthCheckResult:
    def __init__(self, service: str, status: HealthStatus, message: str = "", details: Dict[str, Any] = None):
        self.service = service
        self.status = status
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()

class DependencyChecker:
    """Check health of external dependencies"""
    
    def __init__(self):
        self.timeout = 5.0  # 5 second timeout for health checks
    
    async def check_openai(self) -> HealthCheckResult:
        """Check OpenAI API connectivity"""
        try:
            import openai
            from ..config.config import OPENAI_API_KEY
            
            if not OPENAI_API_KEY:
                return HealthCheckResult(
                    "openai",
                    HealthStatus.UNHEALTHY,
                    "OpenAI API key not configured"
                )
            
            # Test with a small embedding request
            start_time = time.time()
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    openai.embeddings.create,
                    input=["health check"],
                    model="text-embedding-3-small"
                ),
                timeout=self.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response and len(response.data) > 0:
                return HealthCheckResult(
                    "openai",
                    HealthStatus.HEALTHY,
                    "OpenAI API is accessible",
                    {
                        "response_time_ms": round(duration_ms, 2),
                        "model": "text-embedding-3-small"
                    }
                )
            else:
                return HealthCheckResult(
                    "openai",
                    HealthStatus.UNHEALTHY,
                    "OpenAI API returned empty response"
                )
                
        except asyncio.TimeoutError:
            return HealthCheckResult(
                "openai",
                HealthStatus.UNHEALTHY,
                "OpenAI API timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                "openai",
                HealthStatus.UNHEALTHY,
                f"OpenAI API error: {str(e)}"
            )
    
    async def check_pinecone(self) -> HealthCheckResult:
        """Check Pinecone connectivity"""
        try:
            from pinecone import Pinecone
            from ..config.config import PINECONE_API_KEY, PINECONE_PDF_INDEX, PINECONE_VIDEO_INDEX, PINECONE_IMAGE_INDEX
            
            if not PINECONE_API_KEY:
                return HealthCheckResult(
                    "pinecone",
                    HealthStatus.UNHEALTHY,
                    "Pinecone API key not configured"
                )
            
            start_time = time.time()
            
            # Initialize Pinecone client
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if indexes exist and are accessible
            indexes_to_check = [
                ("pdf", PINECONE_PDF_INDEX),
                ("video", PINECONE_VIDEO_INDEX),
                ("image", PINECONE_IMAGE_INDEX)
            ]
            
            index_status = {}
            all_healthy = True
            
            for index_type, index_name in indexes_to_check:
                try:
                    index = pc.Index(index_name)
                    stats = await asyncio.wait_for(
                        asyncio.to_thread(index.describe_index_stats),
                        timeout=self.timeout
                    )
                    
                    index_status[index_type] = {
                        "status": "accessible",
                        "vector_count": stats.get("total_vector_count", 0),
                        "dimension": stats.get("dimension", 0)
                    }
                    
                except Exception as e:
                    index_status[index_type] = {
                        "status": "error",
                        "error": str(e)
                    }
                    all_healthy = False
            
            duration_ms = (time.time() - start_time) * 1000
            
            if all_healthy:
                return HealthCheckResult(
                    "pinecone",
                    HealthStatus.HEALTHY,
                    "All Pinecone indexes are accessible",
                    {
                        "response_time_ms": round(duration_ms, 2),
                        "indexes": index_status
                    }
                )
            else:
                return HealthCheckResult(
                    "pinecone",
                    HealthStatus.DEGRADED,
                    "Some Pinecone indexes are not accessible",
                    {
                        "response_time_ms": round(duration_ms, 2),
                        "indexes": index_status
                    }
                )
                
        except asyncio.TimeoutError:
            return HealthCheckResult(
                "pinecone",
                HealthStatus.UNHEALTHY,
                "Pinecone API timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                "pinecone",
                HealthStatus.UNHEALTHY,
                f"Pinecone API error: {str(e)}"
            )
    
    async def check_aws_s3(self) -> HealthCheckResult:
        """Check AWS S3 connectivity"""
        try:
            import boto3
            from ..config.config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
            
            if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION]):
                return HealthCheckResult(
                    "aws_s3",
                    HealthStatus.UNHEALTHY,
                    "AWS credentials not configured"
                )
            
            start_time = time.time()
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
            
            # Test S3 connectivity by listing buckets
            await asyncio.wait_for(
                asyncio.to_thread(s3_client.list_buckets),
                timeout=self.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                "aws_s3",
                HealthStatus.HEALTHY,
                "AWS S3 is accessible",
                {
                    "response_time_ms": round(duration_ms, 2),
                    "region": AWS_REGION
                }
            )
            
        except asyncio.TimeoutError:
            return HealthCheckResult(
                "aws_s3",
                HealthStatus.UNHEALTHY,
                "AWS S3 timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                "aws_s3",
                HealthStatus.UNHEALTHY,
                f"AWS S3 error: {str(e)}"
            )
    
    async def check_backend_api(self) -> HealthCheckResult:
        """Check backend API connectivity"""
        try:
            import requests
            import os
            
            backend_base_url = os.getenv("BACKEND_BASE_URL", "192.168.68.72")
            backend_port = os.getenv("BACKEND_PORT", "4504")
            
            # Simple health check endpoint (you may need to adjust this)
            backend_url = f"http://{backend_base_url}:{backend_port}/health"
            
            start_time = time.time()
            
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    requests.get,
                    backend_url,
                    timeout=self.timeout
                ),
                timeout=self.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return HealthCheckResult(
                    "backend_api",
                    HealthStatus.HEALTHY,
                    "Backend API is accessible",
                    {
                        "response_time_ms": round(duration_ms, 2),
                        "status_code": response.status_code
                    }
                )
            else:
                return HealthCheckResult(
                    "backend_api",
                    HealthStatus.DEGRADED,
                    f"Backend API returned status {response.status_code}",
                    {
                        "response_time_ms": round(duration_ms, 2),
                        "status_code": response.status_code
                    }
                )
                
        except asyncio.TimeoutError:
            return HealthCheckResult(
                "backend_api",
                HealthStatus.UNHEALTHY,
                "Backend API timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                "backend_api",
                HealthStatus.UNHEALTHY,
                f"Backend API error: {str(e)}"
            )
    
    async def check_file_system(self) -> HealthCheckResult:
        """Check file system accessibility"""
        try:
            import os
            import tempfile
            
            start_time = time.time()
            
            # Test write access to temp directory
            temp_dir = os.getenv("TEMP_UPLOAD_DIR", "./temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Test write and read
            test_file = os.path.join(temp_dir, "health_check_test.txt")
            test_content = "health check"
            
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            with open(test_file, 'r') as f:
                read_content = f.read()
            
            # Clean up
            os.remove(test_file)
            
            duration_ms = (time.time() - start_time) * 1000
            
            if read_content == test_content:
                return HealthCheckResult(
                    "file_system",
                    HealthStatus.HEALTHY,
                    "File system is accessible",
                    {
                        "response_time_ms": round(duration_ms, 2),
                        "temp_dir": temp_dir
                    }
                )
            else:
                return HealthCheckResult(
                    "file_system",
                    HealthStatus.UNHEALTHY,
                    "File system read/write test failed"
                )
                
        except Exception as e:
            return HealthCheckResult(
                "file_system",
                HealthStatus.UNHEALTHY,
                f"File system error: {str(e)}"
            )

async def check_all_dependencies() -> Dict[str, HealthCheckResult]:
    """Check all external dependencies"""
    checker = DependencyChecker()
    
    # Run all checks concurrently
    results = await asyncio.gather(
        checker.check_openai(),
        checker.check_pinecone(),
        checker.check_aws_s3(),
        checker.check_backend_api(),
        checker.check_file_system(),
        return_exceptions=True
    )
    
    # Handle exceptions
    dependency_results = {}
    check_names = ["openai", "pinecone", "aws_s3", "backend_api", "file_system"]
    
    for i, result in enumerate(results):
        service_name = check_names[i]
        
        if isinstance(result, Exception):
            dependency_results[service_name] = HealthCheckResult(
                service_name,
                HealthStatus.UNHEALTHY,
                f"Health check failed: {str(result)}"
            )
        else:
            dependency_results[service_name] = result
    
    return dependency_results

def get_overall_health_status(dependency_results: Dict[str, HealthCheckResult]) -> HealthStatus:
    """Determine overall health status based on dependency results"""
    statuses = [result.status for result in dependency_results.values()]
    
    if all(status == HealthStatus.HEALTHY for status in statuses):
        return HealthStatus.HEALTHY
    elif any(status == HealthStatus.UNHEALTHY for status in statuses):
        return HealthStatus.UNHEALTHY
    elif any(status == HealthStatus.DEGRADED for status in statuses):
        return HealthStatus.DEGRADED
    else:
        return HealthStatus.UNKNOWN
