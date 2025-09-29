"""
Enhanced error handling with retry logic and graceful degradation
"""
import time
import asyncio
from typing import Dict, Any, Optional, Callable, Type, Union, List
from enum import Enum
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log,
    after_log
)
from .logging_config import get_logger

logger = get_logger("error_handling")

class ErrorType(str, Enum):
    TRANSIENT = "transient"  # Temporary issues that might resolve
    PERMANENT = "permanent"  # Issues that won't resolve with retries
    RATE_LIMIT = "rate_limit"  # Rate limiting issues
    AUTHENTICATION = "authentication"  # Auth issues
    VALIDATION = "validation"  # Input validation issues

class ServiceError(Exception):
    """Base exception for service-specific errors"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.PERMANENT, 
                 service: str = "unknown", details: Dict[str, Any] = None):
        self.message = message
        self.error_type = error_type
        self.service = service
        self.details = details or {}
        super().__init__(message)

class OpenAIServiceError(ServiceError):
    """OpenAI API specific errors"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.PERMANENT, 
                 details: Dict[str, Any] = None):
        super().__init__(message, error_type, "openai", details)

class PineconeServiceError(ServiceError):
    """Pinecone API specific errors"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.PERMANENT, 
                 details: Dict[str, Any] = None):
        super().__init__(message, error_type, "pinecone", details)

class S3ServiceError(ServiceError):
    """AWS S3 specific errors"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.PERMANENT, 
                 details: Dict[str, Any] = None):
        super().__init__(message, error_type, "s3", details)

class BackendServiceError(ServiceError):
    """Backend API specific errors"""
    def __init__(self, message: str, error_type: ErrorType = ErrorType.PERMANENT, 
                 details: Dict[str, Any] = None):
        super().__init__(message, error_type, "backend", details)

class RetryConfig:
    """Configuration for retry logic"""
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_multiplier: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_multiplier = exponential_multiplier

# Default retry configurations for different services
RETRY_CONFIGS = {
    "openai": RetryConfig(max_attempts=3, base_delay=1.0, max_delay=30.0),
    "pinecone": RetryConfig(max_attempts=3, base_delay=1.0, max_delay=20.0),
    "s3": RetryConfig(max_attempts=3, base_delay=1.0, max_delay=15.0),
    "backend": RetryConfig(max_attempts=2, base_delay=1.0, max_delay=10.0),
    "default": RetryConfig(max_attempts=2, base_delay=1.0, max_delay=10.0)
}

def classify_openai_error(error: Exception) -> ErrorType:
    """Classify OpenAI errors for appropriate retry strategy"""
    error_str = str(error).lower()
    
    # Rate limiting
    if any(keyword in error_str for keyword in ["rate limit", "quota", "throttle"]):
        return ErrorType.RATE_LIMIT
    
    # Authentication issues
    if any(keyword in error_str for keyword in ["unauthorized", "forbidden", "api key", "authentication"]):
        return ErrorType.AUTHENTICATION
    
    # Validation issues
    if any(keyword in error_str for keyword in ["invalid", "malformed", "validation"]):
        return ErrorType.VALIDATION
    
    # Transient issues (network, server errors)
    if any(keyword in error_str for keyword in ["timeout", "connection", "server error", "service unavailable"]):
        return ErrorType.TRANSIENT
    
    # Default to permanent for unknown errors
    return ErrorType.PERMANENT

def classify_pinecone_error(error: Exception) -> ErrorType:
    """Classify Pinecone errors for appropriate retry strategy"""
    error_str = str(error).lower()
    
    # Rate limiting
    if any(keyword in error_str for keyword in ["rate limit", "quota", "throttle"]):
        return ErrorType.RATE_LIMIT
    
    # Authentication issues
    if any(keyword in error_str for keyword in ["unauthorized", "forbidden", "api key"]):
        return ErrorType.AUTHENTICATION
    
    # Transient issues
    if any(keyword in error_str for keyword in ["timeout", "connection", "server error"]):
        return ErrorType.TRANSIENT
    
    return ErrorType.PERMANENT

def classify_s3_error(error: Exception) -> ErrorType:
    """Classify S3 errors for appropriate retry strategy"""
    error_str = str(error).lower()
    
    # Rate limiting
    if any(keyword in error_str for keyword in ["slow down", "throttle"]):
        return ErrorType.RATE_LIMIT
    
    # Authentication issues
    if any(keyword in error_str for keyword in ["unauthorized", "forbidden", "access denied"]):
        return ErrorType.AUTHENTICATION
    
    # Transient issues
    if any(keyword in error_str for keyword in ["timeout", "connection", "service unavailable"]):
        return ErrorType.TRANSIENT
    
    return ErrorType.PERMANENT

def create_retry_decorator(service: str, error_classifier: Callable = None):
    """Create a retry decorator for a specific service"""
    config = RETRY_CONFIGS.get(service, RETRY_CONFIGS["default"])
    
    # Define retry conditions
    def should_retry(exception):
        if error_classifier:
            error_type = error_classifier(exception)
            # Only retry transient and rate limit errors
            return error_type in [ErrorType.TRANSIENT, ErrorType.RATE_LIMIT]
        
        # Default: retry on common transient exceptions
        return isinstance(exception, (
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError
        ))
    
    return retry(
        stop=stop_after_attempt(config.max_attempts),
        wait=wait_exponential(
            multiplier=config.exponential_multiplier,
            min=config.base_delay,
            max=config.max_delay
        ),
        retry=retry_if_exception_type(Exception) if not error_classifier else should_retry,
        before_sleep=before_sleep_log(logger, "WARNING"),
        after=after_log(logger, "INFO"),
        reraise=True
    )

# Service-specific retry decorators
openai_retry = create_retry_decorator("openai", classify_openai_error)
pinecone_retry = create_retry_decorator("pinecone", classify_pinecone_error)
s3_retry = create_retry_decorator("s3", classify_s3_error)
backend_retry = create_retry_decorator("backend")

class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                logger.info(f"Circuit breaker transitioning to half_open")
            else:
                raise ServiceError(
                    f"Circuit breaker is open for {self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s",
                    ErrorType.TRANSIENT
                )
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                logger.info(f"Circuit breaker closed after successful call")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e

# Global circuit breakers for each service
circuit_breakers = {
    "openai": CircuitBreaker(failure_threshold=3, recovery_timeout=60),
    "pinecone": CircuitBreaker(failure_threshold=5, recovery_timeout=30),
    "s3": CircuitBreaker(failure_threshold=3, recovery_timeout=30),
    "backend": CircuitBreaker(failure_threshold=3, recovery_timeout=30)
}

def safe_api_call(service: str, func: Callable, *args, **kwargs):
    """Execute API call with retry logic and circuit breaker"""
    try:
        # Apply circuit breaker
        circuit_breaker = circuit_breakers.get(service)
        if circuit_breaker:
            result = circuit_breaker.call(func, *args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        return result
    except Exception as e:
        logger.error(f"API call failed for {service}", error=str(e))
        raise

async def safe_async_api_call(service: str, func: Callable, *args, **kwargs):
    """Execute async API call with retry logic and circuit breaker"""
    try:
        # Apply circuit breaker (for async functions)
        circuit_breaker = circuit_breakers.get(service)
        if circuit_breaker:
            result = circuit_breaker.call(lambda: asyncio.create_task(func(*args, **kwargs)))
            if asyncio.iscoroutine(result):
                result = await result
        else:
            result = await func(*args, **kwargs)
        
        return result
    except Exception as e:
        logger.error(f"Async API call failed for {service}", error=str(e))
        raise

def handle_openai_error(error: Exception) -> OpenAIServiceError:
    """Convert OpenAI errors to our custom error types"""
    error_type = classify_openai_error(error)
    return OpenAIServiceError(
        f"OpenAI API error: {str(error)}",
        error_type=error_type,
        details={"original_error": str(error)}
    )

def handle_pinecone_error(error: Exception) -> PineconeServiceError:
    """Convert Pinecone errors to our custom error types"""
    error_type = classify_pinecone_error(error)
    return PineconeServiceError(
        f"Pinecone API error: {str(error)}",
        error_type=error_type,
        details={"original_error": str(error)}
    )

def handle_s3_error(error: Exception) -> S3ServiceError:
    """Convert S3 errors to our custom error types"""
    error_type = classify_s3_error(error)
    return S3ServiceError(
        f"S3 API error: {str(error)}",
        error_type=error_type,
        details={"original_error": str(error)}
    )

def handle_backend_error(error: Exception) -> BackendServiceError:
    """Convert backend API errors to our custom error types"""
    error_str = str(error).lower()
    
    if "timeout" in error_str or "connection" in error_str:
        error_type = ErrorType.TRANSIENT
    elif "unauthorized" in error_str or "forbidden" in error_str:
        error_type = ErrorType.AUTHENTICATION
    else:
        error_type = ErrorType.PERMANENT
    
    return BackendServiceError(
        f"Backend API error: {str(error)}",
        error_type=error_type,
        details={"original_error": str(error)}
    )

def create_fallback_response(service: str, operation: str, error: Exception) -> Dict[str, Any]:
    """Create fallback responses for different services and operations"""
    
    fallbacks = {
        "openai": {
            "embeddings": {
                "error": "Embedding service temporarily unavailable",
                "fallback": "Using cached embeddings or skipping embedding generation"
            },
            "chat": {
                "error": "AI service temporarily unavailable", 
                "fallback": "Returning retrieved content without AI processing"
            }
        },
        "pinecone": {
            "search": {
                "error": "Search service temporarily unavailable",
                "fallback": "Returning empty search results"
            },
            "upsert": {
                "error": "Vector storage temporarily unavailable",
                "fallback": "Queuing vectors for later storage"
            }
        },
        "s3": {
            "download": {
                "error": "File storage temporarily unavailable",
                "fallback": "Using cached files or skipping file processing"
            }
        },
        "backend": {
            "status_update": {
                "error": "Status update service temporarily unavailable",
                "fallback": "Continuing processing without status updates"
            }
        }
    }
    
    service_fallbacks = fallbacks.get(service, {})
    operation_fallback = service_fallbacks.get(operation, {
        "error": f"{service} service temporarily unavailable",
        "fallback": "Operation skipped due to service unavailability"
    })
    
    return {
        "success": False,
        "error": operation_fallback["error"],
        "fallback": operation_fallback["fallback"],
        "service": service,
        "operation": operation,
        "original_error": str(error)
    }
