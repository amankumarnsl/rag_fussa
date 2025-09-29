"""
Structured logging configuration for RAG FUSSA API
"""
import sys
import os
from typing import Dict, Any, Optional
from loguru import logger
import time
import uuid
from contextvars import ContextVar
from fastapi import Request
import json

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
conversation_id_var: ContextVar[Optional[str]] = ContextVar('conversation_id', default=None)

class RequestContextFilter:
    """Loguru filter to add request context to logs"""
    
    def __call__(self, record):
        # Add request context
        record["request_id"] = request_id_var.get() or "unknown"
        record["user_id"] = user_id_var.get() or "anonymous"
        record["conversation_id"] = conversation_id_var.get() or "none"
        
        # Add timestamp in ISO format
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(record["time"].timestamp()))
        
        return True

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup structured logging with loguru
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Remove default logger
    logger.remove()
    
    # Console logging with JSON format for production
    if os.getenv("ENVIRONMENT", "development") == "production":
        # JSON format for production (easier to parse by log aggregators)
        console_format = (
            "{timestamp} | {level} | {name}:{function}:{line} | "
            "request_id={request_id} | user_id={user_id} | conversation_id={conversation_id} | "
            "{message}"
        )
    else:
        # Pretty format for development
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<blue>req:{request_id}</blue> | <yellow>user:{user_id}</yellow> | <magenta>conv:{conversation_id}</magenta> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        filter=RequestContextFilter(),
        colorize=(os.getenv("ENVIRONMENT", "development") != "production")
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=console_format,
            level=log_level,
            filter=RequestContextFilter(),
            rotation="100 MB",
            retention="30 days",
            compression="zip"
        )
    
    # Add error file handler
    error_log_file = log_file.replace('.log', '_errors.log') if log_file else "logs/errors.log"
    logger.add(
        error_log_file,
        format=console_format,
        level="ERROR",
        filter=RequestContextFilter(),
        rotation="50 MB",
        retention="90 days",
        compression="zip"
    )
    
    logger.info("Structured logging initialized", log_level=log_level, log_file=log_file)

def get_logger(name: str = None):
    """Get a logger instance with the specified name"""
    if name:
        return logger.bind(name=name)
    return logger

def log_request_start(request: Request, user_id: str = None, conversation_id: str = None):
    """Log the start of a request"""
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    user_id_var.set(user_id)
    conversation_id_var.set(conversation_id)
    
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    return request_id

def log_request_end(request_id: str, status_code: int, duration_ms: float):
    """Log the end of a request"""
    logger.info(
        "Request completed",
        status_code=status_code,
        duration_ms=duration_ms
    )

def log_api_call(service: str, endpoint: str, method: str = "GET", **kwargs):
    """Log external API calls"""
    logger.info(
        "External API call",
        service=service,
        endpoint=endpoint,
        method=method,
        **kwargs
    )

def log_api_response(service: str, endpoint: str, status_code: int, duration_ms: float, **kwargs):
    """Log external API responses"""
    logger.info(
        "External API response",
        service=service,
        endpoint=endpoint,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs
    )

def log_processing_step(step: str, file_type: str, filename: str, **kwargs):
    """Log document processing steps"""
    logger.info(
        "Processing step",
        step=step,
        file_type=file_type,
        filename=filename,
        **kwargs
    )

def log_conversation_event(event_type: str, conversation_id: str, **kwargs):
    """Log conversation-related events"""
    logger.info(
        "Conversation event",
        event_type=event_type,
        conversation_id=conversation_id,
        **kwargs
    )

def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log errors with context"""
    logger.error(
        "Error occurred",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {}
    )

def log_performance_metric(metric_name: str, value: float, unit: str = "ms", **kwargs):
    """Log performance metrics"""
    logger.info(
        "Performance metric",
        metric_name=metric_name,
        value=value,
        unit=unit,
        **kwargs
    )
