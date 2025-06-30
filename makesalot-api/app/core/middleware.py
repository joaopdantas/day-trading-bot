"""
Custom middleware for request logging and metrics
"""
from fastapi import Request
import time
import logging
from prometheus_client import Counter, Histogram

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_request_count_total',
    'Total count of HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_latency_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        response = None

        try:
            response = await call_next(request)

            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()

            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(time.time() - start_time)

            # Log request details
            logger.info(
                "Request processed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration": time.time() - start_time
                }
            )

            return response

        except Exception as e:
            logger.error(
                "Request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration": time.time() - start_time
                }
            )
            raise
