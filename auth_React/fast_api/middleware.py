import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("app.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        try:
            response = await call_next(request)
        except Exception:
            # let the global exception handler manage it; still log timing
            duration = (time.time() - start) * 1000
            logger.exception("Request %s %s failed after %.2fms", request.method, request.url.path, duration)
            raise
        duration = (time.time() - start) * 1000
        logger.info("%s %s -> %s (%.2fms)", request.method, request.url.path, response.status_code, duration)
        return response
