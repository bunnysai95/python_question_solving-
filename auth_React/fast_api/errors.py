import logging
import traceback
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("app.errors")


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    # Log at warning level so this is visible but not treated as an internal error
    logger.warning("HTTP error %s %s: %s", request.method, request.url, exc.detail)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Validation errors commonly originate from user input; log details for debugging
    errors = exc.errors()
    logger.warning("Validation error on %s %s: %s", request.method, request.url, errors)
    return JSONResponse({"detail": errors}, status_code=422)


async def generic_exception_handler(request: Request, exc: Exception):
    # Catch-all handler for uncaught exceptions — keep response generic but log full traceback
    tb = traceback.format_exc()
    logger.error("Unhandled exception on %s %s: %s\n%s", request.method, request.url, str(exc), tb)
    # Do not leak internal error details to clients in production
    return JSONResponse({"detail": "Internal server error"}, status_code=500)
