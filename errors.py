import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from schemas import ErrorResponse

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details=None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


def error_response(status_code: int, code: str, message: str, details=None, headers=None):
    body = ErrorResponse(code=code, message=message, details=details)
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)


def error_docs(description: str, examples: dict | None = None) -> dict:
    result = {"model": ErrorResponse, "description": description}
    if examples:
        result["content"] = {"application/json": {"examples": examples}}
    return result


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, error: ServiceError):
        return error_response(error.status_code, error.code, error.message, error.details)

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException):
        message = error.detail if isinstance(error.detail, str) else "Ошибка HTTP-запроса"
        details = None if isinstance(error.detail, str) else error.detail
        return error_response(
            error.status_code, f"HTTP_{error.status_code}", message, details, error.headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError):
        # Exclude raw inputs and ctx: they may contain NaN or non-JSON exceptions.
        details = [
            {"location": list(item["loc"]), "type": item["type"], "message": item["msg"]}
            for item in error.errors()
        ]
        return error_response(422, "VALIDATION_ERROR", "Некорректные данные запроса", details)

    @app.exception_handler(ValueError)
    async def data_error_handler(request: Request, error: ValueError):
        logger.exception("Ошибка подготовки данных: %s", request.url.path)
        return error_response(422, "DATA_PROCESSING_ERROR", "Не удалось обработать данные")

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, error: Exception):
        logger.exception("Необработанная ошибка: %s", request.url.path)
        return error_response(500, "INTERNAL_ERROR", "Внутренняя ошибка сервиса. Повторите запрос позже")
