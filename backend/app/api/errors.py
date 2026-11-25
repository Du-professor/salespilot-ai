from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.contracts import ApiError


class ApiProblem(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


def _problem_response(request: Request, problem: ApiProblem) -> JSONResponse:
    payload = ApiError(code=problem.code, message=str(problem), request_id=request.state.request_id)
    return JSONResponse(payload.model_dump(), status_code=problem.status)


def install_api_errors(app: FastAPI) -> None:
    @app.middleware("http")
    async def track_request(request: Request, call_next):
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        if request.url.path.startswith("/api/v1/"):
            response.headers["X-Request-ID"] = request.state.request_id
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ApiProblem)
    async def api_problem(request: Request, problem: ApiProblem):
        return _problem_response(request, problem)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return _problem_response(request, ApiProblem(422, "invalid_request", "请求格式或字段类型不正确。"))
