from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from pydantic import Field

from app.api.errors import ApiProblem
from app.auth.oidc import OidcTokenError
from app.auth.sessions import (
    COOKIE_NAME, COOKIE_PATH, Principal, find_principal, sign_in, sign_out, upsert_oidc_account,
)
from app.contracts.common import ContractModel

router = APIRouter(prefix="/session", tags=["session"])


class LoginRequest(ContractModel):
    username: Annotated[str, Field(strict=True, min_length=1, max_length=64)]
    password: Annotated[str, Field(strict=True, min_length=1, max_length=256)]


class SessionUser(ContractModel):
    org_id: str
    user_id: str
    username: str
    role: str
    can_manage_workspace: bool
    groups: tuple[str, ...] = ()
    auth_source: str = "session"


class AuthConfigView(ContractModel):
    demo_login_enabled: bool
    oidc_enabled: bool
    oidc_issuer: str | None = None
    oidc_audience: str | None = None


def _session_user(actor: Principal) -> SessionUser:
    return SessionUser(
        org_id=actor.org_id, user_id=actor.user_id, username=actor.username, role=actor.role,
        can_manage_workspace=actor.allows("workspace:manage"),
        groups=actor.groups,
        auth_source=actor.auth_source,
    )


def check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is not None and origin not in request.app.state.settings.web_origins:
        raise ApiProblem(403, "origin_not_allowed", "请求来源无法访问此工作区。")


def require_principal(request: Request) -> Principal:
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        verifier = getattr(request.app.state, "oidc_verifier", None)
        if verifier is None:
            raise ApiProblem(401, "oidc_not_configured", "当前服务端未启用 OIDC 令牌认证。")
        raw_token = auth_header[7:].strip()
        try:
            actor = verifier.verify(raw_token)
        except OidcTokenError:
            raise ApiProblem(401, "invalid_oidc_token", "OIDC 访问令牌无效或已过期。") from None
        upsert_oidc_account(request.app.state.database, actor)
        return actor
    token = request.cookies.get(COOKIE_NAME)
    actor = find_principal(request.app.state.database, token) if token else None
    if actor is None:
        raise ApiProblem(401, "authentication_required", "请登录 SalesPilot AI。")
    return actor


def require_permission(actor: Principal, action: str) -> None:
    if not actor.allows(action):
        raise ApiProblem(403, "permission_denied", "当前账号没有此操作权限。")


@router.post("", response_model=SessionUser, dependencies=[Depends(check_origin)])
def create_session(body: LoginRequest, request: Request, response: Response) -> SessionUser:
    settings = request.app.state.settings
    match = sign_in(
        request.app.state.database, body.username.strip(), body.password,
        settings.session_seconds, request.cookies.get(COOKIE_NAME), request.state.request_id,
    )
    if match is None:
        raise ApiProblem(401, "invalid_credentials", "账号或密码不正确。")
    actor, token = match
    response.set_cookie(
        COOKIE_NAME, token, max_age=settings.session_seconds, path=COOKIE_PATH,
        httponly=True, secure=settings.secure_cookie, samesite="lax",
    )
    return _session_user(actor)


@router.get("/auth-config", response_model=AuthConfigView)
def read_auth_config(request: Request) -> AuthConfigView:
    settings = request.app.state.settings
    verifier = getattr(request.app.state, "oidc_verifier", None)
    demo_enabled = bool(settings.demo_password_a or settings.demo_password_b)
    return AuthConfigView(
        demo_login_enabled=demo_enabled,
        oidc_enabled=verifier is not None,
        oidc_issuer=verifier.issuer if verifier is not None else None,
        oidc_audience=verifier.audience if verifier is not None else None,
    )


@router.get("", response_model=SessionUser)
def read_session(actor: Annotated[Principal, Depends(require_principal)]) -> SessionUser:
    return _session_user(actor)


@router.delete("", status_code=204, dependencies=[Depends(check_origin)])
def end_session(request: Request, actor: Annotated[Principal, Depends(require_principal)]) -> Response:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        sign_out(
            request.app.state.database, actor, token, request.state.request_id,
        )
    response = Response(status_code=204)
    response.delete_cookie(
        COOKIE_NAME, path=COOKIE_PATH, httponly=True,
        secure=request.app.state.settings.secure_cookie, samesite="lax",
    )
    return response
