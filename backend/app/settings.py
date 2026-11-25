import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, PositiveInt, SecretStr

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = PROJECT_DIR / ".salespilot" / "demo.sqlite3"
DEFAULT_MODELS_DIR = PROJECT_DIR / ".salespilot" / "models"
DEFAULT_WEB_ORIGINS = ("http://127.0.0.1:4173", "http://localhost:4173")


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: Literal["demo", "enterprise"] = "demo"
    model_provider: Literal["fixture"] = "fixture"
    database_path: Path = DEFAULT_DATABASE
    web_origins: tuple[str, ...] = DEFAULT_WEB_ORIGINS
    secure_cookie: bool = False
    session_seconds: PositiveInt = 8 * 60 * 60
    demo_password_a: SecretStr | None = None
    demo_password_b: SecretStr | None = None
    docling_artifacts: Path | None = None
    rag_mode: Literal["disabled", "demo", "local"] = "disabled"
    rag_store_backend: Literal["sqlite", "postgres"] = "sqlite"
    postgres_dsn: SecretStr | None = None
    redis_url: SecretStr | None = None
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    oidc_default_org: str | None = None
    oidc_org_claim: str = "org_id"
    rag_index_path: Path | None = None
    rag_embedding_model_id: str = "workspace-default"
    rag_models_dir: Path = DEFAULT_MODELS_DIR
    rag_embedding_device: Literal["cpu", "cuda"] = "cpu"
    rag_rebuild_background: bool = True
    object_store_backend: Literal["local", "s3"] = "local"
    object_store_dir: Path | None = None
    s3_endpoint_url: str | None = None
    s3_bucket: str | None = None
    s3_region: str | None = None
    s3_prefix: str = "objects/"
    s3_access_key: str | None = None
    s3_secret_key: SecretStr | None = None
    model_config_key: SecretStr | None = None
    model_allowed_base_urls: tuple[str, ...] = ()

    @classmethod
    def from_environment(cls) -> Self:
        object_dir = os.getenv("SALESPILOT_OBJECT_STORE_DIR") or None
        return cls(
            profile=os.getenv("SALESPILOT_PROFILE", "demo"),
            model_provider=os.getenv("SALESPILOT_MODEL_PROVIDER", "fixture"),
            database_path=os.getenv("SALESPILOT_DB_PATH", str(DEFAULT_DATABASE)),
            web_origins=tuple(
                origin.strip()
                for origin in os.getenv(
                    "SALESPILOT_WEB_ORIGINS", ",".join(DEFAULT_WEB_ORIGINS)
                ).split(",")
                if origin.strip()
            ),
            secure_cookie=os.getenv("SALESPILOT_SECURE_COOKIE", "false"),
            demo_password_a=os.getenv("SALESPILOT_DEMO_PASSWORD_A"),
            demo_password_b=os.getenv("SALESPILOT_DEMO_PASSWORD_B"),
            docling_artifacts=os.getenv("SALESPILOT_DOCLING_ARTIFACTS") or None,
            rag_mode=os.getenv("SALESPILOT_RAG_MODE", "disabled"),
            rag_store_backend=os.getenv("SALESPILOT_RAG_STORE_BACKEND", "sqlite"),
            postgres_dsn=os.getenv("SALESPILOT_POSTGRES_DSN") or None,
            redis_url=os.getenv("SALESPILOT_REDIS_URL") or None,
            oidc_issuer=os.getenv("SALESPILOT_OIDC_ISSUER") or None,
            oidc_audience=os.getenv("SALESPILOT_OIDC_AUDIENCE") or None,
            oidc_jwks_url=os.getenv("SALESPILOT_OIDC_JWKS_URL") or None,
            oidc_default_org=os.getenv("SALESPILOT_OIDC_DEFAULT_ORG") or None,
            oidc_org_claim=os.getenv("SALESPILOT_OIDC_ORG_CLAIM", "org_id"),
            rag_index_path=os.getenv("SALESPILOT_RAG_INDEX_PATH") or None,
            rag_embedding_model_id=os.getenv("SALESPILOT_RAG_EMBEDDING_MODEL_ID", "workspace-default"),
            rag_models_dir=Path(os.getenv("SALESPILOT_RAG_MODELS_DIR", str(DEFAULT_MODELS_DIR))),
            rag_embedding_device=os.getenv("SALESPILOT_RAG_EMBEDDING_DEVICE", "cpu"),
            rag_rebuild_background=os.getenv("SALESPILOT_RAG_REBUILD_BACKGROUND", "true").lower() not in {"0", "false", "no"},
            object_store_backend=os.getenv("SALESPILOT_OBJECT_STORE_BACKEND", "local"),
            object_store_dir=Path(object_dir) if object_dir else None,
            s3_endpoint_url=os.getenv("SALESPILOT_S3_ENDPOINT_URL") or None,
            s3_bucket=os.getenv("SALESPILOT_S3_BUCKET") or None,
            s3_region=os.getenv("SALESPILOT_S3_REGION") or None,
            s3_prefix=os.getenv("SALESPILOT_S3_PREFIX", "objects/"),
            s3_access_key=os.getenv("SALESPILOT_S3_ACCESS_KEY") or None,
            s3_secret_key=os.getenv("SALESPILOT_S3_SECRET_KEY") or None,
            model_config_key=os.getenv("SALESPILOT_MODEL_CONFIG_KEY") or None,
            model_allowed_base_urls=tuple(
                url.strip() for url in os.getenv("SALESPILOT_MODEL_ALLOWED_BASE_URLS", "").split(",")
                if url.strip()
            ),
        )
