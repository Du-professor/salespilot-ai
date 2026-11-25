from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import (
    index_admin, knowledge_bases, library, model_configs, recommendations, session,
)
from app.api.errors import install_api_errors
from app.auth.oidc import OidcVerifier
from app.auth.sessions import seed_demo_account
from app.contracts import ApiError, HealthStatus
from app.contracts.rag_workspace import RagEngine
from app.infrastructure.distributed_lock import (
    RedisIndexWriteLock, SqliteIndexWriteLock, build_redis_client,
)
from app.infrastructure.model_configs import ModelConfigs
from app.infrastructure.object_store import ObjectStore, build_object_store
from app.infrastructure.publish_queue import LibraryPublishQueue
from app.infrastructure.sqlite import Database
from app.settings import Settings


def create_app(
    settings: Settings | None = None,
    *,
    rag_engine: RagEngine | None = None,
    object_store: ObjectStore | None = None,
    redis_client=None,
    oidc_verifier: OidcVerifier | None = None,
) -> FastAPI:
    settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database = Database(settings.database_path)
        database.initialize()
        for username, password in (
            ("sales-a", settings.demo_password_a),
            ("sales-b", settings.demo_password_b),
        ):
            if password is not None:
                seed_demo_account(database, username, password.get_secret_value())
        active_redis = redis_client
        if active_redis is None and settings.redis_url is not None:
            active_redis = build_redis_client(settings.redis_url.get_secret_value())
        database.index_lock = (
            RedisIndexWriteLock(active_redis) if active_redis is not None else SqliteIndexWriteLock(database.path)
        )
        app.state.redis = active_redis
        app.state.database = database
        app.state.object_store = object_store if object_store is not None else build_object_store(settings)
        active_oidc = oidc_verifier
        if active_oidc is None and settings.oidc_issuer and settings.oidc_audience:
            active_oidc = OidcVerifier(
                issuer=settings.oidc_issuer,
                audience=settings.oidc_audience,
                jwks_url=settings.oidc_jwks_url,
                default_org=settings.oidc_default_org,
                org_claim=settings.oidc_org_claim,
            )
        app.state.oidc_verifier = active_oidc
        publish_queue = LibraryPublishQueue(database, redis_client=active_redis)
        publish_queue.recover_interrupted()
        app.state.publish_queue = publish_queue
        model_store = ModelConfigs(database, settings.model_config_key, settings.model_allowed_base_urls)
        app.state.model_configs = model_store
        app.state.workspace_model_routing = rag_engine is None and settings.rag_mode in {"demo", "local"}
        active_rag_engine = rag_engine
        if app.state.workspace_model_routing:
            from app.rag import RagSettings, build_rag_engine

            env_rag = RagSettings.from_environment()
            index_path = settings.rag_index_path or settings.database_path.with_name(
                f"{settings.database_path.stem}-rag.sqlite3",
            )
            updates = {
                "index_path": index_path,
                "store_backend": settings.rag_store_backend,
                "postgres_dsn": settings.postgres_dsn,
                "models_dir": settings.rag_models_dir,
                "embedding_device": settings.rag_embedding_device,
                "docling_artifacts": settings.docling_artifacts,
            }
            if settings.rag_mode == "local":
                updates["embedding_provider"] = "local"
                updates["embedding_model_id"] = settings.rag_embedding_model_id
            else:
                updates["embedding_provider"] = "hash"
                updates["chat_provider"] = "template"
            rag_settings = env_rag.model_copy(update=updates)
            active_rag_engine = build_rag_engine(
                rag_settings,
                chat_resolver=model_store.chat_for_org,
                remote_embedding_resolver=model_store.resolve_remote_embedding,
                reranker_resolver=model_store.reranker_for_org,
                parser_resolver=model_store.parser_for_org,
            )
        app.state.rag_engine = active_rag_engine
        recover = getattr(active_rag_engine, "recover_rebuilds", None)
        if callable(recover):
            recover()
        try:
            yield
        finally:
            publish_queue.shutdown()

    app = FastAPI(
        title="SalesPilot AI",
        version="0.3.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    install_api_errors(app)
    for router in (
        session.router, library.router, knowledge_bases.router, index_admin.router,
        recommendations.router, model_configs.router,
    ):
        app.include_router(
            router, prefix="/api/v1",
            responses={code: {"model": ApiError} for code in (401, 403, 404, 409, 422)},
        )

    @app.get("/healthz", response_model=HealthStatus)
    def healthz() -> HealthStatus:
        return HealthStatus(status="ok")

    return app


app = create_app()
