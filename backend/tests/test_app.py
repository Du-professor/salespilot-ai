import asyncio

from httpx import ASGITransport, AsyncClient

from app.main import create_app


def test_health_check_and_workspace_openapi_are_available():
    async def fetch_documents():
        transport = ASGITransport(app=create_app())
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/healthz"), await client.get("/openapi.json")

    response, schema_response = asyncio.run(fetch_documents())
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert schema_response.status_code == 200
    openapi = schema_response.json()

    paths = openapi["paths"]
    assert "get" in paths["/healthz"]
    assert "/api/v1/library/knowledge-bases" in paths
    assert "/api/v1/library/documents" in paths
    assert "/api/v1/recommendations/turns" in paths
    assert "/api/v1/model-configs" in paths
    # Legacy B0-B3 routes are retired.
    assert "/api/v1/catalog" not in paths
    assert "/api/v1/customers" not in paths
    assert "/api/v1/selections" not in paths
    assert "ApiError" in openapi["components"]["schemas"]
