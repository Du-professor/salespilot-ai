from contextlib import asynccontextmanager

from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.settings import Settings

ORIGIN = "http://test"
PREFIX = "/api/v1"
PASSWORD_A = "demo-a-test-password"
PASSWORD_B = "demo-b-test-password"


@asynccontextmanager
async def application(db_path):
    app = create_app(
        settings=Settings(
            database_path=db_path,
            web_origins=(ORIGIN,),
            demo_password_a=PASSWORD_A,
            demo_password_b=PASSWORD_B,
        ),
    )
    async with app.router.lifespan_context(app):
        yield app


def browser(app):
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url=ORIGIN,
        headers={"Origin": ORIGIN},
    )


async def login(client, username="sales-a", password=PASSWORD_A):
    response = await client.post(
        f"{PREFIX}/session", json={"username": username, "password": password}
    )
    assert response.status_code == 200
    assert "httponly" in response.headers["set-cookie"].lower()
    return response
