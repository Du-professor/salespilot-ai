import hashlib
import secrets
import time
from dataclasses import dataclass

from app.infrastructure.sqlite import DEMO_ORG, Database, record_audit

COOKIE_NAME = "salespilot_session"
COOKIE_PATH = "/api/v1"
ITERATIONS = 600_000
DUMMY_HASH = f"{'0' * 32}:{'0' * 64}"


@dataclass(frozen=True)
class Principal:
    org_id: str
    user_id: str
    username: str
    role: str
    groups: tuple[str, ...] = ()
    auth_source: str = "session"

    def allows(self, action: str) -> bool:
        if not self.org_id:
            return False
        if action == "workspace:manage":
            if self.role == "admin":
                return True
            return self.org_id == DEMO_ORG and self.role == "sales" and self.user_id == "sales-a"
        allowed = {
            "admin": {"workspace:manage", "catalog:read", "selection:run", "report:read"},
            "sales": {"catalog:read", "selection:run", "report:read"},
            "catalog_viewer": {"catalog:read"},
        }
        return action in allowed.get(self.role, set())


def _password_hash(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"{salt.hex()}:{digest.hex()}"


def _matches_password(password: str, stored: str) -> bool:
    if ":" not in stored:
        salt = bytes.fromhex("0" * 32)
        secrets.compare_digest(_password_hash(password, salt), DUMMY_HASH)
        return False
    salt, _ = stored.split(":", 1)
    return secrets.compare_digest(_password_hash(password, bytes.fromhex(salt)), stored)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def seed_demo_account(database: Database, username: str, password: str) -> None:
    if not 12 <= len(password) <= 256:
        raise ValueError("Demo passwords must be between 12 and 256 characters")
    with database.connect() as connection:
        account = connection.execute(
            "SELECT password_hash FROM accounts WHERE org_id = ? AND user_id = ?",
            (DEMO_ORG, username),
        ).fetchone()
        if account and _matches_password(password, account["password_hash"]):
            return
        connection.execute(
            """
            INSERT INTO accounts (org_id, user_id, username, role, password_hash)
            VALUES (?, ?, ?, 'sales', ?)
            ON CONFLICT (org_id, user_id) DO UPDATE SET password_hash = excluded.password_hash
            """,
            (DEMO_ORG, username, username, _password_hash(password, secrets.token_bytes(16))),
        )
        connection.execute(
            "DELETE FROM sessions WHERE org_id = ? AND user_id = ?", (DEMO_ORG, username)
        )


def sign_in(
    database: Database,
    username: str,
    password: str,
    session_seconds: int,
    previous_token: str | None,
    request_id: str,
    *,
    org_id: str | None = None,
) -> tuple[Principal, str] | None:
    with database.connect() as connection:
        if org_id:
            account = connection.execute(
                "SELECT * FROM accounts WHERE org_id = ? AND username = ?", (org_id, username)
            ).fetchone()
        else:
            account = connection.execute(
                """SELECT * FROM accounts WHERE username = ?
                ORDER BY CASE WHEN org_id = ? THEN 0 ELSE 1 END LIMIT 1""",
                (username, DEMO_ORG),
            ).fetchone()
        stored = account["password_hash"] if account else DUMMY_HASH
        if not _matches_password(password, stored) or account is None:
            return None
        principal = Principal(
            account["org_id"], account["user_id"], account["username"], account["role"]
        )
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        if previous_token:
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?", (_token_hash(previous_token),)
            )
        connection.execute(
            "INSERT INTO sessions (token_hash, org_id, user_id, expires_at) VALUES (?, ?, ?, ?)",
            (_token_hash(token), principal.org_id, principal.user_id, now + session_seconds),
        )
        record_audit(
            connection, org_id=principal.org_id, user_id=principal.user_id,
            action="session_created", request_id=request_id,
        )
        return principal, token


def find_principal(database: Database, token: str) -> Principal | None:
    with database.connect() as connection:
        account = connection.execute(
            """
            SELECT a.org_id, a.user_id, a.username, a.role FROM sessions AS s
            JOIN accounts AS a ON a.org_id = s.org_id AND a.user_id = s.user_id
            WHERE s.token_hash = ? AND s.expires_at > ?
            """,
            (_token_hash(token), int(time.time())),
        ).fetchone()
        return Principal(**dict(account)) if account else None


def sign_out(database: Database, actor: Principal, token: str, request_id: str) -> None:
    with database.connect() as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
        record_audit(
            connection, org_id=actor.org_id, user_id=actor.user_id,
            action="session_ended", request_id=request_id,
        )


def upsert_oidc_account(database: Database, principal: Principal) -> None:
    """Ensure external OIDC principals have a shadow row in accounts so conversation FKs remain valid."""
    role = principal.role if principal.role in {"admin", "sales", "catalog_viewer"} else "sales"
    with database.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO accounts (org_id, user_id, username, role, password_hash)
            VALUES (?, ?, ?, ?, '!oidc')
            ON CONFLICT (org_id, user_id) DO UPDATE SET
                username = excluded.username,
                role = excluded.role""",
            (principal.org_id, principal.user_id, principal.username, role),
        )
