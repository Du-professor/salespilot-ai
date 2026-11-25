"""Keycloak / OIDC JWT (RS256 + JWKS) bearer token verification."""
from __future__ import annotations

import base64
import json
import re
import threading
import time
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from app.auth.sessions import Principal

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")


class OidcTokenError(ValueError):
    """Raised when an OIDC Bearer JWT is malformed, expired, untrusted, or fails signature validation."""


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeEncodeError) as exc:
        raise OidcTokenError("invalid base64url segment") from exc


def _rsa_public_key_from_jwk(jwk: dict[str, Any]) -> rsa.RSAPublicKey:
    if jwk.get("kty") != "RSA" or "n" not in jwk or "e" not in jwk:
        raise OidcTokenError("unsupported JWK key type")
    n = int.from_bytes(_b64url_decode(str(jwk["n"])), "big")
    e = int.from_bytes(_b64url_decode(str(jwk["e"])), "big")
    return rsa.RSAPublicNumbers(e, n).public_key()


def _normalize_group(raw: str) -> str:
    cleaned = raw.strip().strip("/")
    return cleaned


class OidcVerifier:
    """Stateless OIDC JWT verifier supporting Keycloak realm/client roles, groups, and org claims."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str | None = None,
        default_org: str | None = None,
        org_claim: str = "org_id",
        jwks_ttl_seconds: float = 300.0,
        leeway_seconds: int = 30,
        timeout_seconds: float = 10.0,
    ):
        self.issuer = issuer.strip().rstrip("/")
        self.audience = audience.strip()
        self.jwks_url = (jwks_url.strip() if jwks_url else f"{self.issuer}/protocol/openid-connect/certs")
        self.default_org = default_org.strip() if default_org else None
        self.org_claim = org_claim.strip() or "org_id"
        self.jwks_ttl_seconds = max(10.0, float(jwks_ttl_seconds))
        self.leeway_seconds = max(0, int(leeway_seconds))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._lock = threading.Lock()
        self._keys_by_kid: dict[str, rsa.RSAPublicKey] = {}
        self._fallback_keys: tuple[rsa.RSAPublicKey, ...] = ()
        self._fetched_at: float = 0.0

    def _refresh_jwks(self) -> None:
        try:
            response = httpx.get(self.jwks_url, timeout=self.timeout_seconds)
            if response.status_code >= 400:
                raise OidcTokenError(f"JWKS endpoint returned HTTP {response.status_code}")
            payload = response.json()
            raw_keys = payload.get("keys")
            if not isinstance(raw_keys, list):
                raise OidcTokenError("JWKS missing keys array")
        except (httpx.HTTPError, ValueError) as exc:
            raise OidcTokenError(f"failed to load OIDC JWKS: {exc}") from exc

        by_kid: dict[str, rsa.RSAPublicKey] = {}
        fallbacks: list[rsa.RSAPublicKey] = []
        for item in raw_keys:
            if not isinstance(item, dict) or item.get("kty") != "RSA":
                continue
            if item.get("use") not in (None, "sig"):
                continue
            try:
                pub = _rsa_public_key_from_jwk(item)
            except OidcTokenError:
                continue
            kid = item.get("kid")
            if isinstance(kid, str) and kid:
                by_kid[kid] = pub
            else:
                fallbacks.append(pub)
        with self._lock:
            self._keys_by_kid = by_kid
            self._fallback_keys = tuple(fallbacks)
            self._fetched_at = time.monotonic()

    def _candidate_keys(self, kid: str | None) -> tuple[rsa.RSAPublicKey, ...]:
        now = time.monotonic()
        with self._lock:
            stale = (now - self._fetched_at) >= self.jwks_ttl_seconds or (not self._keys_by_kid and not self._fallback_keys)
            matched = self._keys_by_kid.get(kid) if kid else None
        if stale or (kid and matched is None):
            self._refresh_jwks()
        with self._lock:
            if kid and kid in self._keys_by_kid:
                return (self._keys_by_kid[kid],)
            if not kid:
                return tuple(self._keys_by_kid.values()) + self._fallback_keys
            return ()

    def verify(self, token: str) -> Principal:
        parts = token.strip().split(".")
        if len(parts) != 3:
            raise OidcTokenError("malformed JWT")
        header_b64, payload_b64, sig_b64 = parts
        try:
            header = json.loads(_b64url_decode(header_b64))
            claims = json.loads(_b64url_decode(payload_b64))
        except (ValueError, UnicodeDecodeError) as exc:
            raise OidcTokenError("invalid JWT JSON") from exc
        if not isinstance(header, dict) or not isinstance(claims, dict):
            raise OidcTokenError("invalid JWT structure")
        if header.get("alg") != "RS256":
            raise OidcTokenError("unsupported JWT alg; expected RS256")

        signature = _b64url_decode(sig_b64)
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        kid = header.get("kid") if isinstance(header.get("kid"), str) else None
        candidates = self._candidate_keys(kid)
        if not candidates:
            raise OidcTokenError("no matching JWK found")

        verified = False
        for public_key in candidates:
            try:
                public_key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
                verified = True
                break
            except InvalidSignature:
                continue
        if not verified:
            raise OidcTokenError("invalid JWT signature")

        now = int(time.time())
        iss = str(claims.get("iss") or "").rstrip("/")
        if iss != self.issuer:
            raise OidcTokenError("JWT issuer mismatch")

        aud = claims.get("aud")
        azp = claims.get("azp")
        aud_list = [aud] if isinstance(aud, str) else (list(aud) if isinstance(aud, list) else [])
        if self.audience not in aud_list and azp != self.audience:
            raise OidcTokenError("JWT audience mismatch")

        exp = claims.get("exp")
        if not isinstance(exp, (int, float)) or now > int(exp) + self.leeway_seconds:
            raise OidcTokenError("JWT expired")
        nbf = claims.get("nbf")
        if nbf is not None and (not isinstance(nbf, (int, float)) or now + self.leeway_seconds < int(nbf)):
            raise OidcTokenError("JWT not yet valid")

        sub = str(claims.get("sub") or "").strip()
        if not _SAFE_ID.match(sub):
            raise OidcTokenError("invalid JWT sub claim")

        raw_org = claims.get(self.org_claim) or claims.get("tenant_id") or self.default_org
        org_id = str(raw_org or "").strip()
        if not _SAFE_ID.match(org_id):
            raise OidcTokenError("missing or invalid organization claim")

        username = str(claims.get("preferred_username") or claims.get("email") or sub).strip()[:128]
        role = self._extract_role(claims)
        groups = self._extract_groups(claims)
        return Principal(
            org_id=org_id,
            user_id=sub,
            username=username or sub,
            role=role,
            groups=groups,
            auth_source="oidc",
        )

    def _extract_role(self, claims: dict[str, Any]) -> str:
        collected: set[str] = set()
        realm_access = claims.get("realm_access")
        if isinstance(realm_access, dict) and isinstance(realm_access.get("roles"), list):
            collected.update(str(r).lower() for r in realm_access["roles"])
        resource_access = claims.get("resource_access")
        if isinstance(resource_access, dict):
            client_entry = resource_access.get(self.audience)
            if isinstance(client_entry, dict) and isinstance(client_entry.get("roles"), list):
                collected.update(str(r).lower() for r in client_entry["roles"])
        direct_roles = claims.get("roles")
        if isinstance(direct_roles, list):
            collected.update(str(r).lower() for r in direct_roles)
        elif isinstance(claims.get("role"), str):
            collected.add(claims["role"].lower())

        if collected & {"admin", "workspace_admin", "salespilot-admin", "workspace:manage"}:
            return "admin"
        if collected & {"sales", "salespilot-sales"}:
            return "sales"
        if collected & {"catalog_viewer", "viewer", "salespilot-viewer"}:
            return "catalog_viewer"
        return "sales"

    @staticmethod
    def _extract_groups(claims: dict[str, Any]) -> tuple[str, ...]:
        raw_groups = claims.get("groups")
        if not isinstance(raw_groups, list):
            return ()
        normalized = []
        for item in raw_groups:
            if not isinstance(item, str):
                continue
            cleaned = _normalize_group(item)
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return tuple(normalized[:64])
