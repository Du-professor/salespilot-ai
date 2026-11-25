import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _validate_digest(digest: str) -> str:
    lowered = digest.lower()
    if not _HEX64.fullmatch(lowered):
        raise ValueError("Invalid SHA-256 object digest")
    return lowered


@runtime_checkable
class ObjectStore(Protocol):
    def put_file(self, source_path: Path, *, digest: str) -> str: ...

    def put_bytes(self, payload: bytes, *, digest: str) -> str: ...

    def open_stream(self, storage_uri: str, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]: ...

    def read_bytes(self, storage_uri: str) -> bytes: ...

    def exists(self, storage_uri: str) -> bool: ...

    def delete(self, storage_uri: str) -> None: ...


class LocalCasObjectStore:
    """Content-addressed local filesystem object store (atomic writes, path-traversal safe)."""

    SCHEME = "cas://"

    def __init__(self, root: Path):
        self.root = root

    def _path_for_digest(self, digest: str) -> Path:
        clean = _validate_digest(digest)
        return self.root / clean[:2] / clean[2:4] / clean

    def _path_for_uri(self, storage_uri: str) -> Path:
        if not storage_uri.startswith(self.SCHEME):
            raise ValueError(f"Unsupported local CAS URI: {storage_uri}")
        return self._path_for_digest(storage_uri[len(self.SCHEME):])

    def put_file(self, source_path: Path, *, digest: str) -> str:
        clean = _validate_digest(digest)
        target = self._path_for_digest(clean)
        uri = f"{self.SCHEME}{clean}"
        if target.is_file():
            return uri
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{clean}.{uuid4().hex}.tmp")
        try:
            shutil.copyfile(source_path, staging)
            os.replace(staging, target)
        except Exception:
            staging.unlink(missing_ok=True)
            raise
        return uri

    def put_bytes(self, payload: bytes, *, digest: str) -> str:
        clean = _validate_digest(digest)
        target = self._path_for_digest(clean)
        uri = f"{self.SCHEME}{clean}"
        if target.is_file():
            return uri
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{clean}.{uuid4().hex}.tmp")
        try:
            staging.write_bytes(payload)
            os.replace(staging, target)
        except Exception:
            staging.unlink(missing_ok=True)
            raise
        return uri

    def open_stream(self, storage_uri: str, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        path = self._path_for_uri(storage_uri)
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    def read_bytes(self, storage_uri: str) -> bytes:
        return self._path_for_uri(storage_uri).read_bytes()

    def exists(self, storage_uri: str) -> bool:
        try:
            return self._path_for_uri(storage_uri).is_file()
        except ValueError:
            return False

    def delete(self, storage_uri: str) -> None:
        try:
            self._path_for_uri(storage_uri).unlink(missing_ok=True)
        except ValueError:
            pass


class S3ObjectStore:
    """S3 / MinIO compatible object store adapter for enterprise deployment."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        prefix: str = "objects/",
        client=None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("S3 对象存储需要安装可选依赖 boto3。") from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=endpoint_url or None,
                region_name=region_name or None,
                aws_access_key_id=access_key or None,
                aws_secret_access_key=secret_key or None,
            )

    def _key_for_digest(self, digest: str) -> str:
        clean = _validate_digest(digest)
        return f"{self.prefix}{clean[:2]}/{clean[2:4]}/{clean}"

    def _parse_uri(self, storage_uri: str) -> tuple[str, str]:
        if not storage_uri.startswith("s3://"):
            raise ValueError(f"Unsupported S3 URI: {storage_uri}")
        rest = storage_uri[len("s3://"):]
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI: {storage_uri}")
        return bucket, key

    def put_file(self, source_path: Path, *, digest: str) -> str:
        key = self._key_for_digest(digest)
        uri = f"s3://{self.bucket}/{key}"
        if not self.exists(uri):
            self._client.upload_file(str(source_path), self.bucket, key)
        return uri

    def put_bytes(self, payload: bytes, *, digest: str) -> str:
        key = self._key_for_digest(digest)
        uri = f"s3://{self.bucket}/{key}"
        if not self.exists(uri):
            self._client.put_object(Bucket=self.bucket, Key=key, Body=payload)
        return uri

    def open_stream(self, storage_uri: str, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        bucket, key = self._parse_uri(storage_uri)
        body = self._client.get_object(Bucket=bucket, Key=key)["Body"]
        try:
            yield from body.iter_chunks(chunk_size=chunk_size)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()

    def read_bytes(self, storage_uri: str) -> bytes:
        return b"".join(self.open_stream(storage_uri))

    def exists(self, storage_uri: str) -> bool:
        try:
            bucket, key = self._parse_uri(storage_uri)
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def delete(self, storage_uri: str) -> None:
        try:
            bucket, key = self._parse_uri(storage_uri)
            self._client.delete_object(Bucket=bucket, Key=key)
        except Exception:
            pass


def build_object_store(settings) -> ObjectStore:
    if settings.object_store_backend == "s3":
        if not settings.s3_bucket:
            raise ValueError("SALESPILOT_S3_BUCKET is required when SALESPILOT_OBJECT_STORE_BACKEND=s3")
        return S3ObjectStore(
            bucket=settings.s3_bucket,
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key.get_secret_value() if settings.s3_secret_key else None,
            prefix=settings.s3_prefix,
        )
    root = settings.object_store_dir or settings.database_path.with_name(
        f"{settings.database_path.stem}-objects",
    )
    return LocalCasObjectStore(root)
