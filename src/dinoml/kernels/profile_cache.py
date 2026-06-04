from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from dinoml.ir import canonical_json
from dinoml.kernels.manifest import PROFILE_CACHE_SCHEMA_VERSION

# Exact lookup stays hash-based, while normalized metadata columns support
# auditing and future non-SQLite backends without changing profiling call sites.

@dataclass(frozen=True)
class ProfileCacheLookup:
    profile_key: str
    key_payload: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ProfileCacheWrite:
    lookup: ProfileCacheLookup
    entry: Mapping[str, Any]


class ProfileCacheBackend(Protocol):
    def lookup_many(self, lookups: Sequence[ProfileCacheLookup]) -> dict[str, dict[str, Any]]:
        ...

    def upsert_many(self, writes: Sequence[ProfileCacheWrite]) -> None:
        ...

    def close(self) -> None:
        ...


def default_profile_cache_path(codegen_plan: Mapping[str, Any]) -> Path:
    support_cache_dir = Path(str(codegen_plan["support_cache_dir"]))
    return support_cache_dir.parent / f"profile_cache.v{PROFILE_CACHE_SCHEMA_VERSION}.sqlite3"


def open_profile_cache_backend(codegen_plan: Mapping[str, Any]) -> ProfileCacheBackend:
    return SQLiteProfileCacheBackend(default_profile_cache_path(codegen_plan))


class SQLiteProfileCacheBackend:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def lookup_many(self, lookups: Sequence[ProfileCacheLookup]) -> dict[str, dict[str, Any]]:
        keys = sorted({str(lookup.profile_key) for lookup in lookups if lookup.profile_key})
        if not keys:
            return {}
        placeholders = ", ".join("?" for _ in keys)
        rows = self._conn.execute(
            f"""
            SELECT
                profile_key,
                key_payload_json,
                entry_json
            FROM profile_cache_entries
            WHERE profile_key IN ({placeholders})
            """,
            keys,
        ).fetchall()
        entries: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = _decode_json_mapping(row["entry_json"])
            if not entry:
                continue
            key_payload = _decode_json_mapping(row["key_payload_json"])
            if not key_payload:
                continue
            profile_key = str(row["profile_key"])
            if str(entry.get("profile_key", "")) != profile_key:
                continue
            if not isinstance(entry.get("key"), Mapping):
                entry["key"] = key_payload
            entries[profile_key] = entry
        return entries

    def upsert_many(self, writes: Sequence[ProfileCacheWrite]) -> None:
        rows = []
        for write in writes:
            lookup = write.lookup
            metadata = dict(lookup.metadata)
            entry = dict(write.entry)
            if str(entry.get("profile_key", "")) != str(lookup.profile_key):
                continue
            rows.append(
                (
                    str(lookup.profile_key),
                    int(metadata.get("schema_version", PROFILE_CACHE_SCHEMA_VERSION)),
                    str(metadata.get("target_name", "")),
                    str(metadata.get("target_arch", "")),
                    str(metadata.get("hardware_fingerprint_key", "")),
                    str(metadata.get("support_library_name", "")),
                    str(metadata.get("support_fingerprint_key", "")),
                    str(metadata.get("kernel_library", "")),
                    str(metadata.get("op_family", "")),
                    str(metadata.get("op", "")),
                    str(metadata.get("dtype", "")),
                    _optional_text(metadata.get("candidate_set_key")),
                    str(metadata.get("candidate_id", "")),
                    _optional_text(metadata.get("candidate_config_key")),
                    str(metadata.get("provider_problem_key", "")),
                    str(metadata.get("problem_key", "")),
                    str(metadata.get("semantics_key", "")),
                    str(metadata.get("variant_key", "")),
                    canonical_json(lookup.key_payload),
                    canonical_json(entry),
                    str(entry.get("updated_at", "")),
                )
            )
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO profile_cache_entries (
                    profile_key,
                    schema_version,
                    target_name,
                    target_arch,
                    hardware_fingerprint_key,
                    support_library_name,
                    support_fingerprint_key,
                    kernel_library,
                    op_family,
                    op,
                    dtype,
                    candidate_set_key,
                    candidate_id,
                    candidate_config_key,
                    provider_problem_key,
                    problem_key,
                    semantics_key,
                    variant_key,
                    key_payload_json,
                    entry_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_key) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    target_name = excluded.target_name,
                    target_arch = excluded.target_arch,
                    hardware_fingerprint_key = excluded.hardware_fingerprint_key,
                    support_library_name = excluded.support_library_name,
                    support_fingerprint_key = excluded.support_fingerprint_key,
                    kernel_library = excluded.kernel_library,
                    op_family = excluded.op_family,
                    op = excluded.op,
                    dtype = excluded.dtype,
                    candidate_set_key = excluded.candidate_set_key,
                    candidate_id = excluded.candidate_id,
                    candidate_config_key = excluded.candidate_config_key,
                    provider_problem_key = excluded.provider_problem_key,
                    problem_key = excluded.problem_key,
                    semantics_key = excluded.semantics_key,
                    variant_key = excluded.variant_key,
                    key_payload_json = excluded.key_payload_json,
                    entry_json = excluded.entry_json,
                    updated_at = excluded.updated_at
                """,
                rows,
            )

    def _configure(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_cache_entries (
                    profile_key TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    target_name TEXT NOT NULL,
                    target_arch TEXT NOT NULL,
                    hardware_fingerprint_key TEXT NOT NULL,
                    support_library_name TEXT NOT NULL,
                    support_fingerprint_key TEXT NOT NULL,
                    kernel_library TEXT NOT NULL,
                    op_family TEXT NOT NULL,
                    op TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    candidate_set_key TEXT,
                    candidate_id TEXT NOT NULL,
                    candidate_config_key TEXT,
                    provider_problem_key TEXT NOT NULL,
                    problem_key TEXT NOT NULL,
                    semantics_key TEXT NOT NULL,
                    variant_key TEXT NOT NULL,
                    key_payload_json TEXT NOT NULL,
                    entry_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS profile_cache_entries_problem_idx
                ON profile_cache_entries (
                    target_name,
                    target_arch,
                    kernel_library,
                    op,
                    dtype,
                    candidate_set_key
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS profile_cache_entries_provider_problem_idx
                ON profile_cache_entries (provider_problem_key)
                """
            )


def _decode_json_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, str) or not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
