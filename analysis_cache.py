"""Shared GitHub-backed cache for Overall Program Analysis reports.

Cache is best-effort: GitHub is the durable shared store, while a process-local
lock prevents duplicate work for the same key within one running app process.
It does not provide a distributed lock across separate Streamlit workers.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import streamlit as st

CACHE_SCHEMA_VERSION = "overall-program-analysis-v1"
CACHE_DIR = "analysis_cache/overall_program_analysis"
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_LAST_CACHE_DIAGNOSTIC: dict[str, Any] = {"status": "not_checked", "elapsed_seconds": 0.0, "detail": ""}


def _setting(name: str, default: str = "") -> str:
    """Read nested [github_cache] secrets first, then flat secrets/env vars."""
    try:
        nested = st.secrets.get("github_cache", {})
        value = nested.get(name.lower()) or nested.get(name) if hasattr(nested, "get") else None
        if value:
            return str(value).strip()
    except Exception:
        pass
    for key in (f"GITHUB_CACHE_{name.upper()}", f"GITHUB_{name.upper()}"):
        try:
            value = st.secrets.get(key)
            if value:
                return str(value).strip()
        except Exception:
            pass
        value = os.getenv(key)
        if value:
            return value.strip()
    return default


def cache_is_configured() -> bool:
    return bool(_setting("token") and _setting("repository"))


def file_fingerprint(path: str | Path | None) -> str:
    if path is None:
        return "missing"
    p = Path(path)
    if not p.exists():
        return "missing"
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_cache_key(program_code: str, primary_data_path: str | Path,
                   advisor_data_path: str | Path | None = None,
                   prompt_version: str = CACHE_SCHEMA_VERSION) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "prompt_version": prompt_version,
        "analysis_type": "Overall program analysis",
        "program_code": str(program_code).strip(),
        "primary_data_sha256": file_fingerprint(primary_data_path),
        "advisor_data_sha256": file_fingerprint(advisor_data_path),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _repo_settings() -> tuple[str, str, str]:
    repository = _setting("repository")
    token = _setting("token")
    branch = _setting("branch", "main") or "main"
    if "/" not in repository:
        raise ValueError("GitHub cache repository must be in OWNER/REPO format.")
    return repository, token, branch


def _api_url(repository: str, key: str) -> str:
    path = f"{CACHE_DIR}/{key}.json"
    return f"https://api.github.com/repos/{repository}/contents/{quote(path, safe='/')}"


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Program-Performance-Agent-Cache",
    }


def get_cached_report(key: str, timeout: float = 8.0) -> dict[str, Any] | None:
    """Return a valid shared cached report; retain diagnostics for the UI."""
    global _LAST_CACHE_DIAGNOSTIC
    started = time.perf_counter()
    if not cache_is_configured():
        _LAST_CACHE_DIAGNOSTIC = {
            "status": "not_configured",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "detail": "GitHub cache token or repository is missing.",
        }
        return None
    try:
        repository, token, branch = _repo_settings()
        response = requests.get(
            _api_url(repository, key), headers=_headers(token),
            params={"ref": branch}, timeout=timeout,
        )
        elapsed = round(time.perf_counter() - started, 3)
        if response.status_code == 404:
            _LAST_CACHE_DIAGNOSTIC = {
                "status": "miss_404", "elapsed_seconds": elapsed,
                "detail": f"No cache file for key {key[:12]} on branch {branch}.",
            }
            return None
        if response.status_code in (401, 403):
            _LAST_CACHE_DIAGNOSTIC = {
                "status": "read_forbidden", "elapsed_seconds": elapsed,
                "detail": f"GitHub returned HTTP {response.status_code}; check token access and repository permissions.",
            }
            return None
        response.raise_for_status()
        content = response.json().get("content", "")
        decoded = base64.b64decode(content).decode("utf-8")
        payload = json.loads(decoded)
        if payload.get("cache_key") != key or not isinstance(payload.get("report"), str):
            _LAST_CACHE_DIAGNOSTIC = {
                "status": "invalid_payload", "elapsed_seconds": elapsed,
                "detail": "The JSON file exists, but its cache_key/report fields do not match this request.",
            }
            return None
        _LAST_CACHE_DIAGNOSTIC = {
            "status": "hit", "elapsed_seconds": elapsed,
            "detail": f"Retrieved {CACHE_DIR}/{key}.json from branch {branch}.",
        }
        return payload
    except requests.Timeout:
        _LAST_CACHE_DIAGNOSTIC = {
            "status": "timeout", "elapsed_seconds": round(time.perf_counter() - started, 3),
            "detail": f"GitHub cache read timed out after {timeout} seconds.",
        }
        return None
    except Exception as exc:
        _LAST_CACHE_DIAGNOSTIC = {
            "status": "read_error", "elapsed_seconds": round(time.perf_counter() - started, 3),
            "detail": f"{type(exc).__name__}: {exc}",
        }
        return None


def get_cache_diagnostic() -> dict[str, Any]:
    """Return status/timing for the most recent cache read in this process."""
    return dict(_LAST_CACHE_DIAGNOSTIC)


def put_cached_report(key: str, report: str, metadata: dict[str, Any] | None = None,
                      timeout: float = 10.0) -> bool:
    """Write report to GitHub. If another request won the race, preserve its copy."""
    if not cache_is_configured():
        return False
    try:
        repository, token, branch = _repo_settings()
        url = _api_url(repository, key)
        headers = _headers(token)
        # Don't create another commit when a same-key report already exists.
        existing = requests.get(url, headers=headers, params={"ref": branch}, timeout=timeout)
        if existing.status_code == 200:
            return True
        if existing.status_code != 404:
            existing.raise_for_status()
        payload = {
            "cache_key": key,
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "report": report,
            "metadata": metadata or {},
        }
        encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")).decode("ascii")
        body = {
            "message": f"Cache Overall Program Analysis {key[:12]}",
            "content": encoded,
            "branch": branch,
        }
        written = requests.put(url, headers=headers, json=body, timeout=timeout)
        if written.status_code in (409, 422):
            # Concurrent request may have created the same key. Read it back.
            retry = requests.get(url, headers=headers, params={"ref": branch}, timeout=timeout)
            if retry.status_code == 200:
                return True
        written.raise_for_status()
        return True
    except Exception:
        return False


def get_process_lock(key: str) -> threading.Lock:
    """A per-process lock; does not claim cross-instance locking."""
    return _lock_for(key)
