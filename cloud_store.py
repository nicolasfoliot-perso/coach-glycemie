"""Accès minimal à la Data API Supabase, toujours sous une session utilisateur."""

from __future__ import annotations

from typing import Any

import httpx

from supabase_sync import SupabaseSession, SupabaseSyncError


ALLOWED_TABLES = {
    "meals",
    "weights",
    "symptom_events",
    "bowel_events",
    "activity_events",
    "favorite_foods",
    "profile",
}


def _table_url(session: SupabaseSession, table: str) -> str:
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Table cloud non autorisée : {table}")
    return f"{session.project_url}/rest/v1/{table}"


def _headers(session: SupabaseSession, prefer: str = "") -> dict[str, str]:
    headers = {
        "apikey": session.publishable_key,
        "Authorization": f"Bearer {session.access_token}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        return str(
            payload.get("message")
            or payload.get("error_description")
            or payload.get("hint")
            or payload
        )
    return str(payload)


def _json(response: httpx.Response, table: str) -> list[dict[str, Any]]:
    if response.status_code >= 400:
        raise SupabaseSyncError(f"{table} : {_detail(response)}")
    if not response.content:
        return []
    payload = response.json()
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [dict(payload)]
    return []


def select_rows(
    session: SupabaseSession,
    table: str,
    *,
    columns: str = "*",
    filters: dict[str, str] | None = None,
    order: str = "",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, str] = {"select": columns}
    params.update(filters or {})
    if order:
        params["order"] = order
    if limit is not None:
        params["limit"] = str(limit)
    response = httpx.get(
        _table_url(session, table),
        headers=_headers(session),
        params=params,
        timeout=30,
    )
    return _json(response, table)


def insert_row(
    session: SupabaseSession,
    table: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    response = httpx.post(
        _table_url(session, table),
        headers=_headers(session, "return=representation"),
        json=values,
        timeout=30,
    )
    rows = _json(response, table)
    return rows[0] if rows else {}


def update_rows(
    session: SupabaseSession,
    table: str,
    values: dict[str, Any],
    *,
    filters: dict[str, str],
) -> list[dict[str, Any]]:
    response = httpx.patch(
        _table_url(session, table),
        headers=_headers(session, "return=representation"),
        params=filters,
        json=values,
        timeout=30,
    )
    return _json(response, table)


def upsert_row(
    session: SupabaseSession,
    table: str,
    values: dict[str, Any],
    *,
    conflict_column: str,
) -> dict[str, Any]:
    response = httpx.post(
        _table_url(session, table),
        headers=_headers(
            session,
            "resolution=merge-duplicates,return=representation",
        ),
        params={"on_conflict": conflict_column},
        json=values,
        timeout=30,
    )
    rows = _json(response, table)
    return rows[0] if rows else {}
