"""Synchronisation explicite de la base SQLite vers Supabase.

Le module utilise uniquement les API HTTPS de Supabase. La cle publiable peut
etre stockee localement, mais le mot de passe de connexion n'est jamais ecrit
sur disque. Les donnees ne sont transferees qu'apres une action explicite de
l'utilisateur dans l'assistant de migration.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx


TABLE_ORDER = (
    "meals",
    "weights",
    "symptom_events",
    "bowel_events",
    "activity_events",
    "favorite_foods",
    "profile",
)

BOOL_COLUMNS = {
    "meals": {"inside_eating_window"},
    "profile": {"auto_macro_targets"},
}

JSON_COLUMNS = {"meals": {"product_matches_json"}}


@dataclass(frozen=True)
class SupabaseSession:
    project_url: str
    publishable_key: str
    access_token: str
    user_id: str
    email: str
    refresh_token: str = ""
    expires_at: int = 0


class SupabaseSyncError(RuntimeError):
    """Erreur lisible provenant de l'authentification ou de la Data API."""


def session_storage_payload(session: SupabaseSession) -> dict[str, str]:
    """Prépare uniquement les éléments nécessaires pour renouveler une session."""
    return {
        "refresh_token": session.refresh_token,
        "user_id": session.user_id,
        "email": session.email,
    }


def session_from_storage_payload(
    project_url: str,
    publishable_key: str,
    payload: Mapping[str, Any],
) -> SupabaseSession:
    """Reconstruit une session expirée qui devra être renouvelée par Supabase."""
    refresh_token = str(payload.get("refresh_token") or "").strip()
    user_id = str(payload.get("user_id") or "").strip()
    email = str(payload.get("email") or "").strip()
    if not refresh_token or not user_id:
        raise ValueError("La session mémorisée est incomplète.")
    return SupabaseSession(
        project_url=_base_url(project_url),
        publishable_key=publishable_key.strip(),
        access_token="",
        user_id=user_id,
        email=email,
        refresh_token=refresh_token,
        expires_at=0,
    )


def _base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url.startswith("https://") or ".supabase.co" not in url:
        raise ValueError("L'URL doit ressembler à https://projet.supabase.co")
    return url


def _auth_headers(publishable_key: str, access_token: str | None = None) -> dict[str, str]:
    token = access_token or publishable_key
    return {
        "apikey": publishable_key.strip(),
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        return str(
            payload.get("msg")
            or payload.get("message")
            or payload.get("error_description")
            or payload.get("error")
            or payload
        )
    return str(payload)


def sign_up(project_url: str, publishable_key: str, email: str, password: str) -> str:
    """Cree le compte prive de l'application et renvoie un message utilisateur."""
    url = _base_url(project_url)
    response = httpx.post(
        f"{url}/auth/v1/signup",
        headers=_auth_headers(publishable_key),
        json={"email": email.strip(), "password": password},
        timeout=20,
    )
    if response.status_code >= 400:
        raise SupabaseSyncError(_error_detail(response))
    payload = response.json()
    if payload.get("session"):
        return "Compte créé et connecté."
    return "Compte créé. Confirmez maintenant l'e-mail reçu avant de vous connecter."


def sign_in(
    project_url: str,
    publishable_key: str,
    email: str,
    password: str,
) -> SupabaseSession:
    """Ouvre une session Supabase Auth avec un mot de passe non persiste."""
    url = _base_url(project_url)
    response = httpx.post(
        f"{url}/auth/v1/token?grant_type=password",
        headers=_auth_headers(publishable_key),
        json={"email": email.strip(), "password": password},
        timeout=20,
    )
    if response.status_code >= 400:
        raise SupabaseSyncError(_error_detail(response))
    payload = response.json()
    user = payload.get("user") or {}
    token = str(payload.get("access_token") or "")
    user_id = str(user.get("id") or "")
    if not token or not user_id:
        raise SupabaseSyncError("Supabase n'a pas renvoyé de session exploitable.")
    return SupabaseSession(
        project_url=url,
        publishable_key=publishable_key.strip(),
        access_token=token,
        user_id=user_id,
        email=str(user.get("email") or email.strip()),
        refresh_token=str(payload.get("refresh_token") or ""),
        expires_at=int(payload.get("expires_at") or (time.time() + 3600)),
    )


def request_password_reset(
    project_url: str,
    publishable_key: str,
    email: str,
    redirect_to: str,
) -> None:
    """Demande un lien de recuperation sans reveler l'existence du compte."""
    url = _base_url(project_url)
    response = httpx.post(
        f"{url}/auth/v1/recover",
        headers=_auth_headers(publishable_key),
        json={"email": email.strip()},
        params={"redirect_to": redirect_to.strip()},
        timeout=20,
    )
    if response.status_code >= 400:
        raise SupabaseSyncError(_error_detail(response))


def recovery_session_from_token(
    project_url: str,
    publishable_key: str,
    access_token: str,
    refresh_token: str = "",
    expires_at: int = 0,
) -> SupabaseSession:
    """Valide le jeton du lien Supabase et construit la session de recuperation."""
    url = _base_url(project_url)
    token = access_token.strip()
    response = httpx.get(
        f"{url}/auth/v1/user",
        headers=_auth_headers(publishable_key, token),
        timeout=20,
    )
    if response.status_code >= 400:
        raise SupabaseSyncError(_error_detail(response))
    user = response.json()
    user_id = str(user.get("id") or "")
    if not token or not user_id:
        raise SupabaseSyncError("Le lien n'a pas ouvert de session de récupération.")
    return SupabaseSession(
        project_url=url,
        publishable_key=publishable_key.strip(),
        access_token=token,
        user_id=user_id,
        email=str(user.get("email") or ""),
        refresh_token=refresh_token.strip(),
        expires_at=int(expires_at or (time.time() + 3600)),
    )


def update_password(session: SupabaseSession, new_password: str) -> None:
    """Remplace le mot de passe d'une session de recuperation authentifiee."""
    response = httpx.put(
        f"{session.project_url}/auth/v1/user",
        headers=_auth_headers(session.publishable_key, session.access_token),
        json={"password": new_password},
        timeout=20,
    )
    if response.status_code >= 400:
        raise SupabaseSyncError(_error_detail(response))


def refresh_session(session: SupabaseSession) -> SupabaseSession:
    """Renouvelle une session sans conserver ni redemander le mot de passe."""
    if not session.refresh_token:
        raise SupabaseSyncError("La session a expiré. Reconnectez-vous.")
    response = httpx.post(
        f"{session.project_url}/auth/v1/token?grant_type=refresh_token",
        headers=_auth_headers(session.publishable_key),
        json={"refresh_token": session.refresh_token},
        timeout=20,
    )
    if response.status_code >= 400:
        raise SupabaseSyncError(_error_detail(response))
    payload = response.json()
    user = payload.get("user") or {}
    token = str(payload.get("access_token") or "")
    if not token:
        raise SupabaseSyncError("Supabase n'a pas renouvelé la session.")
    return SupabaseSession(
        project_url=session.project_url,
        publishable_key=session.publishable_key,
        access_token=token,
        user_id=str(user.get("id") or session.user_id),
        email=str(user.get("email") or session.email),
        refresh_token=str(payload.get("refresh_token") or session.refresh_token),
        expires_at=int(payload.get("expires_at") or (time.time() + 3600)),
    )


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")]


def local_counts(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as connection:
        return {
            table: int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in TABLE_ORDER
        }


def _clean_row(table: str, row: dict[str, Any], user_id: str) -> dict[str, Any]:
    value = dict(row)
    value["user_id"] = user_id
    if table == "profile":
        value.pop("id", None)
    for column in BOOL_COLUMNS.get(table, set()):
        if value.get(column) is not None:
            value[column] = bool(value[column])
    for column in JSON_COLUMNS.get(table, set()):
        raw = value.get(column)
        if isinstance(raw, str):
            try:
                value[column] = json.loads(raw)
            except json.JSONDecodeError:
                value[column] = []
    if table == "meals" and not str(value.get("meal_occurred_at") or "").strip():
        value["meal_occurred_at"] = None
    return value


def iter_local_rows(db_path: Path, table: str) -> Iterable[dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        columns = _table_columns(connection, table)
        order = "id" if "id" in columns else "rowid"
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            yield dict(row)
    finally:
        connection.close()


def remote_counts(session: SupabaseSession) -> dict[str, int]:
    headers = _auth_headers(session.publishable_key, session.access_token)
    headers["Prefer"] = "count=exact"
    counts: dict[str, int] = {}
    for table in TABLE_ORDER:
        response = httpx.get(
            f"{session.project_url}/rest/v1/{table}",
            headers=headers,
            params={"select": "*", "limit": "1"},
            timeout=20,
        )
        if response.status_code >= 400:
            raise SupabaseSyncError(f"{table} : {_error_detail(response)}")
        content_range = response.headers.get("content-range", "*/0")
        try:
            counts[table] = int(content_range.rsplit("/", 1)[1])
        except (IndexError, ValueError):
            counts[table] = len(response.json())
    return counts


def _chunks(values: list[dict[str, Any]], size: int = 100) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def migrate_sqlite(
    db_path: Path,
    session: SupabaseSession,
) -> dict[str, int]:
    """Copie les lignes locales par upsert, sans modifier le fichier SQLite."""
    headers = _auth_headers(session.publishable_key, session.access_token)
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    migrated: dict[str, int] = {}
    for table in TABLE_ORDER:
        rows = [
            _clean_row(table, row, session.user_id)
            for row in iter_local_rows(db_path, table)
        ]
        conflict_column = "user_id" if table == "profile" else "id"
        for batch in _chunks(rows):
            response = httpx.post(
                f"{session.project_url}/rest/v1/{table}",
                headers=headers,
                params={"on_conflict": conflict_column},
                json=batch,
                timeout=45,
            )
            if response.status_code >= 400:
                raise SupabaseSyncError(f"{table} : {_error_detail(response)}")
        migrated[table] = len(rows)
    return migrated

