from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from cloud_store import insert_row, select_rows, update_rows, upsert_row
from supabase_sync import SupabaseSession


def response(method: str, url: str, payload: object, status: int = 200) -> httpx.Response:
    request = httpx.Request(method, url)
    return httpx.Response(
        status,
        content=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        request=request,
    )


class CloudStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = SupabaseSession(
            project_url="https://example.supabase.co",
            publishable_key="sb_publishable_test",
            access_token="user-access-token",
            user_id="00000000-0000-0000-0000-000000000001",
            email="user@example.com",
        )

    @patch("cloud_store.httpx.get")
    def test_select_uses_user_token_and_query(self, mocked_get) -> None:
        mocked_get.return_value = response(
            "GET",
            "https://example.supabase.co/rest/v1/meals",
            [{"id": 7, "description": "Déjeuner"}],
        )
        rows = select_rows(
            self.session,
            "meals",
            columns="id,description",
            order="id.desc",
            limit=10,
        )
        self.assertEqual(rows[0]["id"], 7)
        kwargs = mocked_get.call_args.kwargs
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer user-access-token")
        self.assertEqual(kwargs["params"]["order"], "id.desc")

    @patch("cloud_store.httpx.post")
    def test_insert_returns_created_id(self, mocked_post) -> None:
        mocked_post.return_value = response(
            "POST",
            "https://example.supabase.co/rest/v1/weights",
            [{"id": 1000001, "weight_kg": 79.4}],
            201,
        )
        created = insert_row(self.session, "weights", {"weight_kg": 79.4})
        self.assertEqual(created["id"], 1000001)

    @patch("cloud_store.httpx.patch")
    def test_update_is_scoped_by_filter(self, mocked_patch) -> None:
        mocked_patch.return_value = response(
            "PATCH",
            "https://example.supabase.co/rest/v1/favorite_foods",
            [{"id": 4, "name": "Yaourt"}],
        )
        rows = update_rows(
            self.session,
            "favorite_foods",
            {"name": "Yaourt"},
            filters={"id": "eq.4"},
        )
        self.assertEqual(rows[0]["id"], 4)
        self.assertEqual(mocked_patch.call_args.kwargs["params"], {"id": "eq.4"})

    @patch("cloud_store.httpx.post")
    def test_profile_upsert_uses_user_id_conflict(self, mocked_post) -> None:
        mocked_post.return_value = response(
            "POST",
            "https://example.supabase.co/rest/v1/profile",
            [{"user_id": self.session.user_id}],
            201,
        )
        upsert_row(
            self.session,
            "profile",
            {"user_id": self.session.user_id, "display_name": "Nicolas"},
            conflict_column="user_id",
        )
        self.assertEqual(
            mocked_post.call_args.kwargs["params"],
            {"on_conflict": "user_id"},
        )


if __name__ == "__main__":
    unittest.main()
