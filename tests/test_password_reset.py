from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from supabase_sync import (
    SupabaseSession,
    recovery_session_from_token,
    request_password_reset,
    update_password,
)


class PasswordResetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.url = "https://example.supabase.co"
        self.key = "publishable-key"

    @patch("supabase_sync.httpx.post")
    def test_request_password_reset_uses_recover_endpoint(self, post: Mock) -> None:
        post.return_value = Mock(status_code=200)

        request_password_reset(
            self.url,
            self.key,
            " user@example.com ",
            "https://coach.example/reset",
        )

        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], f"{self.url}/auth/v1/recover")
        self.assertEqual(kwargs["json"], {"email": "user@example.com"})
        self.assertEqual(
            kwargs["params"],
            {"redirect_to": "https://coach.example/reset"},
        )

    @patch("supabase_sync.httpx.get")
    def test_recovery_token_returns_authenticated_session(
        self, get: Mock
    ) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {"id": "user-1", "email": "user@example.com"}
        get.return_value = response

        session = recovery_session_from_token(
            self.url,
            self.key,
            "access-token",
            "refresh-token",
            2_000_000_000,
        )

        self.assertEqual(session.user_id, "user-1")
        self.assertEqual(session.access_token, "access-token")
        self.assertEqual(session.refresh_token, "refresh-token")
        self.assertEqual(get.call_args.args[0], f"{self.url}/auth/v1/user")
        self.assertEqual(
            get.call_args.kwargs["headers"]["Authorization"],
            "Bearer access-token",
        )

    @patch("supabase_sync.httpx.put")
    def test_update_password_uses_recovery_access_token(self, put: Mock) -> None:
        put.return_value = Mock(status_code=200)
        session = SupabaseSession(
            project_url=self.url,
            publishable_key=self.key,
            access_token="access-token",
            user_id="user-1",
            email="user@example.com",
        )

        update_password(session, "new-password")

        self.assertEqual(put.call_args.args[0], f"{self.url}/auth/v1/user")
        self.assertEqual(put.call_args.kwargs["json"], {"password": "new-password"})
        self.assertEqual(
            put.call_args.kwargs["headers"]["Authorization"],
            "Bearer access-token",
        )


if __name__ == "__main__":
    unittest.main()
