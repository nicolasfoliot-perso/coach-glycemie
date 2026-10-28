from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from supabase_sync import (
    SupabaseSession,
    request_password_reset,
    update_password,
    verify_recovery_code,
)


class PasswordResetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.url = "https://example.supabase.co"
        self.key = "publishable-key"

    @patch("supabase_sync.httpx.post")
    def test_request_password_reset_uses_recover_endpoint(self, post: Mock) -> None:
        post.return_value = Mock(status_code=200)

        request_password_reset(self.url, self.key, " user@example.com ")

        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], f"{self.url}/auth/v1/recover")
        self.assertEqual(kwargs["json"], {"email": "user@example.com"})

    @patch("supabase_sync.httpx.post")
    def test_verify_recovery_code_returns_authenticated_session(
        self, post: Mock
    ) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_at": 2_000_000_000,
            "user": {"id": "user-1", "email": "user@example.com"},
        }
        post.return_value = response

        session = verify_recovery_code(
            self.url,
            self.key,
            "user@example.com",
            "123456",
        )

        self.assertEqual(session.user_id, "user-1")
        self.assertEqual(session.access_token, "access-token")
        self.assertEqual(
            post.call_args.kwargs["json"],
            {
                "email": "user@example.com",
                "token": "123456",
                "type": "recovery",
            },
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
