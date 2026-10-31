"""Errors returned to clients, shaped like OpenAI's so existing SDKs surface them properly."""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param
        self.headers = headers

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }
