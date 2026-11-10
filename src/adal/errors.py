"""Every failure adaL means to report, each with a stable code a caller can branch on (the Glyphh server's
shape). Anything else is a crash: logged, and the caller sees only E_INTERNAL."""

from typing import Any, ClassVar


class AdalError(Exception):
    code: ClassVar[str] = "E_INTERNAL"
    status: ClassVar[int] = 500

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message: str = message
        self.detail: dict[str, Any] = detail


class Invalid(AdalError):
    code = "E_INVALID"
    status = 422


class Unauthorized(AdalError):
    code = "E_UNAUTHORIZED"
    status = 401


class NotFound(AdalError):
    code = "E_NOT_FOUND"
    status = 404


class LimitReached(AdalError):
    code = "E_LIMIT"
    status = 429
