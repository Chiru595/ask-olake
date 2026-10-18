"""Rate limiting (per client IP behind a proxy) and admin-key auth."""
import hmac

from fastapi import Header, HTTPException, Request
from slowapi import Limiter

from app.config import get_settings


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=client_ip, headers_enabled=True)


def admin_ok(key: str | None) -> bool:
    expected = get_settings().admin_key
    return bool(expected) and bool(key) and hmac.compare_digest(key, expected)


async def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    if not admin_ok(x_admin_key):
        raise HTTPException(status_code=401, detail="invalid admin key")
