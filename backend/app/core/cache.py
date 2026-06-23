import functools
import hashlib
import json
from typing import Any, Callable, Dict

from redis.asyncio import Redis
from redis.exceptions import RedisError

from backend.app.core.config import get_settings


def serialize_response(data: Dict[str, Any]) -> str:
    return json.dumps(data, default=str)


def deserialize_response(data: str) -> Dict[str, Any]:
    return json.loads(data)


def _make_cache_key(func: Callable, args: tuple, kwargs: Dict[str, Any]) -> str:
    raw = f"{func.__module__}.{func.__qualname__}:{args}:{kwargs}"
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_response(ttl_seconds: int = 300):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = get_settings()
            redis = Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                decode_responses=True,
            )
            try:
                cache_key = _make_cache_key(func, args, kwargs)
                cached = await redis.get(cache_key)
                if cached is not None:
                    return deserialize_response(cached)
                result = await func(*args, **kwargs)
                await redis.setex(cache_key, ttl_seconds, serialize_response(result))
                return result
            except RedisError:
                return await func(*args, **kwargs)
            finally:
                await redis.aclose()
        return wrapper
    return decorator
