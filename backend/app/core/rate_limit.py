import time
from typing import Optional

from fastapi import HTTPException, Request, status
from redis.asyncio import Redis
from redis.exceptions import RedisError

from backend.app.core.config import get_settings


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def __call__(self, request: Request) -> None:
        settings = get_settings()
        client_ip = request.client.host if request.client else "unknown"
        redis = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            decode_responses=True,
        )
        try:
            key = f"ratelimit:{client_ip}"
            now = time.time()
            window_start = now - self.window_seconds
            await redis.zremrangebyscore(key, "-inf", window_start)
            count = await redis.zcard(key)
            if count >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Try again later.",
                )
            await redis.zadd(key, {str(now): now})
            await redis.expire(key, self.window_seconds)
        except RedisError:
            pass
        finally:
            await redis.aclose()
