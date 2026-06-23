from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from backend.app.api.v1.agent import router as agent_router
from backend.app.core.config import get_settings
from src.document_processing.process_documents import router as document_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="SmartMove AI Backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(agent_router, prefix="/api")
    app.include_router(document_router, prefix="/api/documents")

    @app.get("/")
    async def root() -> dict:
        return {"status": "ok", "service": "SmartMove AI"}

    @app.get("/robots.txt")
    async def robots() -> PlainTextResponse:
        return PlainTextResponse("User-agent: *\nDisallow:")

    return app


app = create_app()
