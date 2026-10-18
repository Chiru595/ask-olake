"""MCP server (streamable HTTP at /mcp) exposing the same pipeline as tools. Requires the admin key."""
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.security import admin_ok


def build_mcp(get_pipeline) -> MCPServer:
    mcp = MCPServer(name="ask-olake", instructions="Answers questions about OLake and OLake Fusion from the official docs, with source URLs.")

    @mcp.tool(description="Answer a question about OLake / OLake Fusion from the docs. Returns answer and source URLs.")
    async def ask_olake(question: str) -> dict:
        res = await get_pipeline().ask(question)
        return {"answer": res.answer, "sources": res.sources}

    @mcp.tool(description="Search the OLake docs and repos; returns the most relevant sections with URLs and scores.")
    async def search_docs(query: str) -> list[dict]:
        blocks, _, _, _ = await get_pipeline().retrieve(query, {})
        return [{"url": b.url, "title": b.title, "section": " > ".join(b.heading_path), "score": round(b.score, 4),
                 "text": b.text[:1500]} for b in blocks]

    return mcp


class MCPAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        auth = request.headers.get("authorization", "")
        key = request.headers.get("x-admin-key") or (auth[7:] if auth.lower().startswith("bearer ") else None)
        if not admin_ok(key):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def mcp_asgi(mcp: MCPServer):
    app = mcp.streamable_http_app(streamable_http_path="/", stateless_http=True, json_response=True,
                                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    app.add_middleware(MCPAuthMiddleware)
    return app
