import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import providers
import tools
from security import RateLimiter, verify_password

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("personal-ai-assistant")

app = FastAPI(title="Personal AI Assistant API", version="0.2.0")

# Native mobile apps do not rely on browser CORS, but this also supports local web previews.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

OWNER_PASSWORD = os.getenv("OWNER_PASSWORD", "")
JWT_SECRET = os.getenv("JWT_SECRET", "")

if not OWNER_PASSWORD:
    logger.warning("OWNER_PASSWORD is not set: login is disabled until you set it.")
if len(JWT_SECRET) < 32:
    logger.warning("JWT_SECRET is missing or shorter than 32 characters: login is disabled until you fix it.")

# Brute-force protection: 5 wrong passwords per IP and 20 in total per 15 minutes.
login_ip_limiter = RateLimiter(limit=5, window_seconds=900)
login_global_limiter = RateLimiter(limit=20, window_seconds=900)
# Protects your AI quota: 20 chat messages per minute.
chat_limiter = RateLimiter(limit=20, window_seconds=60)


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=256)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=30)
    provider: Optional[Literal["gemini", "openai", "anthropic"]] = None


def issue_token() -> str:
    if len(JWT_SECRET) < 32:
        raise HTTPException(
            status_code=503,
            detail="Server authentication is not configured. Set JWT_SECRET (32+ characters).",
        )
    now = datetime.now(timezone.utc)
    payload = {"sub": "owner", "iat": now, "exp": now + timedelta(days=30)}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def require_owner(authorization: str = Header(default="")) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please sign in.")
    if len(JWT_SECRET) < 32:
        raise HTTPException(status_code=503, detail="Server authentication is not configured.")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Your session expired. Please sign in again.")
    if payload.get("sub") != "owner":
        raise HTTPException(status_code=403, detail="This assistant is private.")
    return "owner"


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def request_logging(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s -> %s (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
        return response
    except Exception:
        logger.exception("Unhandled request error on %s", request.url.path)
        raise


@app.get("/")
async def root():
    return {"ok": True, "service": "personal-ai-assistant-api", "docs": "Use /health to check status."}


@app.get("/health")
async def health():
    return {"ok": True, "service": "personal-ai-assistant-api"}


@app.post("/auth/login")
async def login(body: LoginRequest, request: Request):
    if not OWNER_PASSWORD:
        raise HTTPException(status_code=503, detail="Set OWNER_PASSWORD in the backend environment.")
    ip = client_ip(request)
    wait = login_ip_limiter.retry_after(ip) or login_global_limiter.retry_after("all")
    if wait:
        raise HTTPException(status_code=429, detail=f"Too many sign-in attempts. Try again in {wait} seconds.")
    if not verify_password(body.password, OWNER_PASSWORD):
        login_ip_limiter.hit(ip)
        login_global_limiter.hit("all")
        raise HTTPException(status_code=401, detail="Incorrect owner password.")
    login_ip_limiter.reset(ip)
    return {"access_token": issue_token(), "token_type": "bearer"}


@app.get("/providers")
async def list_providers(_: str = Depends(require_owner)):
    configured = {
        "gemini": bool(os.getenv("GEMINI_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
    }
    return {"default": os.getenv("AI_PROVIDER", "gemini").lower(), "configured": configured}


@app.get("/skills")
async def list_skills(_: str = Depends(require_owner)):
    return {"skills": [{"name": t.name, "description": t.description} for t in tools.available_tools()]}


@app.post("/chat")
async def chat(body: ChatRequest, _: str = Depends(require_owner)):
    provider = (body.provider or os.getenv("AI_PROVIDER", "gemini")).lower()
    if provider not in providers.PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported AI provider.")
    # Require the latest message to be from the user to prevent malformed chat turns.
    if body.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="The last chat message must be from the user.")
    wait = chat_limiter.allow("owner")
    if wait:
        raise HTTPException(status_code=429, detail=f"You are sending messages too fast. Try again in {wait} seconds.")
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    try:
        reply, skills_used = await providers.ask(provider, messages)
    except providers.ProviderError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail)
    return {"reply": reply, "provider": provider, "skills_used": sorted(set(skills_used))}
