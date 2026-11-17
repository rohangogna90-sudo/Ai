"""Assistant skills (tools) the AI can call.

Every skill is a small async function registered with @register(). Providers (Gemini,
OpenAI, Anthropic) receive the same list, so adding a skill here adds it everywhere.

Skills that need no API key: date/time, calculator, weather, currency, Wikipedia,
YouTube links, notes, tasks.
Optional (only enabled when the key is set): web_search (TAVILY_API_KEY),
YouTube video results (YOUTUBE_API_KEY).
"""
import ast
import json
import logging
import math
import operator
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote, quote_plus
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger("personal-ai-assistant.tools")

USER_AGENT = "personal-ai-assistant/0.2 (private personal assistant)"


class ToolError(Exception):
    """A problem whose message is safe to show to the AI (and the user)."""


# --------------------------------------------------------------------------- #
# HTTP helpers (httpx imported lazily so the module can be tested offline)
# --------------------------------------------------------------------------- #
async def get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: float = 15.0) -> Any:
    import httpx

    merged = {"User-Agent": USER_AGENT}
    merged.update(headers or {})
    async with httpx.AsyncClient(timeout=timeout, headers=merged) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def post_json_tool(url: str, payload: dict, headers: Optional[dict] = None, timeout: float = 20.0) -> Any:
    import httpx

    merged = {"User-Agent": USER_AGENT}
    merged.update(headers or {})
    async with httpx.AsyncClient(timeout=timeout, headers=merged) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    name: str
    description: str
    properties: dict
    required: list
    func: Callable[..., Awaitable[dict]]
    enabled: Callable[[], bool]


_TOOLS: dict[str, Tool] = {}


def register(name: str, description: str, properties: dict, required: Optional[list] = None,
             enabled: Optional[Callable[[], bool]] = None):
    def decorator(func):
        _TOOLS[name] = Tool(name, description, properties, required or [], func, enabled or (lambda: True))
        return func

    return decorator


def available_tools() -> list[Tool]:
    return [t for t in _TOOLS.values() if t.enabled()]


def tool_names() -> list[str]:
    return [t.name for t in available_tools()]


def result_to_text(result: Any, limit: int = 8000) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    if len(text) > limit:
        text = json.dumps({"truncated_result": text[:limit]}, ensure_ascii=False)
    return text


async def run_tool(name: str, args: Any) -> dict:
    """Run one skill safely. Always returns a dict (errors become {"error": ...})."""
    tool = _TOOLS.get(name)
    if tool is None or not tool.enabled():
        return {"error": f"Unknown or unavailable tool: {name}"}
    if not isinstance(args, dict):
        args = {}
    try:
        result = await tool.func(**args)
        return result if isinstance(result, dict) else {"result": result}
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError:
        logger.warning("Bad arguments for tool %s: %s", name, list(args))
        return {"error": "Invalid arguments for this tool."}
    except Exception:
        logger.exception("Tool %s failed", name)
        return {"error": "The tool failed unexpectedly. Try again or answer without it."}


# --------------------------------------------------------------------------- #
# Date and time
# --------------------------------------------------------------------------- #
def _zone(name: Optional[str]) -> tuple[str, ZoneInfo]:
    name = (name or os.getenv("DEFAULT_TIMEZONE") or "UTC").strip()
    try:
        return name, ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise ToolError(f"Unknown timezone '{name}'. Use an IANA name such as Asia/Kolkata or America/New_York.")


def current_datetime_text() -> str:
    try:
        name, zone = _zone(None)
    except ToolError:
        name, zone = "UTC", timezone.utc
    return datetime.now(zone).strftime(f"%A, %d %B %Y, %I:%M %p ({name})")


@register(
    "get_datetime",
    "Get the current date and time, optionally in a specific timezone.",
    {"timezone": {"type": "string", "description": "IANA timezone name, e.g. Asia/Kolkata. Optional."}},
)
async def get_datetime(timezone: Optional[str] = None) -> dict:
    name, zone = _zone(timezone)
    now = datetime.now(zone)
    return {
        "timezone": name,
        "iso": now.isoformat(timespec="seconds"),
        "weekday": now.strftime("%A"),
        "date": now.strftime("%d %B %Y"),
        "time": now.strftime("%I:%M %p"),
    }


# --------------------------------------------------------------------------- #
# Calculator (safe: parses the expression, never uses eval)
# --------------------------------------------------------------------------- #
def _safe_pow(a, b):
    if abs(b) > 1000:
        raise ToolError("Exponent is too large.")
    if isinstance(a, int) and isinstance(b, int) and b > 0 and a.bit_length() * b > 10000:
        raise ToolError("Result is too large.")
    return operator.pow(a, b)


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_FUNCTIONS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "log10": math.log10, "log2": math.log2, "ln": math.log, "exp": math.exp,
    "abs": abs, "round": round, "min": min, "max": max, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("Only numbers are allowed in calculations.")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        args = [_eval_node(a) for a in node.args]
        if node.func.id == "factorial":
            if len(args) != 1 or float(args[0]) != int(args[0]) or not 0 <= args[0] <= 500:
                raise ToolError("factorial needs a whole number from 0 to 500.")
            args = [int(args[0])]
        return _FUNCTIONS[node.func.id](*args)
    raise ToolError("Unsupported expression. Use numbers, + - * / // % ** ( ) and common math functions.")


def _format_number(value):
    if isinstance(value, complex):
        raise ToolError("The result is not a real number.")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ToolError("The result is not a finite number.")
        if value.is_integer() and abs(value) < 1e15:
            return int(value)
        return float(f"{value:.12g}")
    return value


@register(
    "calculator",
    "Evaluate a math expression exactly. Supports + - * / // % ** ( ), pi, e, and functions "
    "like sqrt, sin, cos, log, abs, round, min, max, factorial. Use it for any arithmetic instead of guessing.",
    {"expression": {"type": "string", "description": "e.g. (1250 * 18) / 100 or sqrt(144) + 2**10"}},
    ["expression"],
)
async def calculator(expression: str) -> dict:
    expression = str(expression).strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if not expression or len(expression) > 200:
        raise ToolError("Give an expression of 1 to 200 characters.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolError("That is not a valid math expression.")
    try:
        value = _eval_node(tree)
    except ZeroDivisionError:
        raise ToolError("Division by zero.")
    except (ValueError, OverflowError, RecursionError, TypeError):
        raise ToolError("That calculation is not possible.")
    return {"expression": expression, "result": _format_number(value)}


# --------------------------------------------------------------------------- #
# Weather (Open-Meteo, no API key)
# --------------------------------------------------------------------------- #
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Light freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Light freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light rain showers", 81: "Rain showers", 82: "Violent rain showers",
    85: "Light snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}


async def _geocode(location: str) -> dict:
    location = str(location).strip()
    if not location:
        raise ToolError("Give a city or place name.")
    # The geocoder matches the place name only, so "Moga, Punjab" must be searched as "Moga"
    # and the rest ("Punjab") used to pick the right result.
    name, _, hint = location.partition(",")
    data = await get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": name.strip(), "count": 5, "language": "en", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        raise ToolError(f"Could not find a place called '{location}'.")
    hint = hint.strip().lower()
    if hint:
        for place in results:
            haystack = " ".join(str(place.get(k, "")) for k in ("admin1", "admin2", "country", "country_code")).lower()
            if hint in haystack:
                return place
    return results[0]


@register(
    "get_weather",
    "Get current weather and a short forecast for a city or place (temperatures in Celsius).",
    {
        "location": {"type": "string", "description": "City or place, e.g. 'Delhi' or 'Paris, France'."},
        "days": {"type": "integer", "description": "Forecast days from 1 to 7. Default 3."},
    },
    ["location"],
)
async def get_weather(location: str, days: int = 3) -> dict:
    try:
        days = max(1, min(int(days), 7))
    except (TypeError, ValueError):
        days = 3
    place = await _geocode(location)
    data = await get_json(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": days,
        },
    )
    current = data.get("current") or {}
    daily = data.get("daily") or {}
    forecast = []
    for i, day in enumerate(daily.get("time") or []):
        def pick(key):
            values = daily.get(key) or []
            return values[i] if i < len(values) else None

        forecast.append({
            "date": day,
            "conditions": WEATHER_CODES.get(pick("weather_code"), "Unknown"),
            "high_c": pick("temperature_2m_max"),
            "low_c": pick("temperature_2m_min"),
            "rain_chance_percent": pick("precipitation_probability_max"),
        })
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    return {
        "location": ", ".join(p for p in parts if p),
        "current": {
            "conditions": WEATHER_CODES.get(current.get("weather_code"), "Unknown"),
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_kmh": current.get("wind_speed_10m"),
            "precipitation_mm": current.get("precipitation"),
        },
        "forecast": forecast,
        "source": "Open-Meteo",
    }


# --------------------------------------------------------------------------- #
# Currency (Frankfurter / European Central Bank reference rates, no API key)
# --------------------------------------------------------------------------- #
@register(
    "convert_currency",
    "Convert an amount between currencies using the latest European Central Bank reference rates "
    "(about 30 major currencies including USD, EUR, INR, GBP, JPY).",
    {
        "amount": {"type": "number", "description": "Amount to convert."},
        "from_currency": {"type": "string", "description": "3-letter code, e.g. USD."},
        "to_currency": {"type": "string", "description": "3-letter code, e.g. INR."},
    },
    ["amount", "from_currency", "to_currency"],
)
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        raise ToolError("Amount must be a number.")
    source, target = str(from_currency).strip().upper(), str(to_currency).strip().upper()
    if not (re.fullmatch(r"[A-Z]{3}", source) and re.fullmatch(r"[A-Z]{3}", target)):
        raise ToolError("Currencies must be 3-letter codes such as USD or INR.")
    if source == target:
        return {"amount": amount, "from": source, "to": target, "rate": 1, "converted": amount}
    try:
        data = await get_json("https://api.frankfurter.dev/v1/latest", params={"base": source, "symbols": target})
        rate = data["rates"][target]
    except Exception:
        raise ToolError("Could not get that rate. One of the currencies may not be supported.")
    return {
        "amount": amount,
        "from": source,
        "to": target,
        "rate": rate,
        "converted": round(amount * rate, 4),
        "rate_date": data.get("date"),
        "source": "European Central Bank reference rates via Frankfurter",
    }


# --------------------------------------------------------------------------- #
# Wikipedia (no API key)
# --------------------------------------------------------------------------- #
@register(
    "wikipedia_search",
    "Look up facts about people, places, history, science and other topics on Wikipedia. "
    "Returns short intros of the best matches. Use language 'hi' for Hindi or 'pa' for Punjabi if needed.",
    {
        "query": {"type": "string", "description": "Topic to look up."},
        "language": {"type": "string", "description": "Wikipedia language code, default 'en'."},
    },
    ["query"],
)
async def wikipedia_search(query: str, language: str = "en") -> dict:
    language = str(language or "en").strip().lower()
    if not re.fullmatch(r"[a-z]{2,3}(-[a-z]{2,8})?", language):
        language = "en"
    query = str(query).strip()
    if not query:
        raise ToolError("Give something to look up.")
    data = await get_json(
        f"https://{language}.wikipedia.org/w/api.php",
        params={
            "action": "query", "format": "json", "generator": "search", "gsrsearch": query, "gsrlimit": 3,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": "max", "redirects": 1,
        },
    )
    pages = sorted(((data.get("query") or {}).get("pages") or {}).values(), key=lambda p: p.get("index", 99))
    results = []
    for page in pages:
        title = page.get("title", "")
        extract = (page.get("extract") or "").strip()
        if not title or not extract:
            continue
        results.append({
            "title": title,
            "summary": extract[:900],
            "url": f"https://{language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
        })
    if not results:
        raise ToolError(f"No Wikipedia article found for '{query}'.")
    return {"results": results, "source": "Wikipedia"}


# --------------------------------------------------------------------------- #
# Web search (optional: needs TAVILY_API_KEY, free tier available at tavily.com)
# --------------------------------------------------------------------------- #
@register(
    "web_search",
    "Search the web for current news and up-to-date information. Returns an answer plus sources.",
    {
        "query": {"type": "string", "description": "What to search for."},
        "max_results": {"type": "integer", "description": "Number of results, 1 to 8. Default 5."},
    },
    ["query"],
    enabled=lambda: bool(os.getenv("TAVILY_API_KEY")),
)
async def web_search(query: str, max_results: int = 5) -> dict:
    try:
        max_results = max(1, min(int(max_results), 8))
    except (TypeError, ValueError):
        max_results = 5
    try:
        data = await post_json_tool(
            "https://api.tavily.com/search",
            {"query": str(query), "max_results": max_results, "include_answer": True, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {os.getenv('TAVILY_API_KEY', '')}"},
        )
    except Exception:
        raise ToolError("Web search failed. Check TAVILY_API_KEY and quota on the server.")
    return {
        "answer": data.get("answer"),
        "results": [
            {"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:500]}
            for r in (data.get("results") or [])[:max_results]
        ],
        "source": "Tavily web search",
    }


# --------------------------------------------------------------------------- #
# YouTube (a search link always works; real results need YOUTUBE_API_KEY)
# --------------------------------------------------------------------------- #
@register(
    "youtube_search",
    "Find YouTube videos for a topic. Returns video links (or a YouTube search link).",
    {"query": {"type": "string", "description": "What to search for."}},
    ["query"],
)
async def youtube_search(query: str) -> dict:
    query = str(query).strip()
    if not query:
        raise ToolError("Give something to search for.")
    search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        return {"search_url": search_url, "note": "Open the link to see results."}
    try:
        data = await get_json(
            "https://www.googleapis.com/youtube/v3/search",
            params={"part": "snippet", "type": "video", "maxResults": 5, "q": query, "key": key},
        )
    except Exception:
        return {"search_url": search_url, "note": "YouTube API failed, so here is a search link instead."}
    videos = []
    for item in data.get("items") or []:
        video_id = (item.get("id") or {}).get("videoId")
        snippet = item.get("snippet") or {}
        if video_id:
            videos.append({
                "title": snippet.get("title"),
                "channel": snippet.get("channelTitle"),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            })
    return {"videos": videos, "search_url": search_url}


# --------------------------------------------------------------------------- #
# Notes and tasks (SQLite). On Render's free plan the disk resets on redeploy/restart.
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    due TEXT NOT NULL DEFAULT '',
    done INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


def _db() -> sqlite3.Connection:
    folder = Path(os.getenv("DATA_DIR", "./data"))
    folder.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(folder / "assistant.db")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _int_id(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ToolError("The id must be a number.")


@register(
    "save_note",
    "Save a note or fact the user wants to keep.",
    {
        "content": {"type": "string", "description": "The note text."},
        "title": {"type": "string", "description": "Short title. Optional."},
    },
    ["content"],
)
async def save_note(content: str, title: str = "") -> dict:
    content, title = str(content).strip(), str(title or "").strip()[:100]
    if not content:
        raise ToolError("The note is empty.")
    with closing(_db()) as conn:
        with conn:
            cursor = conn.execute(
                "INSERT INTO notes (title, content, created_at) VALUES (?, ?, ?)", (title, content[:2000], _now())
            )
    return {"saved": True, "id": cursor.lastrowid}


@register(
    "list_notes",
    "List the user's saved notes, newest first.",
    {"limit": {"type": "integer", "description": "How many notes to return, 1 to 50. Default 20."}},
)
async def list_notes(limit: int = 20) -> dict:
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 20
    with closing(_db()) as conn:
        rows = conn.execute(
            "SELECT id, title, content, created_at FROM notes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return {"notes": [dict(r) for r in rows]}


@register(
    "delete_note",
    "Delete a saved note by its id (use list_notes first to find the id).",
    {"id": {"type": "integer", "description": "Note id."}},
    ["id"],
)
async def delete_note(id: int) -> dict:
    with closing(_db()) as conn:
        with conn:
            cursor = conn.execute("DELETE FROM notes WHERE id = ?", (_int_id(id),))
    if cursor.rowcount == 0:
        raise ToolError("No note with that id.")
    return {"deleted": True}


@register(
    "add_task",
    "Add an item to the user's to-do list.",
    {
        "title": {"type": "string", "description": "What needs to be done."},
        "due": {"type": "string", "description": "Due date or time as text, e.g. 2026-11-20 or 'Friday evening'. Optional."},
    },
    ["title"],
)
async def add_task(title: str, due: str = "") -> dict:
    title, due = str(title).strip()[:300], str(due or "").strip()[:60]
    if not title:
        raise ToolError("The task is empty.")
    with closing(_db()) as conn:
        with conn:
            cursor = conn.execute("INSERT INTO tasks (title, due, created_at) VALUES (?, ?, ?)", (title, due, _now()))
    return {"added": True, "id": cursor.lastrowid}


@register(
    "list_tasks",
    "Show the user's to-do list. By default only unfinished tasks.",
    {"include_done": {"type": "boolean", "description": "Also show completed tasks. Default false."}},
)
async def list_tasks(include_done: bool = False) -> dict:
    query = "SELECT id, title, due, done FROM tasks"
    if not include_done:
        query += " WHERE done = 0"
    with closing(_db()) as conn:
        rows = conn.execute(query + " ORDER BY id").fetchall()
    return {"tasks": [{**dict(r), "done": bool(r["done"])} for r in rows]}


@register(
    "complete_task",
    "Mark a to-do item as done by its id (use list_tasks first to find the id).",
    {"id": {"type": "integer", "description": "Task id."}},
    ["id"],
)
async def complete_task(id: int) -> dict:
    with closing(_db()) as conn:
        with conn:
            cursor = conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (_int_id(id),))
    if cursor.rowcount == 0:
        raise ToolError("No task with that id.")
    return {"completed": True}
