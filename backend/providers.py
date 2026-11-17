"""AI provider adapters (Gemini, OpenAI, Anthropic) with a shared tool-calling loop."""
import json
import logging
import os
from typing import Any, Optional

import tools

logger = logging.getLogger("personal-ai-assistant.providers")

MAX_TOOL_ROUNDS = 5
PROVIDERS = ("gemini", "openai", "anthropic")


class ProviderError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def build_system_prompt() -> str:
    return (
        "You are a helpful personal AI assistant. Be clear, practical, warm and honest.\n"
        "- Reply in the same language the user writes in. This includes Hindi, Punjabi and Hinglish "
        "(Hindi written in English letters). Keep a friendly, natural tone.\n"
        "- You have tools (skills). Use them whenever they help: the calculator for any arithmetic, "
        "get_weather for weather, convert_currency for exchange rates, wikipedia_search for facts, "
        "web_search for current news if available, notes and tasks when the user wants to save or "
        "track something. Never guess live data such as weather, prices or today's date.\n"
        "- Only say you did something if a tool actually did it. If a tool fails or no tool fits, "
        "say so plainly and offer what you can do instead.\n"
        "- Do not paste raw tool output or JSON. Summarize it in plain language.\n"
        "- Keep answers short unless the user asks for detail.\n"
        f"Current date and time: {tools.current_datetime_text()}."
    )


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
async def post_json(url: str, headers: dict, payload: dict, timeout: float = 60.0) -> tuple[int, Optional[dict]]:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException:
        raise ProviderError(504, "The AI provider took too long to answer. Please try again.")
    except httpx.HTTPError:
        raise ProviderError(502, "Could not reach the AI provider. Please try again.")
    try:
        data = response.json()
    except ValueError:
        data = None
    return response.status_code, data


def _raise_provider_error(name: str, status: int, data: Any) -> None:
    message = ""
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", ""))[:300]
        elif isinstance(error, str):
            message = error[:300]
    logger.error("%s returned HTTP %s: %s", name, status, message)
    if status in (401, 403) or "api key" in message.lower():
        detail = f"{name} rejected the API key. Check the key in the server settings."
    elif status == 429:
        detail = f"{name} rate limit or quota reached. Wait a moment and try again."
    elif status == 404:
        detail = f"{name} could not find that model. Check the model name in the server settings."
    elif status == 400:
        detail = f"{name} rejected the request. Check the model name and settings."
    else:
        detail = f"{name} could not complete the request (HTTP {status})."
    raise ProviderError(429 if status == 429 else 502, detail)


def _parse_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


# --------------------------------------------------------------------------- #
# Tool schema converters
# --------------------------------------------------------------------------- #
def _gemini_tools() -> list:
    declarations = []
    for t in tools.available_tools():
        declaration = {"name": t.name, "description": t.description}
        if t.properties:  # Gemini rejects an object schema with no properties
            declaration["parameters"] = {"type": "object", "properties": t.properties, "required": t.required}
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else []


def _openai_tools() -> list:
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": {"type": "object", "properties": t.properties, "required": t.required},
            "strict": False,
        }
        for t in tools.available_tools()
    ]


def _anthropic_tools() -> list:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": {"type": "object", "properties": t.properties, "required": t.required},
        }
        for t in tools.available_tools()
    ]


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
async def ask_gemini(messages: list[dict]) -> tuple[str, list[str]]:
    # BUG FIX: the original code passed the key itself to os.getenv(), so it never found it.
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ProviderError(503, "Gemini is not configured on the server. Set GEMINI_API_KEY.")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    used: list[str] = []
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        payload = {
            "systemInstruction": {"parts": [{"text": build_system_prompt()}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048},
        }
        if round_no < MAX_TOOL_ROUNDS and _gemini_tools():
            payload["tools"] = _gemini_tools()
        status, data = await post_json(url, {"x-goog-api-key": api_key}, payload)
        if status >= 400 or not isinstance(data, dict):
            _raise_provider_error("Gemini", status, data)
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            logger.error("Gemini returned no candidates (blockReason=%s)", reason)
            raise ProviderError(502, "Gemini did not return an answer. Try rephrasing your message.")
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        calls = [p["functionCall"] for p in parts if isinstance(p.get("functionCall"), dict)]
        if not calls:
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if not text:
                raise ProviderError(502, "Gemini returned an empty response.")
            return text, used
        # Send the model's turn back exactly as received (Gemini 3 needs its thought signatures).
        content.setdefault("role", "model")
        contents.append(content)
        response_parts = []
        for call in calls:
            name = call.get("name", "")
            result = await tools.run_tool(name, _parse_args(call.get("args")))
            used.append(name)
            function_response = {"name": name, "response": {"result": result}}
            if call.get("id"):
                function_response["id"] = call["id"]
            response_parts.append({"functionResponse": function_response})
        contents.append({"role": "user", "parts": response_parts})
    raise ProviderError(502, "Gemini kept calling tools without answering.")


# --------------------------------------------------------------------------- #
# OpenAI (Responses API)
# --------------------------------------------------------------------------- #
async def ask_openai(messages: list[dict]) -> tuple[str, list[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ProviderError(503, "OpenAI is not configured on the server. Set OPENAI_API_KEY.")
    model = os.getenv("OPENAI_MODEL", "gpt-6-astra")
    items: list[dict] = [{"role": m["role"], "content": m["content"]} for m in messages]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    used: list[str] = []
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        payload = {"model": model, "instructions": build_system_prompt(), "input": items, "max_output_tokens": 2048}
        if round_no < MAX_TOOL_ROUNDS and _openai_tools():
            payload["tools"] = _openai_tools()
        status, data = await post_json("https://api.openai.com/v1/responses", headers, payload)
        if status >= 400 or not isinstance(data, dict):
            _raise_provider_error("OpenAI", status, data)
        output = data.get("output") or []
        calls = [i for i in output if i.get("type") == "function_call"]
        if not calls:
            pieces = []
            for item in output:
                if item.get("type") == "message":
                    for block in item.get("content") or []:
                        if block.get("type") == "output_text":
                            pieces.append(block.get("text", ""))
            text = "".join(pieces).strip()
            if not text:
                raise ProviderError(502, "OpenAI returned an empty response.")
            return text, used
        items.extend(output)
        for call in calls:
            name = call.get("name", "")
            result = await tools.run_tool(name, _parse_args(call.get("arguments")))
            used.append(name)
            items.append({"type": "function_call_output", "call_id": call.get("call_id"), "output": tools.result_to_text(result)})
    raise ProviderError(502, "OpenAI kept calling tools without answering.")


# --------------------------------------------------------------------------- #
# Anthropic (Messages API)
# --------------------------------------------------------------------------- #
async def ask_anthropic(messages: list[dict]) -> tuple[str, list[str]]:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ProviderError(503, "Claude is not configured on the server. Set ANTHROPIC_API_KEY.")
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    convo: list[dict] = [{"role": m["role"], "content": m["content"]} for m in messages]
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    used: list[str] = []
    for round_no in range(MAX_TOOL_ROUNDS + 1):
        payload = {"model": model, "max_tokens": 2048, "system": build_system_prompt(), "messages": convo}
        if round_no < MAX_TOOL_ROUNDS and _anthropic_tools():
            payload["tools"] = _anthropic_tools()
        status, data = await post_json("https://api.anthropic.com/v1/messages", headers, payload)
        if status >= 400 or not isinstance(data, dict):
            _raise_provider_error("Claude", status, data)
        blocks = data.get("content") or []
        calls = [b for b in blocks if b.get("type") == "tool_use"]
        if not calls:
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            if not text:
                raise ProviderError(502, "Claude returned an empty response.")
            return text, used
        convo.append({"role": "assistant", "content": blocks})
        results = []
        for call in calls:
            name = call.get("name", "")
            result = await tools.run_tool(name, call.get("input"))
            used.append(name)
            results.append({"type": "tool_result", "tool_use_id": call.get("id"), "content": tools.result_to_text(result)})
        convo.append({"role": "user", "content": results})
    raise ProviderError(502, "Claude kept calling tools without answering.")


async def ask(provider: str, messages: list[dict]) -> tuple[str, list[str]]:
    if provider == "gemini":
        return await ask_gemini(messages)
    if provider == "openai":
        return await ask_openai(messages)
    if provider == "anthropic":
        return await ask_anthropic(messages)
    raise ProviderError(400, "Unsupported AI provider.")
