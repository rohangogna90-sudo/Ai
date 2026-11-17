"""Offline tests: run with  python -m unittest discover -s tests  (from the backend folder)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import providers  # noqa: E402
import tools  # noqa: E402
from security import RateLimiter, verify_password  # noqa: E402


class CalculatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_basic_math(self):
        self.assertEqual((await tools.run_tool("calculator", {"expression": "(1250 * 18) / 100"}))["result"], 225)
        self.assertEqual((await tools.run_tool("calculator", {"expression": "2^10 + sqrt(144)"}))["result"], 1036)
        self.assertEqual((await tools.run_tool("calculator", {"expression": "10 / 4"}))["result"], 2.5)

    async def test_rejects_code_and_abuse(self):
        for bad in ["__import__('os').system('ls')", "open('x')", "9**9**9", "1/0", "(-8)**0.5", "'a'*5", "factorial(9999)"]:
            result = await tools.run_tool("calculator", {"expression": bad})
            self.assertIn("error", result, bad)


class SmallToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_datetime_and_bad_timezone(self):
        ok = await tools.run_tool("get_datetime", {"timezone": "Asia/Kolkata"})
        self.assertEqual(ok["timezone"], "Asia/Kolkata")
        self.assertIn("error", await tools.run_tool("get_datetime", {"timezone": "Mars/Base"}))

    async def test_unknown_tool_and_bad_args(self):
        self.assertIn("error", await tools.run_tool("nope", {}))
        self.assertIn("error", await tools.run_tool("calculator", {"wrong": 1}))

    async def test_notes_and_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            os.environ["DATA_DIR"] = folder
            saved = await tools.run_tool("save_note", {"content": "Buy milk", "title": "Shopping"})
            self.assertTrue(saved["saved"])
            notes = await tools.run_tool("list_notes", {})
            self.assertEqual(notes["notes"][0]["content"], "Buy milk")
            self.assertTrue((await tools.run_tool("delete_note", {"id": saved["id"]}))["deleted"])
            self.assertIn("error", await tools.run_tool("delete_note", {"id": 999}))
            task = await tools.run_tool("add_task", {"title": "Call bank", "due": "Friday"})
            self.assertEqual(len((await tools.run_tool("list_tasks", {}))["tasks"]), 1)
            await tools.run_tool("complete_task", {"id": task["id"]})
            self.assertEqual((await tools.run_tool("list_tasks", {}))["tasks"], [])
            self.assertEqual(len((await tools.run_tool("list_tasks", {"include_done": True}))["tasks"]), 1)
            os.environ.pop("DATA_DIR")

    async def test_weather_with_region_hint(self):
        calls = []

        async def fake_get_json(url, params=None, headers=None, timeout=15.0):
            calls.append((url, params))
            if "geocoding" in url:
                return {"results": [
                    {"name": "Springfield", "admin1": "Illinois", "country": "United States", "latitude": 1, "longitude": 2},
                    {"name": "Springfield", "admin1": "Missouri", "country": "United States", "latitude": 3, "longitude": 4},
                ]}
            return {
                "current": {"temperature_2m": 21.5, "weather_code": 2, "relative_humidity_2m": 40},
                "daily": {"time": ["2026-11-17"], "weather_code": [61], "temperature_2m_max": [24], "temperature_2m_min": [15], "precipitation_probability_max": [70]},
            }

        original, tools.get_json = tools.get_json, fake_get_json
        try:
            result = await tools.run_tool("get_weather", {"location": "Springfield, Missouri", "days": "2"})
        finally:
            tools.get_json = original
        self.assertEqual(calls[0][1]["name"], "Springfield")      # only the place name is searched
        self.assertEqual(calls[1][1]["latitude"], 3)               # region hint picked Missouri
        self.assertEqual(result["current"]["conditions"], "Partly cloudy")
        self.assertEqual(result["forecast"][0]["conditions"], "Light rain")

    async def test_currency(self):
        async def fake_get_json(url, params=None, headers=None, timeout=15.0):
            return {"date": "2026-11-16", "rates": {"INR": 90.0}}

        original, tools.get_json = tools.get_json, fake_get_json
        try:
            result = await tools.run_tool("convert_currency", {"amount": 10, "from_currency": "usd", "to_currency": "inr"})
            bad = await tools.run_tool("convert_currency", {"amount": 10, "from_currency": "dollar", "to_currency": "inr"})
        finally:
            tools.get_json = original
        self.assertEqual(result["converted"], 900.0)
        self.assertIn("error", bad)

    async def test_optional_tools_hidden_without_keys(self):
        os.environ.pop("TAVILY_API_KEY", None)
        self.assertNotIn("web_search", tools.tool_names())
        os.environ["TAVILY_API_KEY"] = "x"
        self.assertIn("web_search", tools.tool_names())
        os.environ.pop("TAVILY_API_KEY")


class ProviderLoopTests(unittest.IsolatedAsyncioTestCase):
    def script(self, responses):
        sent = []

        async def fake_post_json(url, headers, payload, timeout=60.0):
            sent.append(json.loads(json.dumps(payload)))
            return responses.pop(0)

        original, providers.post_json = providers.post_json, fake_post_json
        self.addCleanup(setattr, providers, "post_json", original)
        return sent

    async def test_gemini_tool_loop_keeps_signatures(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        model_turn = {"role": "model", "parts": [{"functionCall": {"name": "calculator", "args": {"expression": "2+2"}}, "thoughtSignature": "SIG"}]}
        sent = self.script([
            (200, {"candidates": [{"content": model_turn}]}),
            (200, {"candidates": [{"content": {"role": "model", "parts": [{"text": "It is 4."}]}}]}),
        ])
        reply, used = await providers.ask_gemini([{"role": "user", "content": "2+2?"}])
        self.assertEqual((reply, used), ("It is 4.", ["calculator"]))
        second = sent[1]["contents"]
        self.assertEqual(second[1]["parts"][0]["thoughtSignature"], "SIG")
        self.assertEqual(second[2]["parts"][0]["functionResponse"]["response"]["result"]["result"], 4)
        self.assertIn("tools", sent[0])

    async def test_gemini_missing_key_and_http_error(self):
        os.environ.pop("GEMINI_API_KEY", None)
        with self.assertRaises(providers.ProviderError) as ctx:
            await providers.ask_gemini([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.status, 503)
        os.environ["GEMINI_API_KEY"] = "test-key"
        self.script([(429, {"error": {"message": "quota"}})])
        with self.assertRaises(providers.ProviderError) as ctx:
            await providers.ask_gemini([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.status, 429)

    async def test_gemini_stops_runaway_tool_calls(self):
        os.environ["GEMINI_API_KEY"] = "test-key"
        call = {"role": "model", "parts": [{"functionCall": {"name": "calculator", "args": {"expression": "1+1"}}}]}
        final = {"role": "model", "parts": [{"text": "Done."}]}
        sent = self.script([(200, {"candidates": [{"content": dict(call)}]}) for _ in range(providers.MAX_TOOL_ROUNDS)]
                           + [(200, {"candidates": [{"content": final}]})])
        reply, _ = await providers.ask_gemini([{"role": "user", "content": "loop"}])
        self.assertEqual(reply, "Done.")
        self.assertNotIn("tools", sent[-1])  # last round forces a text answer

    async def test_openai_tool_loop(self):
        os.environ["OPENAI_API_KEY"] = "test-key"
        sent = self.script([
            (200, {"output": [{"type": "function_call", "call_id": "c1", "name": "calculator", "arguments": "{\"expression\": \"3*3\"}"}]}),
            (200, {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Nine."}]}]}),
        ])
        reply, used = await providers.ask_openai([{"role": "user", "content": "3*3?"}])
        self.assertEqual((reply, used), ("Nine.", ["calculator"]))
        last = sent[1]["input"][-1]
        self.assertEqual(last["type"], "function_call_output")
        self.assertEqual(json.loads(last["output"])["result"], 9)

    async def test_anthropic_tool_loop(self):
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        sent = self.script([
            (200, {"content": [{"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "5*5"}}]}),
            (200, {"content": [{"type": "text", "text": "25."}]}),
        ])
        reply, used = await providers.ask_anthropic([{"role": "user", "content": "5*5?"}])
        self.assertEqual((reply, used), ("25.", ["calculator"]))
        result_block = sent[1]["messages"][-1]["content"][0]
        self.assertEqual((result_block["type"], result_block["tool_use_id"]), ("tool_result", "t1"))

    def test_schemas_are_valid_shape(self):
        for t in providers._gemini_tools()[0]["functionDeclarations"]:
            if "parameters" in t:
                self.assertTrue(t["parameters"]["properties"])
        self.assertTrue(all(t["parameters"]["type"] == "object" for t in providers._openai_tools()))


class SecurityTests(unittest.TestCase):
    def test_password_check_handles_unicode(self):
        self.assertTrue(verify_password("पासवर्ड-123", "पासवर्ड-123"))
        self.assertFalse(verify_password("wrong", "पासवर्ड-123"))
        self.assertFalse(verify_password("anything", ""))

    def test_rate_limiter_window(self):
        now = [0.0]
        limiter = RateLimiter(limit=3, window_seconds=60, clock=lambda: now[0])
        for _ in range(3):
            self.assertIsNone(limiter.allow("a"))
        self.assertGreaterEqual(limiter.allow("a"), 1)
        self.assertIsNone(limiter.allow("b"))
        now[0] = 61
        self.assertIsNone(limiter.allow("a"))
        limiter.reset("a")
        self.assertIsNone(limiter.retry_after("a"))


if __name__ == "__main__":
    unittest.main()
