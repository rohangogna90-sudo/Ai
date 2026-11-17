# Personal AI Assistant (v0.2)

A private, single-owner Android AI assistant.

- Expo / React Native chat app (`mobile/`)
- FastAPI backend (`backend/`) that talks to Gemini, OpenAI or Anthropic
- Owner-password login, signed tokens, login rate limiting
- **Skills (tools)** the AI can use on its own, listed below
- Provider keys stay on the server and are never put in the APK

## Skills

| Skill | What it does | Needs a key? |
|---|---|---|
| `get_datetime` | Current date/time in any timezone | No |
| `calculator` | Safe exact math (never uses `eval`) | No |
| `get_weather` | Current weather + up to 7-day forecast (Open-Meteo) | No |
| `convert_currency` | Live ECB reference rates, about 30 currencies (Frankfurter) | No |
| `wikipedia_search` | Facts from Wikipedia, any language (`hi`, `pa`, ...) | No |
| `youtube_search` | YouTube search link, or real video results with a key | Optional `YOUTUBE_API_KEY` |
| `web_search` | Current news and web answers (Tavily) | `TAVILY_API_KEY` |
| `save_note` / `list_notes` / `delete_note` | Personal notes | No |
| `add_task` / `list_tasks` / `complete_task` | To-do list | No |

The assistant replies in the language you write in (English, Hindi, Punjabi, Hinglish...).
The app shows a small "Used: ..." label under answers that used a skill.

To add a skill, write one `async` function in `backend/tools.py` with the `@register(...)`
decorator. It is automatically available on Gemini, OpenAI and Anthropic.

## 1. Deploy the backend on Render

1. Put this project in a **private** GitHub repository.
2. In Render, create a **Web Service** (or use the `render.yaml` Blueprint).
3. Root Directory: `backend`. Build: `pip install -r requirements.txt`.
   Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Add environment variables:

| Variable | Value |
|---|---|
| `OWNER_PASSWORD` | A long private password only you know |
| `JWT_SECRET` | 32+ random characters (the Blueprint generates it for you) |
| `AI_PROVIDER` | `gemini` |
| `GEMINI_API_KEY` | Your Gemini API key (the **value**, not the name) |
| `GEMINI_MODEL` | `gemini-3.8-flash` (check Google's model list for newer ones) |
| `DEFAULT_TIMEZONE` | e.g. `Asia/Kolkata` |
| `TAVILY_API_KEY` | Optional, turns on `web_search` |
| `YOUTUBE_API_KEY` | Optional, real YouTube results |

Optional providers: `OPENAI_API_KEY`/`OPENAI_MODEL`, `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL`.

Test: open `https://YOUR-SERVICE.onrender.com/health`. It should return `{"ok":true,...}`.

### Render free plan notes
- The service **sleeps after about 15 minutes idle**. The first request can take up to a minute.
  The app pings `/health` at launch and waits up to 90 seconds.
- The disk is **not permanent**: notes and tasks are lost when the service restarts or redeploys.
  For permanent storage, add a paid Render Disk (set `DATA_DIR` to its mount path) or move to a database.
- Rate limits are kept in memory and reset on restart. That is fine for one owner.

## 2. Configure and build the Android app

1. Open `mobile/`, run `npm install`.
2. Put your Render URL (no trailing slash) in **`mobile/eas.json`** for the `preview` and
   `production` profiles (`EXPO_PUBLIC_API_URL`).
   `.env` is git-ignored, so Expo's cloud build does **not** see it. For `npx expo start`
   on your computer you can still copy `.env.example` to `.env`.
3. Build the APK:

```bash
npx eas-cli login
npx eas-cli build --platform android --profile preview
```

Open the download link on your phone and install the APK.

## 3. Run the tests (no internet needed)

```bash
cd backend
python -m unittest discover -s tests -v
```

## Security notes
- Never paste API keys into code, chat or GitHub. Keys only go in Render's Environment page.
- If a key was ever exposed, delete it and create a new one right away.
- Login is limited to 5 wrong tries per IP (20 total) per 15 minutes.
- Keep the GitHub repository private.

## Next ideas
1. Voice: speech-to-text and text-to-speech (English, Hindi, Punjabi first)
2. Save chat history on the phone
3. Reminders with push notifications
4. Read web pages / documents the user shares
5. Permission-based self-improvement (propose a patch, show a diff, apply only with approval)
