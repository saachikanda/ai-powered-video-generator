# 🎬 AI Video Studio

A full-stack AI-powered video generator that turns a text prompt into a fully rendered `.mp4` file — complete with AI-generated images, professional voiceover, synced captions, and cinematic transitions.

---

## How It Works

```
Topic → AI Script → Scene Images → TTS Voiceover → Captions → MP4
```

1. You describe a video topic in the UI
2. The backend generates a multi-scene script using Gemini 2.0 Flash (or Pollinations AI as fallback)
3. Each scene gets AI-generated images via Pollinations (with Picsum/Unsplash as fallbacks)
4. Voiceover is synthesised using Edge TTS with word-level timestamp sync
5. MoviePy renders everything into a final `.mp4` with cinematic transitions

---

## Requirements

### Python packages

```bash
pip install flask flask-cors requests numpy pillow moviepy edge-tts google-genai
```

### System dependencies

- **FFmpeg** — required by MoviePy for encoding
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

### Optional

- A **Gemini API key** for higher-quality, faster script generation (falls back to Pollinations AI if not set)

---

## Setup & Running

**1. Clone or download the project files:**

```
project/
├── generator.py     ← Python backend
├── INDEX2.HTML      ← Frontend UI
└── voice.py         ← Gemini Live audio module (required)
```

**2. (Optional) Set your Gemini API key:**

```bash
# macOS / Linux
export GEMINI_API_KEY=your_key_here

# Windows
set GEMINI_API_KEY=your_key_here
```

**3. Start the backend:**

```bash
python generator.py
```

The server starts at `http://localhost:5000`. Poppins fonts are downloaded automatically on first run.

**4. Open the frontend:**

Open `INDEX2.HTML` in your browser, or navigate to `http://localhost:5000`.

---

## Using the UI

| Field | Description |
|---|---|
| **Describe your video** | The topic or product you want a video about |
| **Style** | Tone of the video (Promotional, Documentary, Cinematic, etc.) |
| **Voice language** | Language for the TTS voiceover |
| **Target duration** | 15 / 30 / 60 / 90 seconds |
| **Scenes** | Number of scenes to generate (3, 5, or 7) |

Click **Generate video** and watch the pipeline progress in real time. When complete, the video plays directly in the browser and a **Download .mp4** button appears.

---

## Architecture

### Backend (`generator.py`)

- **Flask** REST API with CORS enabled
- Background job queue — each generation runs in a daemon thread
- Progress streamed to the frontend via polling (`/api/status/<job_id>`)

### AI & Media Pipeline

| Stage | Tool | Fallback |
|---|---|---|
| Script & titles | Gemini 2.0 Flash | Pollinations AI (`openai` model) |
| Image generation | Pollinations AI (flux/turbo) | Picsum → Unsplash → Lorem Picsum |
| Voiceover | Gemini Live (via `voice.py`) | Edge TTS |
| Caption sync | Word-boundary timestamps | Linear interpolation |
| Video encoding | MoviePy + FFmpeg (libx264/aac) | — |

### Fonts

Poppins (Bold, Medium, Regular) are auto-downloaded from Google Fonts on first run and cached in your system's temp directory. Falls back to Liberation Sans or DejaVu if unavailable.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Server status, active AI backend, voice info |
| `POST` | `/api/generate` | Start a generation job |
| `GET` | `/api/status/<job_id>` | Poll job progress and logs |
| `GET` | `/api/preview/<job_id>` | Stream the finished video |
| `GET` | `/api/download/<job_id>` | Download the `.mp4` file |
| `GET` | `/api/voices` | List all available TTS voices |

### POST `/api/generate` — request body

```json
{
  "topic":    "Mac mini M4 — compact design and performance",
  "style":    "Promotional / Marketing",
  "lang":     "en",
  "duration": 30,
  "scenes":   5
}
```

---

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *(unset)* | Enables Gemini 2.0 Flash for script generation |
| `EDGE_TTS_VOICE` | *(auto)* | Override the TTS voice for all scenes |
| `PORT` | `5000` | Port the Flask server listens on |

### Video output settings

Hardcoded in `generator.py` — change these at the top of the file if needed:

```python
VIDEO_W, VIDEO_H, FPS = 1280, 720, 24
XFADE_S = 0.6   # cross-dissolve duration between scenes (seconds)
```

---

## Cinematic Transition Effects

Seven complex motion effects are rotated across scenes:

- `parallax` — multi-layer sinusoidal drift
- `pendulum` — arc swing with cosine-damped vertical motion
- `diagonal_drift` — smooth diagonal travel across the frame
- `rotation_zoom` — slow rotation with progressive zoom
- `elastic_pan` — ease-out-back overshoot pan
- `breathe` — pulsing zoom cycle
- `fade_zoom` — peak-zoom at mid-clip, symmetric

---

## Supported Languages / Voices

44 voices across 15+ languages including English (US, UK, AU, IN, ZA, NG), Spanish, French, German, Italian, Portuguese, Hindi, Arabic, Chinese, Japanese, Korean, Russian, and more. The full list is available via `/api/voices`.

---

## Known Limitations

- Image generation via Pollinations can be slow under load (up to 120s timeout per image)
- Gemini Live audio (`voice.py`) must be present and correctly configured; Edge TTS is the automatic fallback
- Long topics (over 80 characters) are automatically summarised to a shorter subject line before generation
- All jobs are held in memory — restarting the server clears all job history

---

## License

MIT — use freely, modify as needed.
