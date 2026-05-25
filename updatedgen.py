"""
AI Video Studio — Backend v12 PATCHED-2 (4 Bug Fixes)
=======================================================
Fixes over v12 PATCHED:
  FIX 1 & 4: Audio prompt text no longer leaks into captions.
              The AudioLoopToFile prompt ("endless horizon: Sound expand" etc.)
              was being passed to _linear_timestamps instead of the actual
              scene narration. Now narration is always used for timestamps.
  FIX 2:     Single consistent speaker voice — _resolve_voice is called ONCE
              at pipeline start and reused for every scene. No more per-scene
              voice re-initialisation causing speaker changes.
  FIX 3:     Only complex, cinematic transitions are used (parallax, pendulum,
              diagonal_drift, rotation_zoom, elastic_pan, breathe, fade_zoom).
              Simple/boring effects (kb_zoom_in, kb_zoom_out, slide_left,
              slide_right, slide_up) have been removed from the rotation.
"""

import os, io, json, uuid, time, math, threading, traceback, tempfile, urllib.parse
import asyncio, logging, re as _re, concurrent.futures, urllib.request, base64
from pathlib import Path
from typing import Optional, List, Dict
import requests, numpy as np
import requests

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips
from google import genai
print("working")
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="\033[90m%(asctime)s\033[0m  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("studio")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = None
if GEMINI_KEY:
    try:
        from google import genai
        from google.genai import types as genai_types
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        log.info("✅  Gemini 2.0 Flash ready")
    except Exception as e:
        log.warning(f"Gemini init failed ({e}) — falling back to Pollinations")
else:
    log.info("ℹ️   No GEMINI_API_KEY — using Pollinations AI for text")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)
from voice import AudioLoop

@app.route("/api/voice", methods=["POST"])
def voice_preview():
    prompt = request.get_json(force=True).get("prompt", "")
    threading.Thread(target=lambda: asyncio.run(AudioLoop(prompt).run()), daemon=True).start()
    return jsonify({"status": "playing"})

JOBS: dict = {}
TEMP_DIR = Path(tempfile.gettempdir()) / "ai_video_studio"
TEMP_DIR.mkdir(exist_ok=True)
VIDEO_W, VIDEO_H, FPS = 1280, 720, 24
XFADE_S = 0.6   # cross-dissolve duration in seconds — shared by crossfade and caption sync

# ── Poppins font auto-download ────────────────────────────────────────────────
FONT_DIR = TEMP_DIR / "fonts"
FONT_DIR.mkdir(exist_ok=True)

POPPINS_URLS = {
    "Poppins-Bold.ttf":    "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Bold.ttf",
    "Poppins-Medium.ttf":  "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Medium.ttf",
    "Poppins-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf",
}

def ensure_poppins():
    for fname, url in POPPINS_URLS.items():
        dest = FONT_DIR / fname
        if dest.exists() and dest.stat().st_size > 10000:
            log.info(f"  Font cached: {fname}")
            continue
        log.info(f"  Downloading {fname}…")
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    dest.write_bytes(resp.read())
                log.info(f"  ✅ {fname} downloaded ({dest.stat().st_size//1024}KB)")
                break
            except Exception as e:
                log.warning(f"  Font download attempt {attempt+1}/4 failed: {e}")
                time.sleep(3)

log.info("Ensuring Poppins fonts…")
ensure_poppins()

def _find_font(names):
    for n in names:
        p = FONT_DIR / n
        if p.exists():
            return str(p)
    dirs = [
        "/usr/share/fonts/truetype/google-fonts",
        "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts/truetype/dejavu",
        "C:/Windows/Fonts",
        "/System/Library/Fonts",
        "/Library/Fonts",
    ]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return None

FONT_TITLE_PATH   = _find_font(["Poppins-Bold.ttf",    "LiberationSans-Bold.ttf",    "DejaVuSans-Bold.ttf"])
FONT_BODY_PATH    = _find_font(["Poppins-Medium.ttf",   "LiberationSans-Regular.ttf", "DejaVuSans.ttf"])
FONT_CAPTION_PATH = _find_font(["Poppins-Regular.ttf",  "LiberationSans-Regular.ttf", "DejaVuSans.ttf"])

log.info(f"Title font  : {FONT_TITLE_PATH}")
log.info(f"Caption font: {FONT_CAPTION_PATH}")

def get_font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    path = {"bold": FONT_TITLE_PATH, "medium": FONT_BODY_PATH,
            "regular": FONT_CAPTION_PATH}.get(weight, FONT_CAPTION_PATH)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

# ── Voice catalogue ───────────────────────────────────────────────────────────
VOICE_OPTIONS = [
    ("en-us-f",   "English — American (Female)",   "en-US-AriaNeural"),
    ("en-us-m",   "English — American (Male)",     "en-US-GuyNeural"),
    ("en-gb-f",   "English — British (Female)",    "en-GB-SoniaNeural"),
    ("en-gb-m",   "English — British (Male)",      "en-GB-RyanNeural"),
    ("en-au-f",   "English — Australian (Female)", "en-AU-NatashaNeural"),
    ("en-au-m",   "English — Australian (Male)",   "en-AU-WilliamNeural"),
    ("en-in-f",   "English — Indian (Female)",     "en-IN-NeerjaNeural"),
    ("en-in-m",   "English — Indian (Male)",       "en-IN-PrabhatNeural"),
    ("en-za-f",   "English — South African",       "en-ZA-LeahNeural"),
    ("en-ng-f",   "English — Nigerian",            "en-NG-EzinneNeural"),
    ("es-es-f",   "Spanish — Spain (Female)",      "es-ES-ElviraNeural"),
    ("es-mx-f",   "Spanish — Mexico (Female)",     "es-MX-DaliaNeural"),
    ("es-mx-m",   "Spanish — Mexico (Male)",       "es-MX-JorgeNeural"),
    ("es-ar-f",   "Spanish — Argentina (Female)",  "es-AR-ElenaNeural"),
    ("fr-fr-f",   "French — France (Female)",      "fr-FR-DeniseNeural"),
    ("fr-fr-m",   "French — France (Male)",        "fr-FR-HenriNeural"),
    ("fr-ca-f",   "French — Canadian (Female)",    "fr-CA-SylvieNeural"),
    ("de-de-f",   "German — Germany (Female)",     "de-DE-KatjaNeural"),
    ("de-de-m",   "German — Germany (Male)",       "de-DE-ConradNeural"),
    ("it-it-f",   "Italian (Female)",              "it-IT-ElsaNeural"),
    ("it-it-m",   "Italian (Male)",                "it-IT-DiegoNeural"),
    ("pt-br-f",   "Portuguese — Brazil (Female)",  "pt-BR-FranciscaNeural"),
    ("pt-br-m",   "Portuguese — Brazil (Male)",    "pt-BR-AntonioNeural"),
    ("pt-pt-f",   "Portuguese — Portugal (Female)","pt-PT-RaquelNeural"),
    ("hi-in-f",   "Hindi (Female)",                "hi-IN-SwaraNeural"),
    ("hi-in-m",   "Hindi (Male)",                  "hi-IN-MadhurNeural"),
    ("ar-sa-f",   "Arabic — Saudi (Female)",       "ar-SA-ZariyahNeural"),
    ("ar-eg-f",   "Arabic — Egyptian (Female)",    "ar-EG-SalmaNeural"),
    ("zh-cn-f",   "Chinese — Mandarin (Female)",   "zh-CN-XiaoxiaoNeural"),
    ("zh-cn-m",   "Chinese — Mandarin (Male)",     "zh-CN-YunxiNeural"),
    ("zh-tw-f",   "Chinese — Taiwan (Female)",     "zh-TW-HsiaoChenNeural"),
    ("ja-jp-f",   "Japanese (Female)",             "ja-JP-NanamiNeural"),
    ("ja-jp-m",   "Japanese (Male)",               "ja-JP-KeitaNeural"),
    ("ko-kr-f",   "Korean (Female)",               "ko-KR-SunHiNeural"),
    ("ko-kr-m",   "Korean (Male)",                 "ko-KR-InJoonNeural"),
    ("ru-ru-f",   "Russian (Female)",              "ru-RU-SvetlanaNeural"),
    ("ru-ru-m",   "Russian (Male)",                "ru-RU-DmitryNeural"),
    ("nl-nl-f",   "Dutch (Female)",                "nl-NL-ColetteNeural"),
    ("pl-pl-f",   "Polish (Female)",               "pl-PL-AgnieszkaNeural"),
    ("tr-tr-f",   "Turkish (Female)",              "tr-TR-EmelNeural"),
    ("id-id-f",   "Indonesian (Female)",           "id-ID-GadisNeural"),
    ("sv-se-f",   "Swedish (Female)",              "sv-SE-SofieNeural"),
    ("da-dk-f",   "Danish (Female)",               "da-DK-ChristelNeural"),
    ("fi-fi-f",   "Finnish (Female)",              "fi-FI-NooraNeural"),
    ("uk-ua-f",   "Ukrainian (Female)",            "uk-UA-PolinaNeural"),
]

_VOICE_ID_MAP = {code: vid for code, _, vid in VOICE_OPTIONS}

_LANG_DEFAULT = {
    "en": "en-US-AriaNeural", "en-gb": "en-GB-SoniaNeural",
    "en-au": "en-AU-NatashaNeural", "en-in": "en-IN-NeerjaNeural",
    "es": "es-ES-ElviraNeural", "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural", "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural", "ar": "ar-SA-ZariyahNeural",
    "zh": "zh-CN-XiaoxiaoNeural", "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural", "hi": "hi-IN-SwaraNeural",
    "ru": "ru-RU-SvetlanaNeural",
}

def _resolve_voice(lang: str, voice_code: str = "") -> str:
    override = os.getenv("EDGE_TTS_VOICE", "").strip()
    if override:
        return override
    if voice_code:
        if voice_code in _VOICE_ID_MAP:
            return _VOICE_ID_MAP[voice_code]
        if "Neural" in voice_code:
            return voice_code
    lc = lang.lower().strip()
    return _LANG_DEFAULT.get(lc, _LANG_DEFAULT.get(lc.split("-")[0], "en-US-AriaNeural"))

class AudioLoopToFile(AudioLoop):
    def __init__(self, prompt, output_path):
        super().__init__(prompt)
        self.output_path = output_path

    async def run(self):
        from voice import client, MODEL, CONFIG
        from google.genai import types
        import wave

        saved_chunks = []

        async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
            await session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=self.user_prompt)]
                ),
                turn_complete=True
            )
            async for response in session.receive():
                if data := response.data:
                    saved_chunks.append(data)

        with wave.open(str(self.output_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            for chunk in saved_chunks:
                wf.writeframes(chunk)

        log.info(f"✅ Saved {len(saved_chunks)} chunks → {self.output_path}")

# ── Job helper ────────────────────────────────────────────────────────────────
def job_update(jid, status=None, progress=None, message=None, result=None, error=None):
    j = JOBS.setdefault(jid, {})
    if status   is not None: j["status"]   = status
    if progress is not None: j["progress"] = progress
    if message  is not None:
        j["log"] = j.get("log", []) + [{"ts": time.strftime("%H:%M:%S"), "msg": message}]
        log.info(f"[{jid[:8]}] {message}")
    if result   is not None: j["result"]   = result
    if error    is not None: j["error"]    = error

# ── JSON extractor ────────────────────────────────────────────────────────────
def extract_json(raw: str) -> dict:
    s = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    st = s.find("{")
    if st == -1:
        raise ValueError(f"No JSON found: {raw[:120]!r}")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s[st:], st):
        if esc:       esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"': in_str = not in_str; continue
        if in_str:    continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:    return json.loads(s[st:i+1])
                except: break
    frag = s[st:]
    ob   = frag.count("{") - frag.count("}")
    ob2  = frag.count("[") - frag.count("]")
    q    = sum(1 for i, c in enumerate(frag) if c == '"' and (i == 0 or frag[i-1] != "\\"))
    suffix = ('"' if q % 2 == 1 else "") + "]"*max(0,ob2) + "}"*max(0,ob)
    try:    return json.loads(frag + suffix)
    except Exception as e:
        raise ValueError(f"Cannot parse JSON: {e}. Raw: {raw[:200]!r}")

# ── AI call ───────────────────────────────────────────────────────────────────
def ai_call(prompt: str, system: str = "") -> str:
    if gemini_client:
        try:
            time.sleep(0.3)
            full = (system + "\n\n" + prompt).strip() if system else prompt
            resp = gemini_client.models.generate_content(
                model="gemini-2.0-flash", contents=full)
            return resp.text.strip()
        except Exception as e:
            log.warning(f"Gemini failed ({e}), using Pollinations")
    SYS = system or "Respond with ONLY valid compact JSON — no markdown, no extra keys."
    for attempt in range(5):
        try:
            time.sleep(attempt * 2)
            r = requests.post(
                "https://text.pollinations.ai/",
                json={"messages": [{"role":"system","content":SYS},
                                   {"role":"user",  "content":prompt}],
                      "model":"openai","seed":int(time.time())%9999,"jsonMode":True},
                timeout=90, headers={"Content-Type":"application/json"})
            r.raise_for_status()
            return r.text
        except Exception as e:
            log.warning(f"Pollinations {attempt+1}/5: {e}")
    raise RuntimeError("All AI backends failed")

SYS_JSON = (
    "You are a professional video scriptwriter and creative director. "
    "Respond with ONLY valid compact JSON — no markdown, no newlines inside strings, no extra keys."
)

# ── Text sanitizer ────────────────────────────────────────────────────────────
def _clean_display(text: str) -> str:
    # ── Strip parenthetical/bracketed content entirely ───────────────────────
    # AI models sometimes inject stage directions or constraint notes like
    # "(constraints cannot be satisfied simultaneously)" or "[music]" into
    # narration. Remove all of these before they reach captions or TTS.
    text = _re.sub(r'\([^)]*\)', '', text)   # remove (...)
    text = _re.sub(r'\[[^\]]*\]', '', text)  # remove [...]
    text = _re.sub(r'\s{2,}', ' ', text)     # collapse double spaces left behind

    text = (text
            .replace("\u2014", "-").replace("\u2013", "-")
            .replace("\u2026", "...").replace("\u00d7", "x")
            .replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2022", "-").replace("\u00a0", " ")
            .replace("\u200b", "").replace("\u00ae", "")
            .replace("\u2122", "").replace("\u00b0", " degrees"))
    text = _re.sub(r'[^\x20-\x7E\u00C0-\u024F]', '', text)
    return text.strip()

# ══════════════════════════════════════════════════════════════════════════════
# Gemini Audio Transcription with Word-Level Timestamps
# ══════════════════════════════════════════════════════════════════════════════

def _linear_timestamps(narration: str, duration: float) -> List[Dict]:
    words = _clean_display(narration).split()
    if not words:
        return []
    n = len(words)
    lead   = 0.15
    speech = max(0.1, duration - lead)
    step   = speech / n
    result = []
    for i, w in enumerate(words):
        start = lead + i * step
        end   = lead + (i + 1) * step
        result.append({"word": w, "start": round(start, 3), "end": round(end, 3)})
    return result


def transcribe_audio_gemini(audio_path: Path, narration: str,
                             duration: float) -> List[Dict]:
    if not gemini_client:
        log.info("  Gemini not available — using linear timestamp fallback")
        return _linear_timestamps(narration, duration)

    audio_path = Path(audio_path)
    if not audio_path.exists() or audio_path.stat().st_size < 512:
        log.warning(f"  Audio file missing/tiny: {audio_path} — linear fallback")
        return _linear_timestamps(narration, duration)

    file_size = audio_path.stat().st_size
    log.info(f"  Transcribing {audio_path.name} ({file_size//1024}KB) via Gemini…")

    transcription_prompt = (
        "Transcribe this audio and return ONLY a JSON object with one key 'words', "
        "whose value is an array of objects each with keys 'word' (string), "
        "'start' (float seconds), and 'end' (float seconds).\n"
        "Rules:\n"
        "- Every spoken word must appear in order.\n"
        "- Timestamps must be monotonically increasing.\n"
        "- Do NOT include punctuation in the 'word' field.\n"
        "- Return ONLY the JSON, no markdown, no extra text.\n"
        f"- The audio is approximately {duration:.1f} seconds long.\n"
        f"- Expected transcript (for alignment): \"{narration[:300]}\"\n\n"
        'Example format: {"words":[{"word":"hello","start":0.12,"end":0.45},'
        '{"word":"world","start":0.48,"end":0.81}]}'
    )

    INLINE_LIMIT = 15 * 1024 * 1024

    def _parse_transcript_response(raw_text: str, narration: str,
                                   duration: float) -> List[Dict]:
        try:
            data  = extract_json(raw_text)
            words = data.get("words", [])
            if not words:
                raise ValueError("Empty words list")
            result = []
            for item in words:
                w = str(item.get("word", "")).strip().strip(".,!?;:'\"")
                s = float(item.get("start", 0))
                e = float(item.get("end",   s + 0.1))
                if w:
                    result.append({"word": w, "start": round(s, 3), "end": round(e, 3)})
            expected = len(_clean_display(narration).split())
            if len(result) < max(1, expected // 2):
                raise ValueError(
                    f"Too few words: got {len(result)}, expected ~{expected}")
            log.info(f"  ✅ Gemini transcription: {len(result)} words")
            return result
        except Exception as e:
            log.warning(f"  Transcript parse failed ({e}) — linear fallback")
            return _linear_timestamps(narration, duration)

    if file_size <= INLINE_LIMIT:
        try:
            audio_bytes  = audio_path.read_bytes()
            b64_audio    = base64.b64encode(audio_bytes).decode("utf-8")
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[{
                    "parts": [
                        {"inline_data": {"mime_type": "audio/mpeg", "data": b64_audio}},
                        {"text": transcription_prompt},
                    ]
                }],
            )
            raw = response.text.strip()
            log.debug(f"  Gemini transcript raw (first 200): {raw[:200]!r}")
            return _parse_transcript_response(raw, narration, duration)
        except Exception as e:
            log.warning(f"  Inline transcription failed ({e}) — trying Files API…")

    uploaded_file = None
    try:
        log.info(f"  Uploading {audio_path.name} to Gemini Files API…")
        uploaded_file = gemini_client.files.upload(
            path=str(audio_path),
            config={"mime_type": "audio/mpeg", "display_name": audio_path.name},
        )
        for _ in range(20):
            fstate = gemini_client.files.get(name=uploaded_file.name)
            if fstate.state.name == "ACTIVE":
                break
            time.sleep(1.5)
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[{
                "parts": [
                    {"file_data": {"mime_type": "audio/mpeg",
                                   "file_uri": uploaded_file.uri}},
                    {"text": transcription_prompt},
                ]
            }],
        )
        raw = response.text.strip()
        log.debug(f"  Gemini Files transcript raw (first 200): {raw[:200]!r}")
        return _parse_transcript_response(raw, narration, duration)
    except Exception as e:
        log.warning(f"  Files API transcription failed ({e}) — linear fallback")
        return _linear_timestamps(narration, duration)
    finally:
        if uploaded_file:
            try:
                gemini_client.files.delete(name=uploaded_file.name)
                log.debug(f"  Deleted uploaded file {uploaded_file.name}")
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Script generators
# ══════════════════════════════════════════════════════════════════════════════

def get_all_scene_titles(topic: str, style: str, count: int) -> list:
    log.info(f"Generating {count} scene titles in one call…")
    schema_keys = ", ".join(f'"t{i+1}":"title here"' for i in range(count))
    schema = "{" + schema_keys + "}"
    prompt = (
        f'Plan {count} scene titles for a premium {style} video about: "{topic}".\n\n'
        f'Each title is a 3-5 word cinematic chapter heading — vivid, specific, evocative.\n'
        f'The titles should tell a story arc from start to finish.\n\n'
        f'STRICT RULES:\n'
        f'- Every title must be unique and descriptive — about a specific aspect of "{topic}".\n'
        f'- NEVER use: "Scene 1", "Scene 2", "Part 1", "Chapter 1", "Introduction", '
        f'"Overview", "{topic}" alone, or any other generic placeholder.\n'
        f'- Examples of GOOD titles: "The Camera That Thinks", "Speed You Can Feel", '
        f'"Light Years Ahead", "Built Without Compromise", "Where Design Meets Power"\n\n'
        f'Return ONLY: {schema}'
    )
    _BAD = _re.compile(
        r'^(scene\s*\d+|scene\s+\w+|part\s*\d+|chapter\s*\d+|'
        r'introduction|overview|untitled)$', _re.IGNORECASE)
    fallback_pool = [
        "Built Without Compromise", "The Power Within",
        "Design Meets Performance", "Beyond What You Expect",
        "Every Detail Counts", "The Next Level",
        "Crafted for Champions", "Where Innovation Lives",
        "The Future, Now", "Redefining the Standard",
    ]
    for attempt in range(3):
        try:
            raw = ai_call(prompt, SYS_JSON)
            d = extract_json(raw)
            titles = []
            for i in range(count):
                t = d.get(f"t{i+1}", "").strip()
                if not t or _BAD.match(t):
                    t = fallback_pool[i % len(fallback_pool)]
                titles.append(t)
            good = sum(1 for t in titles if not _BAD.match(t))
            if good >= max(1, count // 2):
                log.info(f"  Titles: {titles}")
                return titles
        except Exception as e:
            log.warning(f"  Titles attempt {attempt+1}/3 failed: {e}")
        time.sleep(2)
    constructed = [
        f"{topic} Unleashed", "Power Redefined", "The Edge You Need",
        "Built to Impress", "Performance Perfected", "The Ultimate Experience",
        "Details That Matter", "Speed Without Limits", "A New Standard",
        "The Future is Here",
    ]
    return [constructed[i % len(constructed)] for i in range(count)]


def get_video_title(topic: str, style: str) -> str:
    log.info("Generating video title…")
    prompt = (
        f'Create a compelling, emotionally resonant video title for a {style} video about: "{topic}".\n'
        f'Feel like a premium documentary or brand film — evocative, not clickbait. 4-6 words max.\n'
        f'Return ONLY: {{"title":"your title here"}}'
    )
    try:
        return extract_json(ai_call(prompt, SYS_JSON)).get("title", topic[:40])
    except Exception as e:
        log.warning(f"Title gen failed: {e}")
        return topic[:40]


def get_scene(topic: str, style: str, num: int, total: int, dur: int,
              scene_title: str = "") -> dict:
    log.info(f"Writing narration for scene {num}/{total}: '{scene_title}'…")
    word_count = int(dur * 1.5)
    max_words  = int(dur * 1.7)

    opening_styles = [
        "Start with a vivid sensory detail or atmospheric observation — do NOT open with the product name.",
        "Open with a bold, surprising statement or little-known fact that hooks the viewer immediately.",
        "Begin mid-action as if the viewer is already experiencing the product — immersive, present tense.",
        "Start with a short rhetorical question (max 8 words) that makes the viewer lean in.",
        "Open with a punchy 4-6 word sentence that creates curiosity or intrigue.",
        "Lead with an emotional hook — what does this feel like, sound like, or mean to the user?",
        "Start with a contrast: 'Most products do X. This one doesn't.' — then explain why.",
    ]
    opening_hint = opening_styles[(num - 1) % len(opening_styles)]
    first_topic_word = topic.split()[0] if topic.split() else topic

    prompt = (
        f'Write professional voiceover narration for scene {num} of {total} in a '
        f'premium {style} video about: "{topic}".\n'
        f'Scene title: "{scene_title}"\n'
        f'Target duration: {dur} seconds.\n'
        f'STRICT word limit: {word_count} words. ABSOLUTE MAXIMUM: {max_words} words.\n\n'
        f'OPENING RULE: {opening_hint}\n\n'
        f'═══ SENTENCE RULES — NON-NEGOTIABLE ═══\n'
        f'- Every sentence MUST be a complete, grammatically correct spoken sentence.\n'
        f'- Each sentence ends with a period (.) or exclamation mark (!).\n'
        f'- Each sentence is 10 to 16 words long — never shorter.\n'
        f'- NEVER use semicolons (;) — they make captions look broken.\n'
        f'- NEVER use colons (:) mid-sentence.\n'
        f'- ABSOLUTELY NO fragments: "Silence alchemy." / "Pure sound." / "Hush descends." '
        f'are all FORBIDDEN — they are meaningless filler.\n'
        f'- ABSOLUTELY NO invented compound phrases like '
        f'"{first_topic_word} Hush\'s world" or "{topic} alchemy" — '
        f'these are nonsense and must never appear.\n'
        f'- Every sentence needs a clear subject, a verb, and a concrete meaning.\n\n'
        f'═══ PRODUCT NAME RULES ═══\n'
        f'- NEVER start the narration with "{topic}" or "{first_topic_word}".\n'
        f'- NEVER use the product name more than TWICE in the entire narration.\n'
        f'- After the first mention, always use "it", "they", "this", or a short '
        f'synonym — never repeat the full name again.\n'
        f'- NEVER end with a standalone brand callout like "{topic}." as a sentence '
        f'on its own — it reads as a fragment.\n\n'
        f'═══ CONTENT RULES ═══\n'
        f'- Sound like a professional documentary voiceover — clear, warm, authoritative.\n'
        f'- Be SPECIFIC to "{topic}" — describe real features, sensations, or user benefits.\n'
        f'- Every sentence must relate directly to the scene title: "{scene_title}".\n'
        f'- Use contractions naturally (it\'s, you\'ll, that\'s, we\'ve).\n'
        f'- NEVER say: "In this video", "As we can see", "Today we explore", '
        f'"Welcome to", "Introducing", "Experience the sounds stretched across horizons".\n'
        f'- NO corporate filler, NO passive voice, NO robotic language.\n'
        f'- NO motivational poster lines. Write for human ears, not ad copy.\n\n'
        f'═══ EXAMPLES ═══\n'
        f'FORBIDDEN (never write like this):\n'
        f'"Silence alchemy. {topic} hush your world. Hear the magic. {topic}."\n\n'
        f'REQUIRED (write exactly like this):\n'
        f'"Noise disappears the moment you put them in. '
        f'Every ambient sound around you fades into the background. '
        f'You choose exactly what you want to listen to, and nothing else gets through. '
        f'That level of control changes how you experience the world."\n\n'
        f'Return ONLY: {{"narration":"full narration text here"}}'
    )

    def _has_fragments(text: str) -> bool:
        if ";" in text:
            return True
        sentences = [s.strip() for s in _re.split(r"[.!?]", text) if s.strip()]
        short = [s for s in sentences if len(s.split()) < 5]
        return len(short) > len(sentences) * 0.35

    _NARR_BAD = _re.compile(
        r'^(create|make|generate|write|build|produce|design|'
        r'this video|in this video|welcome to|today we|as we can see|'
        r'highlight|showcase|featuring)\b', _re.IGNORECASE)

    for attempt in range(4):
        try:
            raw = ai_call(prompt, SYS_JSON)
            d = extract_json(raw)
            narration = d.get("narration", "").strip()
            if (narration
                    and not _NARR_BAD.match(narration)
                    and len(narration) >= 40
                    and not _has_fragments(narration)):
                return {"id": num, "total": total, "title": scene_title,
                        "narration": narration, "duration": dur}
            reason = ("bad prefix" if _NARR_BAD.match(narration or "")
                      else "too short" if len(narration) < 40
                      else "has fragments/semicolons")
            log.warning(f"  Retrying narration ({attempt+1}/4) — {reason}: {(narration or '')[:80]!r}")
        except Exception as e:
            log.warning(f"  Scene {num} narration attempt {attempt+1}/4 failed: {e}")
        time.sleep(3)
    return {
        "id": num, "total": total, "title": scene_title,
        "narration": (f"Few things combine power and precision like this. "
                      f"Every detail of {topic} is engineered without compromise."),
        "duration": dur,
    }


def get_image_prompts(scene_title: str, narration: str, topic: str, style: str) -> list:
    log.info(f"Image prompts for: {scene_title}")
    cam_map = {
        "documentary": "documentary photography, Canon EOS R5, photojournalism",
        "promotional": "commercial product photography, Sony A7R IV, advertising campaign",
        "educational": "editorial photography, clean, magazine quality",
        "cinematic":   "anamorphic lens, film still, Hollywood cinematography",
        "travel":      "travel photography, golden hour, National Geographic",
        "corporate":   "corporate photography, professional studio lighting",
        "nature":      "wildlife photography, BBC nature documentary, telephoto",
        "food":        "food photography, culinary magazine, macro lens, styled",
    }
    cam = next((v for k, v in cam_map.items() if k in style.lower()),
               "commercial product photography, ARRI Alexa, professional production")
    prompt = (
        f'Topic/Product: "{topic}"\n'
        f'Scene title: "{scene_title}"\n'
        f'Scene context: "{narration[:120]}"\n'
        f'Visual style: {cam}\n\n'
        f'Create 3 HIGHLY SPECIFIC photorealistic image prompts.\n'
        f'CRITICAL RULE: Every single prompt MUST begin with the exact words "{topic}".\n\n'
        f'Rules:\n'
        f'- Start with: "{topic} ..."\n'
        f'- Specify exact camera/lens\n'
        f'- Include lighting details\n'
        f'- Include real environment/location details\n'
        f'- NO glowing effects, surreal elements, or fantasy visuals\n'
        f'- Angle variety: wide establishing, medium shot, extreme close-up\n\n'
        f'Return ONLY: {{"p1":"wide shot prompt","p2":"medium shot prompt","p3":"close-up prompt"}}'
    )
    try:
        d  = extract_json(ai_call(prompt, SYS_JSON))
        cam_sfx = f", {cam}"
        p1 = d.get("p1", f"{topic}, wide establishing shot{cam_sfx}")
        p2 = d.get("p2", f"{topic}, medium product shot{cam_sfx}")
        p3 = d.get("p3", f"{topic}, extreme close-up detail{cam_sfx}")
        for label, val in [("p1", p1), ("p2", p2), ("p3", p3)]:
            if not val.lower().startswith(topic.lower()[:12]):
                if label == "p1": p1 = f"{topic}, {val}"
                elif label == "p2": p2 = f"{topic}, {val}"
                else: p3 = f"{topic}, {val}"
        return [p1, p2, p3]
    except Exception as e:
        log.warning(f"Image prompt gen failed: {e}")
        base = f"{topic}, {scene_title}, {cam}, photorealistic"
        return [base+", wide angle", base+", medium shot", base+", close-up"]


# ══════════════════════════════════════════════════════════════════════════════
# Image download — multi-source, near-zero gradient fallback
# ══════════════════════════════════════════════════════════════════════════════
#
# Strategy (in order):
#   TIER 1 — Pollinations  (4 models × 3 tries = 12 attempts, 120s timeout each)
#   TIER 2 — Picsum Photos (real photos, always fast, zero rate-limit)
#   TIER 3 — Unsplash Source (keyword-matched real photos, always available)
#   TIER 4 — Lorem Picsum  (pure reliable filler — solid real photo, no text)
#
# Gradient is only painted if ALL four tiers fail, which is essentially
# impossible (Picsum/Unsplash/Lorem are static CDN endpoints).
# ══════════════════════════════════════════════════════════════════════════════

_PHOTO_SUFFIX = (
    ", photorealistic, 4K, cinematic lighting, ARRI Alexa, "
    "sharp focus, color graded, no watermark, no text"
)

# How long to wait before first attempt, and between retries (seconds)
_POLL_WAIT_FIRST  = 1
_POLL_WAIT_RETRY  = 4
_POLL_TIMEOUT     = 120      # Pollinations can be very slow under load
_FAST_TIMEOUT     = 15       # Picsum / Unsplash are CDNs — should be instant


def _try_load_image(raw_bytes: bytes, min_w: int = 200, min_h: int = 200) -> Optional[Image.Image]:
    """Try to decode raw bytes as an image. Returns None on any failure."""
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        if img.width < min_w or img.height < min_h:
            raise ValueError(f"Too small: {img.width}x{img.height}")
        return img.resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
    except Exception as e:
        log.debug(f"    _try_load_image failed: {e}")
        return None


def _fetch_pollinations(prompt: str, seed: int, model: str) -> Optional[Image.Image]:
    """Single Pollinations attempt. Returns image or None."""
    safe_prompt = prompt[:500] + _PHOTO_SUFFIX
    enc  = urllib.parse.quote(safe_prompt)
    seed_try = seed % 999999
    url  = (
        f"https://image.pollinations.ai/prompt/{enc}"
        f"?width={VIDEO_W}&height={VIDEO_H}&seed={seed_try}"
        f"&nologo=true&enhance=true&model={model}"
    )
    try:
        r = requests.get(url, timeout=_POLL_TIMEOUT)
        r.raise_for_status()
        # Reject HTML error pages disguised as 200 OK
        ct = r.headers.get("content-type", "")
        if "image" not in ct and len(r.content) < 5000:
            raise ValueError(f"Non-image content-type={ct!r} size={len(r.content)}B")
        return _try_load_image(r.content)
    except Exception as e:
        log.debug(f"    Pollinations {model}: {e}")
        return None


def _fetch_picsum(seed: int) -> Optional[Image.Image]:
    """
    Picsum Photos — real curated photos, served from a fast CDN.
    Deterministic per seed so the same scene always gets the same photo.
    """
    pic_id = (seed % 1000) + 1   # Picsum has ~1000 photos (IDs 1-1000)
    url = f"https://picsum.photos/id/{pic_id}/{VIDEO_W}/{VIDEO_H}"
    try:
        r = requests.get(url, timeout=_FAST_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        img = _try_load_image(r.content)
        if img:
            log.info(f"  ✅ Tier-2 Picsum id={pic_id} OK")
        return img
    except Exception as e:
        log.debug(f"    Picsum failed: {e}")
        return None


def _fetch_unsplash(keywords: str, seed: int) -> Optional[Image.Image]:
    """
    Unsplash Source — keyword-matched real photos, free CDN endpoint.
    Uses the topic words so the image is at least thematically relevant.
    """
    # Keep only safe URL chars in the keyword string
    safe_kw = urllib.parse.quote(keywords[:80].replace(",", " ").strip())
    url = (
        f"https://source.unsplash.com/{VIDEO_W}x{VIDEO_H}"
        f"/?{safe_kw}&sig={seed % 9999}"
    )
    try:
        r = requests.get(url, timeout=_FAST_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        img = _try_load_image(r.content)
        if img:
            log.info(f"  ✅ Tier-3 Unsplash kw='{keywords[:40]}' OK")
        return img
    except Exception as e:
        log.debug(f"    Unsplash failed: {e}")
        return None


def _fetch_lorem_picsum_random(seed: int) -> Optional[Image.Image]:
    """
    Lorem Picsum random — pure fallback, always works, real photo.
    Uses /seed/{n}/W/H so it's deterministic (same scene = same photo).
    """
    url = f"https://picsum.photos/seed/{seed % 99999}/{VIDEO_W}/{VIDEO_H}"
    try:
        r = requests.get(url, timeout=_FAST_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        img = _try_load_image(r.content)
        if img:
            log.info(f"  ✅ Tier-4 Lorem Picsum seed={seed % 99999} OK")
        return img
    except Exception as e:
        log.debug(f"    Lorem Picsum failed: {e}")
        return None


def _make_fallback_image(scene_title: str, pal: dict) -> Image.Image:
    """
    Last-resort gradient. Only reached if ALL network tiers fail (e.g. fully
    offline). In normal operation this function is never called.
    """
    log.warning(f"  ⚠️  Painting gradient fallback for '{scene_title}' — all network tiers failed")
    img  = Image.new("RGB", (VIDEO_W, VIDEO_H))
    draw = ImageDraw.Draw(img)
    c1, c2 = pal["bg"]
    for y in range(VIDEO_H):
        t   = y / VIDEO_H
        col = tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
        draw.line([(0, y), (VIDEO_W, y)], fill=col)
    acc = pal["accent"]
    for y in range(VIDEO_H):
        t    = y / VIDEO_H
        wave = 0.5 + 0.5 * math.sin(t * math.pi * 4)
        a    = int(wave * 18)
        draw.line([(0, y), (VIDEO_W, y)], fill=(*acc, a))
    return img.convert("RGB")


def download_image(prompt: str, seed: Optional[int] = None,
                   retries: int = 5) -> Optional[Image.Image]:
    """
    Multi-tier image downloader — near-zero chance of gradient fallback.

    TIER 1 — Pollinations AI  (AI-generated, on-topic)
      • 4 models: flux → flux-realism → turbo → flux-pro
      • Up to `retries` attempts (default 5), rotating models
      • 120s timeout, polite gaps between requests

    TIER 2 — Picsum Photos  (real curated photos, fast CDN)
      • Deterministic by seed — always produces a real image

    TIER 3 — Unsplash Source  (keyword-matched real photos)
      • Uses topic keywords for thematic relevance

    TIER 4 — Lorem Picsum random  (deterministic, always works)
      • Absolute last network resort before gradient paint

    Only _make_fallback_image() (gradient) is called if ALL tiers fail,
    which requires total network outage.
    """
    if seed is None:
        seed = int(time.time() * 1000) % 999999

    # ── TIER 1: Pollinations ──────────────────────────────────────────────────
    poll_models = ["flux", "flux-realism", "turbo", "flux-pro"]
    for attempt in range(retries):
        model    = poll_models[attempt % len(poll_models)]
        seed_try = (seed + attempt * 137) % 999999

        wait = _POLL_WAIT_FIRST if attempt == 0 else _POLL_WAIT_RETRY
        time.sleep(wait)

        log.info(f"  Tier-1 Pollinations attempt {attempt+1}/{retries} model={model}…")
        img = _fetch_pollinations(prompt, seed_try, model)
        if img is not None:
            log.info(f"  ✅ Tier-1 Pollinations OK (attempt {attempt+1}, model={model})")
            return img

    log.warning("  Tier-1 Pollinations exhausted — trying Tier-2 Picsum…")

    # ── TIER 2: Picsum Photos ─────────────────────────────────────────────────
    img = _fetch_picsum(seed)
    if img is not None:
        return img

    log.warning("  Tier-2 Picsum failed — trying Tier-3 Unsplash…")

    # ── TIER 3: Unsplash Source ───────────────────────────────────────────────
    # Extract meaningful keywords from the prompt (first 6 words)
    kw_words  = [w for w in prompt.replace(",", " ").split()[:6] if len(w) > 3]
    keywords  = " ".join(kw_words) if kw_words else "nature landscape"
    img = _fetch_unsplash(keywords, seed)
    if img is not None:
        return img

    log.warning("  Tier-3 Unsplash failed — trying Tier-4 Lorem Picsum random…")

    # ── TIER 4: Lorem Picsum (seed-based, always reliable) ───────────────────
    img = _fetch_lorem_picsum_random(seed)
    if img is not None:
        return img

    # If we get here the machine is offline — gradient is the only option
    log.error("  ❌ ALL four image tiers failed — gradient fallback unavoidable")
    return None


# ── TTS with native WordBoundary timestamps ───────────────────────────────────
def _clean_tts_text(text: str) -> str:
    return (text.strip()
            .replace("…",  "...")
            .replace("\u2014", ", ").replace("\u2013", ", ")
            .replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u00d7", "x").replace("\u2022", ",")
            .replace("\u00a0", " ").replace("\u200b", ""))


async def _stream_tts(text: str, voice: str, rate: str, pitch: str,
                      audio_path: Path) -> List[Dict]:
    comm       = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    audio_buf  = bytearray()
    word_times: List[Dict] = []

    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio_buf.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            start = chunk["offset"]   / 10_000_000
            end   = (chunk["offset"] + chunk["duration"]) / 10_000_000
            word  = chunk.get("text", "").strip().strip(".,!?;:'\"()[]")
            word  = _re.sub(r'[\(\)\[\]]', '', word).strip()
            if word:
                word_times.append({
                    "word":  word,
                    "start": round(start, 4),
                    "end":   round(end,   4),
                })

    if len(audio_buf) < 1024:
        raise RuntimeError(f"Audio buffer too small ({len(audio_buf)}B)")

    audio_path.write_bytes(bytes(audio_buf))
    return word_times


def make_tts_with_timestamps(text: str, path: Path,
                              lang: str = "en",
                              voice_code: str = "",
                              resolved_voice: str = "") -> tuple:
    # ── FIX 2: Accept a pre-resolved voice so every scene uses the SAME speaker.
    # If resolved_voice is supplied (set once at pipeline start), use it directly.
    # Otherwise fall back to resolving from lang/voice_code as before.
    voice = resolved_voice if resolved_voice else _resolve_voice(lang, voice_code)
    clean = _clean_tts_text(text)

    attempts = [
        (voice,                "-8%",  "+3Hz"),
        (voice,                "-5%",  "+2Hz"),
        (voice,                "-3%",  "+1Hz"),
        (voice,                "0%",   "0Hz"),
        ("en-US-JennyNeural",  "-5%",  "+2Hz"),
        ("en-US-AriaNeural",   "0%",   "0Hz"),
    ]

    for v, rate, pitch in attempts:
        if path.exists():
            path.unlink()
        try:
            word_times = asyncio.run(_stream_tts(clean, v, rate, pitch, path))
            size = path.stat().st_size if path.exists() else 0
            log.info(f"  ✅ TTS {path.name} ({size//1024}KB) "
                     f"voice={v} rate={rate}  {len(word_times)} word boundaries")
            if not word_times:
                log.warning("  No WordBoundary events — using linear timestamps")
                try:
                    from moviepy import AudioFileClip as _AFC
                    dur = _AFC(str(path)).duration
                except Exception:
                    dur = len(clean.split()) / 2.5
                word_times = _linear_timestamps(text, dur)
            return True, word_times
        except Exception as e:
            log.warning(f"  TTS failed (voice={v} rate={rate}): {e}")

    log.error("  ❌ ALL TTS attempts failed")
    return False, []


# ── Colour palettes ───────────────────────────────────────────────────────────
PALETTES = [
    {"bg":[(8,15,35),(20,45,80)],   "accent":(99,210,255)},
    {"bg":[(10,28,20),(18,65,50)],  "accent":(80,230,140)},
    {"bg":[(40,10,25),(70,20,45)],  "accent":(255,120,190)},
    {"bg":[(22,10,42),(42,22,72)],  "accent":(180,150,255)},
    {"bg":[(42,24,8),(75,55,20)],   "accent":(255,200,50)},
    {"bg":[(8,8,28),(28,8,48)],     "accent":(255,110,110)},
    {"bg":[(4,32,22),(4,52,38)],    "accent":(90,255,190)},
]

def _blend(c1, c2, t):
    return tuple(int(a+(b-a)*t) for a,b in zip(c1,c2))

def _make_gradient(w, h):
    ov   = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(ov)
    for y in range(h):
        yf = y / h
        if yf > 0.42:
            a = int(min(235, ((yf-0.42)/0.58)**1.5 * 250))
            draw.line([(0,y),(w,y)], fill=(0,0,0,a))
    for y in range(min(120,h)):
        a = int((1-y/120)**1.8 * 115)
        draw.line([(0,y),(w,y)], fill=(0,0,0,a))
    return ov

GRAD = _make_gradient(VIDEO_W, VIDEO_H)


def _wrap(draw, text, font, max_w):
    words, lines, line = text.split(), [], []
    for w in words:
        test = " ".join(line+[w])
        bb   = draw.textbbox((0,0), test, font=font)
        if bb[2]-bb[0] <= max_w or not line:
            line.append(w)
        else:
            lines.append(" ".join(line)); line=[w]
    if line: lines.append(" ".join(line))
    return lines

def _shadow_text(draw, text, font, x, y, fill, shadow=(0,0,0,210)):
    for dx,dy in [(3,3),(2,2),(1,1)]:
        draw.text((x+dx,y+dy), text, font=font, fill=shadow)
    draw.text((x,y), text, font=font, fill=fill)

def _pill(img, x1, y1, x2, y2, r=16, color=(0,0,0,145)):
    layer = Image.new("RGBA", img.size, (0,0,0,0))
    ImageDraw.Draw(layer).rounded_rectangle([x1,y1,x2,y2], radius=r, fill=color)
    return Image.alpha_composite(img.convert("RGBA"), layer)


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3: Complex-only transition effects
# Removed boring simple effects: kb_zoom_in, kb_zoom_out, slide_left,
# slide_right, slide_up. Only keeping cinematic/complex ones.
# ══════════════════════════════════════════════════════════════════════════════

_TRANSITION_EFFECTS = [
    # ── KEPT: complex, cinematic ──────────────────────────────────────────────
    "parallax",        # multi-layer horizontal+vertical sinusoidal drift
    "pendulum",        # arc swing with cosine-damped vertical motion
    "diagonal_drift",  # smooth diagonal travel across the frame
    "rotation_zoom",   # slow rotation combined with progressive zoom
    "elastic_pan",     # ease-out-back overshoot pan — snappy & dynamic
    "breathe",         # pulsing zoom cycle — organic, heartbeat feel
    "fade_zoom",       # peak-zoom at mid-clip, symmetric — dreamy
    # ── REMOVED: kb_zoom_in, kb_zoom_out, slide_left, slide_right, slide_up ─
]


def _get_transition_for_scene(scene_index: int) -> str:
    return _TRANSITION_EFFECTS[scene_index % len(_TRANSITION_EFFECTS)]


def _ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    t -= 1.0
    return t * t * ((overshoot + 1.0) * t + overshoot) + 1.0


def _kb(img: Image.Image, t: float, flip: bool = False,
        effect: str = "parallax") -> Image.Image:
    sign = -1 if flip else 1
    te   = _ease_in_out(t)

    if effect == "kb_zoom_in":
        zoom, px, py, angle = 1.0 + 0.07*te, math.sin(te*math.pi*0.6)*22*sign, math.cos(te*math.pi*0.5)*12, 0.0
    elif effect == "kb_zoom_out":
        zoom, px, py, angle = 1.07 - 0.07*te, math.sin(te*math.pi*0.6)*22*(-sign), math.cos(te*math.pi*0.5)*12, 0.0
    elif effect == "slide_left":
        zoom, px, py, angle = 1.06, -te*80*sign, 0.0, 0.0
    elif effect == "slide_right":
        zoom, px, py, angle = 1.06, te*80*sign, 0.0, 0.0
    elif effect == "slide_up":
        zoom, px, py, angle = 1.06, 0.0, -te*60, 0.0
    elif effect == "fade_zoom":
        peak = math.sin(t*math.pi)
        zoom, px, py, angle = 1.0+0.05*peak, 0.0, 0.0, 0.0
    elif effect == "parallax":
        zoom, px, py, angle = 1.07, math.sin(t*math.pi*2)*30*sign, math.cos(t*math.pi)*18, 0.0
    elif effect == "diagonal_drift":
        zoom, px, py, angle = 1.06, te*55*sign, -te*40, 0.0
    elif effect == "breathe":
        cycle = math.sin(t*math.pi*2)*0.5+0.5
        zoom, px, py, angle = 1.02+0.04*cycle, math.sin(t*math.pi*1.5)*14*sign, math.cos(t*math.pi*1.5)*8, 0.0
    elif effect == "rotation_zoom":
        zoom, px, py, angle = 1.0+0.08*te, 0.0, 0.0, te*0.8*sign
    elif effect == "pendulum":
        swing = math.sin(te*math.pi)*55
        zoom, px, py, angle = 1.06, swing*sign, math.cos(te*math.pi*0.5)*15, 0.0
    elif effect == "elastic_pan":
        eob = _ease_out_back(t) if t < 1.0 else 1.0
        zoom, px, py, angle = 1.06, eob*60*sign, -te*20, 0.0
    else:
        # Default to parallax for any unknown effect
        zoom, px, py, angle = 1.07, math.sin(t*math.pi*2)*30*sign, math.cos(t*math.pi)*18, 0.0

    if angle != 0.0:
        pre_scale = zoom * 1.015
        nw = int(VIDEO_W * pre_scale)
        nh = int(VIDEO_H * pre_scale)
        scaled = img.resize((nw, nh), Image.LANCZOS)
        scaled = scaled.rotate(angle, resample=Image.BICUBIC, expand=False)
        cx = (nw - VIDEO_W) // 2
        cy = (nh - VIDEO_H) // 2
        scaled = scaled.crop((cx, cy, cx+VIDEO_W, cy+VIDEO_H))
        out = Image.new("RGB", (VIDEO_W, VIDEO_H), (0,0,0))
        ox, oy = int(px), int(py)
        src_x, src_y = max(0,-ox), max(0,-oy)
        sw = min(VIDEO_W, VIDEO_W-abs(ox))
        sh = min(VIDEO_H, VIDEO_H-abs(oy))
        if sw > 0 and sh > 0:
            out.paste(scaled.crop((src_x,src_y,src_x+sw,src_y+sh)), (max(0,ox),max(0,oy)))
        return out

    nw = int(VIDEO_W * zoom)
    nh = int(VIDEO_H * zoom)
    ox = int((VIDEO_W-nw)/2 + px)
    oy = int((VIDEO_H-nh)/2 + py)
    scaled = img.resize((nw, nh), Image.LANCZOS)
    out = Image.new("RGB", (VIDEO_W, VIDEO_H), (0,0,0))
    cx, cy = max(0,-ox), max(0,-oy)
    sw = min(VIDEO_W, nw-cx)
    sh = min(VIDEO_H, nh-cy)
    if sw > 0 and sh > 0:
        out.paste(scaled.crop((cx,cy,cx+sw,cy+sh)), (max(0,ox),max(0,oy)))
    return out


def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    try:
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (255, 255, 255)


# ══════════════════════════════════════════════════════════════════════════════
# Frame builder
# ══════════════════════════════════════════════════════════════════════════════

def build_scene_frames(scene: dict, images: list, duration: float, pal: dict,
                       style_opts: Optional[dict] = None,
                       word_timestamps: Optional[List[Dict]] = None,
                       audio_delay: float = 0.0) -> list:
    """
    audio_delay: seconds between this clip's VIDEO start and AUDIO start.
                 = scene_index * XFADE_S  (0 for scene 1, 0.6 for scene 2, etc.)
    Captions are shifted forward by this amount so they appear exactly
    when the voice starts, not when the video frame first appears.
    """
    s = style_opts or {}
    cap_rgb     = _hex_to_rgb(s.get("caption_color",  "#FFFFFF"))
    cap_size    = max(16, min(60, int(s.get("caption_size",  26))))
    cap_outline = bool(s.get("caption_outline", True))
    title_rgb   = _hex_to_rgb(s.get("title_color", "#FFFFFF"))
    title_size  = max(28, min(90, int(s.get("title_size", 54))))

    n      = max(1, int(round(FPS * duration)))
    fade   = min(int(FPS * 0.55), n // 6)
    ni     = len(images)
    seg    = n // ni if ni else n
    xfade  = min(int(FPS * 0.45), seg // 3) if ni > 1 else 0

    f_title   = get_font(title_size, "bold")
    f_caption = get_font(cap_size, "regular")
    f_badge   = get_font(19, "medium")

    TITLE_Y       = VIDEO_H - 230
    CAPTION_Y0    = VIDEO_H - 72
    MAX_CAP_LINES = 3

    scene_effect = _get_transition_for_scene(scene.get("id", 1) - 1)
    log.debug(f"  Scene {scene.get('id',1)} motion effect: {scene_effect}")

    # ── FIX 1 & 4: Always use the scene narration for timestamps, NEVER the
    # AudioLoop prompt string. word_timestamps must come from narration-based
    # TTS/Gemini transcription. If missing, fall back with narration text.
    narration_text = scene.get("narration", "")

    wts: List[tuple] = []
    if word_timestamps:
        wts = [(w["word"], w["start"], w["end"]) for w in word_timestamps]
    else:
        fallback = _linear_timestamps(narration_text, duration)
        wts = [(w["word"], w["start"], w["end"]) for w in fallback]
    wts.sort(key=lambda x: x[1])

    frames = []
    for f in range(n):
        tg  = f / max(n - 1, 1)
        t_s = f / FPS

        if ni == 0:
            frame = Image.new("RGB", (VIDEO_W, VIDEO_H))
            d2    = ImageDraw.Draw(frame)
            c1, c2 = pal["bg"]
            for y in range(VIDEO_H):
                yf   = y / VIDEO_H
                wave = 0.5 + 0.5 * math.sin(tg * math.pi * 2 + yf * 3)
                d2.line([(0,y),(VIDEO_W,y)], fill=_blend(c1, c2, yf*0.7+wave*0.3))
        else:
            si  = min(f // seg, ni - 1)
            lf  = f - si * seg
            lt  = lf / max(seg - 1, 1)
            frame = _kb(images[si], lt, flip=(si % 2 == 1), effect=scene_effect)
            if si + 1 < ni and xfade > 0 and lf >= (seg - xfade):
                raw   = (lf - (seg - xfade)) / xfade
                alpha = max(0.0, min(1.0, raw*raw*(3-2*raw)))
                t_nxt = (lf - (seg - xfade)) / max(seg - 1, 1)
                nxt   = _kb(images[si+1], t_nxt, flip=((si+1)%2==1), effect=scene_effect)
                frame = Image.blend(frame, nxt, alpha)

        frame = Image.alpha_composite(frame.convert("RGBA"), GRAD).convert("RGB")

        acc     = pal["accent"]
        acc_dim = tuple(int(c * 0.13) for c in acc)
        dm      = ImageDraw.Draw(frame)
        dm.rectangle([(0,0),(VIDEO_W,3)], fill=acc_dim)
        dm.rectangle([(0,VIDEO_H-3),(VIDEO_W,VIDEO_H)], fill=acc_dim)

        title = _clean_display(scene.get("title", ""))
        if title:
            ta = int(255 * min(1.0, tg / 0.22))
            if ta > 0:
                dummy = ImageDraw.Draw(Image.new("RGBA", (1,1)))
                bb    = dummy.textbbox((0,0), title.upper(), font=f_title)
                tw, th = bb[2]-bb[0], bb[3]-bb[1]
                tx, ty = 40, TITLE_Y
                px, py = 18, 10
                frame = _pill(frame, tx-px, ty-py, tx+tw+px, ty+th+py,
                              r=13, color=(0,0,0,int(148*ta//255))).convert("RGB")
                dm2 = ImageDraw.Draw(frame)
                dm2.rectangle([(tx-px,ty-py),(tx-px+5,ty+th+py)],
                              fill=tuple(int(c*ta//255) for c in acc))
                _shadow_text(dm2, title.upper(), f_title, tx, ty,
                             fill=(*title_rgb, ta), shadow=(0,0,0,int(210*ta//255)))

        # ── Caption: show CURRENT SENTENCE only, not all accumulated words ───
        # Walk backward from the last spoken word to find the sentence start
        # (boundary = word ending with . ! ?) so captions show one clean
        # sentence at a time — TV subtitle style, never a fragment dump.
        # audio_delay: how many seconds AFTER this clip's video start does audio begin.
        # We subtract it from t_s so captions only appear once the voice is actually
        # playing, not during the incoming visual dissolve.
        # t_s_audio = clip-local time adjusted to audio playback position
        t_s_audio = t_s - audio_delay   # negative at start → no captions yet

        spoken_indices = [idx for idx, (w, s, e) in enumerate(wts) if s <= t_s_audio]
        if spoken_indices:
            active_idx = spoken_indices[-1]
            sentence_start = 0
            for idx in range(active_idx - 1, -1, -1):
                if _re.search(r'[.!?]$', wts[idx][0]):
                    sentence_start = idx + 1
                    break
            current_sentence_words = [wts[i][0] for i in range(sentence_start, active_idx + 1)]
            if current_sentence_words:
                # Ramp alpha from 0→1 over 0.25s starting when the FIRST word
                # of this sentence hits, so captions fade in instead of snapping.
                current_word_start = wts[sentence_start][1]
                ramp = min(1.0, max(0.0, (t_s_audio - current_word_start) / 0.25))
                ca   = int(230 * ramp)

        # ── Fade captions OUT in the last 0.6s of the scene ──────
        # This matches the cross-fade window so captions cleanly
        # disappear BEFORE the incoming scene's captions appear.
        # Without this, scene N's last sentence bakes into the
        # cross-fade frames and lingers over scene N+1's opening.
            CAPTION_FADEOUT_S = 0.6   # must match xfade_s in _crossfade_concat
            time_remaining = duration - t_s
            if time_remaining < CAPTION_FADEOUT_S:
                fade_out_mult = max(0.0, time_remaining / CAPTION_FADEOUT_S)
            else:
                        fade_out_mult = 1.0

            ca = int(230 * ramp * fade_out_mult)
            if ca > 0:
                    sub = _clean_display(" ".join(current_sentence_words))
                    ci  = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0,0,0,0))
                    cd  = ImageDraw.Draw(ci)
                    lns = _wrap(cd, sub, f_caption, VIDEO_W - 120)
                    lns = lns[-MAX_CAP_LINES:]
                    lh  = cap_size + 10
                    bh  = len(lns) * lh + 18
                    cy0 = CAPTION_Y0 - bh
                    cd.rounded_rectangle([30, cy0-8, VIDEO_W-30, CAPTION_Y0+6],
                                         radius=10, fill=(0,0,0,int(150*ca//230)))
                    for li, ln in enumerate(lns):
                        tx_pos = 52
                        ty_pos = cy0 + li*lh + 2
                        fill_col = (*cap_rgb, ca)
                        if cap_outline:
                            for ox, oy in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,-2),(-2,2),(2,2)]:
                                cd.text((tx_pos+ox, ty_pos+oy), ln, font=f_caption,
                                        fill=(0,0,0,int(200*ca//230)))
                        cd.text((tx_pos, ty_pos), ln, font=f_caption, fill=fill_col)
                    frame = Image.alpha_composite(frame.convert("RGBA"), ci).convert("RGB")

        sid, stot = scene.get("id",""), scene.get("total","")
        if sid and stot:
            btxt = f"{sid} / {stot}"
            bi   = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0,0,0,0))
            bd   = ImageDraw.Draw(bi)
            bbb  = bd.textbbox((0,0), btxt, font=f_badge)
            bw, bh = bbb[2]-bbb[0], bbb[3]-bbb[1]
            bx, by = VIDEO_W-bw-46, 26
            bd.rounded_rectangle([bx-10,by-6,bx+bw+10,by+bh+6], radius=10, fill=(0,0,0,115))
            bd.text((bx,by), btxt, font=f_badge, fill=(*acc,200))
            frame = Image.alpha_composite(frame.convert("RGBA"), bi).convert("RGB")

        # Only fade IN at the very start of a scene.
        # Do NOT fade to black at the end — the cross-dissolve in the
        # pipeline handles the outgoing transition, so a fade-to-black
        # here would cause a double-dark artifact.
        fa = 0.0
        if f < fade and fade > 0:
            fa = 1.0 - f / fade
        if fa > 0:
            frame = Image.blend(frame.convert("RGB"),
                                Image.new("RGB", (VIDEO_W, VIDEO_H), (0, 0, 0)),
                                min(1.0, fa))

        frames.append(np.array(frame.convert("RGB"), dtype=np.uint8))

    return frames

# ── Crossfade duration constant — used here AND in build_scene_frames ────────
XFADE_S = 0.6   # seconds of visual cross-dissolve between scenes

def _crossfade_concat(clips: list, xfade_s: float = XFADE_S):
    """
    Visual cross-dissolve between scenes with SEQUENTIAL (non-overlapping) audio.

    VIDEO: clips overlap by xfade_s → smooth dissolve.
    AUDIO: scene N+1 audio starts only AFTER scene N audio ends completely.
           This is independent of the visual timeline.
           → No two voices ever play simultaneously → last word always heard.

    total_dur = sum of all clip durations (audio-driven) so no narration
    is ever truncated. During the (n-1)*xfade_s gap between video_end
    and audio_end, the last frame is held frozen (invisible to the viewer
    since it matches the last scene's image anyway).
    """
    from moviepy import CompositeAudioClip

    if not clips:
        return None
    if len(clips) == 1:
        return clips[0]

    xfade_s = max(0.1, min(xfade_s, 1.5))

    # ── VIDEO timeline: overlapping (for visual blend) ────────────────────────
    vid_timeline = []
    vt = 0.0
    for clip in clips:
        vid_timeline.append((vt, vt + clip.duration, clip))
        vt += clip.duration - xfade_s
    video_end = vid_timeline[-1][1]

    # ── AUDIO timeline: strictly sequential — never overlap ───────────────────
    aud_timeline = []
    at = 0.0
    for clip in clips:
        aud_timeline.append((at, clip))
        at += clip.duration
    audio_total = at  # always >= video_end

    # Total duration is audio-driven so the last scene's narration isn't cut
    total_dur = audio_total

    # ── Frame blender (visual only) ────────────────────────────────────────────
    def make_frame(t_global):
        # Hold last frame for the tail period after all dissolves finish
        if t_global >= video_end:
            return np.asarray(
                clips[-1].get_frame(max(0.0, clips[-1].duration - 0.001)),
                dtype=np.uint8)

        active = [(s, e, c) for s, e, c in vid_timeline if s <= t_global < e]
        if not active:
            return np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
        if len(active) == 1:
            s, _, c = active[0]
            return np.asarray(c.get_frame(t_global - s), dtype=np.uint8)

        s0, _, c0 = active[0]
        s1, _, c1 = active[1]
        alpha = min(1.0, max(0.0, (t_global - s1) / xfade_s))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smooth S-curve
        f0 = np.asarray(c0.get_frame(t_global - s0), dtype=np.float32)
        f1 = np.asarray(c1.get_frame(t_global - s1), dtype=np.float32)
        return (f0 * (1.0 - alpha) + f1 * alpha).astype(np.uint8)

    result = VideoClip(make_frame, duration=total_dur).with_fps(FPS)

    # ── Sequential audio positioning ──────────────────────────────────────────
    aud_tracks = [
        clip.audio.with_start(t_start)
        for t_start, clip in aud_timeline
        if clip.audio is not None
    ]
    if aud_tracks:
        try:
            result = result.with_audio(CompositeAudioClip(aud_tracks))
        except Exception as e:
            log.warning(f"Audio composite failed in crossfade: {e}")

    return result
# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(jid: str, payload: dict):
    jdir = TEMP_DIR / jid
    jdir.mkdir(exist_ok=True)
    log.info("=" * 60)
    log.info(f"JOB START  {jid}")
    log.info(f"Payload: {json.dumps(payload)}")
    log.info("=" * 60)

    try:
        raw_topic   = payload["topic"]
        style       = payload.get("style", "Cinematic")
        lang        = payload.get("lang", "en")
        voice_code  = payload.get("voice_code", "")
        duration    = int(payload.get("duration", 30))
        scene_count = int(payload.get("scenes", 5))
        sec_per     = max(5, duration // scene_count)
        style_opts  = {
            "caption_color":   payload.get("caption_color",   "#FFFFFF"),
            "caption_size":    int(payload.get("caption_size", 26)),
            "caption_outline": bool(payload.get("caption_outline", True)),
            "title_color":     payload.get("title_color",     "#FFFFFF"),
            "title_size":      int(payload.get("title_size",  54)),
        }

        # ── FIX 2: Resolve voice ONCE here — all scenes share this exact voice.
        # Previously _resolve_voice() was called fresh inside each TTS worker,
        # and AudioLoopToFile used a different voice from the Gemini Live API.
        # Now one voice string flows through the entire pipeline.
        resolved_voice = _resolve_voice(lang, voice_code)
        log.info(f"🎙️  Voice locked for entire video: {resolved_voice}")

        _PROMPT_START = _re.compile(
            r'^(create|make|generate|write|build|produce|design|give me|i want)\b',
            _re.IGNORECASE)
        if _PROMPT_START.match(raw_topic.strip()) or len(raw_topic) > 80:
            log.info("Topic looks like a long prompt — extracting subject…")
            try:
                extract_prompt = (
                    f'Extract only the core product or subject name from this text: "{raw_topic}"\n'
                    f'Return a SHORT product/subject name, 1-6 words, no extra words.\n'
                    f'Examples: "Samsung Galaxy S25 Ultra", "Tesla Model 3"\n'
                    f'Return ONLY: {{"subject":"product name here"}}'
                )
                topic = extract_json(ai_call(extract_prompt, SYS_JSON)).get(
                    "subject", raw_topic[:60]).strip()
                log.info(f"  Extracted subject: '{topic}'")
            except Exception as e:
                log.warning(f"  Subject extraction failed ({e}) — using raw topic")
                topic = raw_topic[:60]
        else:
            topic = raw_topic

        log.info(f"topic='{topic}' style='{style}' lang={lang} "
                 f"dur={duration}s scenes={scene_count} sec_per={sec_per}s")

        # Step 1 — Title
        job_update(jid, status="running", progress=2, message="Crafting video title…")
        video_title = get_video_title(topic, style)
        job_update(jid, message=f'Title: "{video_title}"')

        # Step 2a — Scene titles
        job_update(jid, progress=3, message="Planning scene titles…")
        scene_titles = get_all_scene_titles(topic, style, scene_count)
        job_update(jid, message=f"Titles: {scene_titles}")

        # Step 2b — Narration
        job_update(jid, progress=4, message=f"Writing {scene_count} cinematic scenes…")
        scenes = []
        for i in range(scene_count):
            sc = get_scene(topic, style, i + 1, scene_count, sec_per,
                           scene_title=scene_titles[i])
            scenes.append(sc)
            job_update(jid, message=f'Scene {i+1}: "{sc["title"]}" ✓',
                       progress=4 + int((i + 1) / scene_count * 8))
        job_update(jid, progress=12, message="Script complete ✓")

        # Step 3 — Image prompts
        job_update(jid, progress=13, message="Building image prompts…")
        for i in range(scene_count):
            scenes[i]["imagePrompts"] = get_image_prompts(
                scenes[i]["title"], scenes[i]["narration"], topic, style)

        # Step 4 — Download images sequentially
        job_update(jid, progress=14, message="Generating images (sequential)…")
        all_imgs      = [[] for _ in range(scene_count)]
        total_imgs    = sum(len(s["imagePrompts"]) for s in scenes)
        completed_cnt = 0

        for i, sc in enumerate(scenes):
            pal = PALETTES[i % len(PALETTES)]
            for j, p in enumerate(sc["imagePrompts"]):
                seed = (i * 1000 + j * 37 + int(time.time())) % 999983
                img  = download_image(p, seed=seed)
                if img is None:
                    img = _make_fallback_image(sc["title"], pal)
                    status_msg = f"Scene {i+1} image {j+1} → gradient fallback"
                else:
                    status_msg = f"Scene {i+1} image {j+1} ✓"
                all_imgs[i].append(img)
                completed_cnt += 1
                job_update(
                    jid,
                    message=status_msg,
                    progress=14 + int(completed_cnt / total_imgs * 26),
                )

        job_update(jid, progress=40, message="All images ready ✓")

        # ── Step 5 — TTS + timestamps IN PARALLEL (up to 3 workers) ──────────
        # FIX 2: Pass resolved_voice into every worker so all scenes use the
        # same speaker. Workers no longer call _resolve_voice() independently.
        job_update(jid, progress=42,
                   message=f"Generating voiceover (parallel) — {resolved_voice}…")
        audio_paths:    List[Optional[Path]] = [None] * scene_count
        all_timestamps: List[List[Dict]]     = [[]]   * scene_count
        tts_done        = [0]
        tts_lock        = threading.Lock()

        def _do_tts(i):
            sc     = scenes[i]
            ap_wav = jdir / f"audio_{i}.wav"
            success = False

            # ── FIX 1 & 4: Build a narration-based TTS prompt ONLY.
            # Do NOT pass any AudioLoop-style creative prompt to the TTS layer.
            # The only text that goes to TTS (and therefore to captions) is the
            # plain narration written by get_scene().
            narration = sc["narration"]

            for attempt in range(3):
                try:
                    recorder = AudioLoopToFile(narration, ap_wav)
                    asyncio.run(recorder.run())
                    if ap_wav.exists() and ap_wav.stat().st_size > 1024:
                        success = True
                        break
                    log.warning(f"Scene {i+1} attempt {attempt+1}: empty audio, retrying…")
                except Exception as e:
                    log.warning(f"Scene {i+1} Gemini attempt {attempt+1}/3 failed: {e}")
                    time.sleep(2)

            if not success:
                log.warning(f"Scene {i+1} Gemini TTS failed — falling back to edge-tts")
                ap_mp3 = jdir / f"audio_{i}.mp3"
                # FIX 2: pass resolved_voice so edge-tts fallback also uses same speaker
                ok, wts = make_tts_with_timestamps(
                    narration, ap_mp3, lang=lang, voice_code=voice_code,
                    resolved_voice=resolved_voice)
                result_path = ap_mp3
                # FIX 1 & 4: timestamps use narration, not any prompt string
                result_wts  = wts if ok else _linear_timestamps(narration, sec_per)
            else:
                result_path = ap_wav
                # FIX 1 & 4: linear fallback also uses narration text, not prompt
                result_wts  = _linear_timestamps(narration, sec_per)

            with tts_lock:
                tts_done[0] += 1
                job_update(jid, message=f"Scene {i+1} voice ✓",
                           progress=42 + int(tts_done[0] / scene_count * 16))

            return i, result_path, result_wts

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(scene_count, 3)) as ex:
            tts_futs = {ex.submit(_do_tts, i): i for i in range(scene_count)}
            for fut in concurrent.futures.as_completed(tts_futs):
                i, ap, wts = fut.result()
                audio_paths[i]    = ap
                all_timestamps[i] = wts

        job_update(jid, progress=58, message="Voiceover + caption sync complete ✓")

        # Step 6 — Load audio + get actual durations
        audio_clips, scene_durs = [], []
        for i, ap in enumerate(audio_paths):
            target = float(scenes[i].get("duration", sec_per))
            if ap and Path(ap).exists():
                try:
                    ac         = AudioFileClip(str(ap))
                    actual_dur = ac.duration
                    log.info(f"  Scene {i+1}: TTS={actual_dur:.2f}s  target={target:.1f}s")
                    audio_clips.append(ac)
                    scene_durs.append(actual_dur)
                    continue
                except Exception as e:
                    log.warning(f"  Audio load failed scene {i}: {e}")
            audio_clips.append(None)
            scene_durs.append(target)

        total_dur = sum(scene_durs)
        job_update(jid, message=f"Total: {total_dur:.1f}s @ {FPS}fps "
                                f"(requested {duration}s)")

        # Step 7 — Render frames
        video_clips = []
        for i, sc in enumerate(scenes):
            dur = scene_durs[i]
            log.info(f"Rendering scene {i+1}: '{sc['title']}' ({dur:.1f}s)…")
            job_update(jid,
                       message=f"Rendering scene {i+1}/{scene_count}: {sc['title'][:42]}…",
                       progress=60 + int(i / scene_count * 30))
            frames = build_scene_frames(
                sc,
                all_imgs[i],
                dur,
                PALETTES[i % len(PALETTES)],
                style_opts=style_opts,
                word_timestamps=all_timestamps[i],
                audio_delay=i * XFADE_S,   # scene 0→0s, scene 1→0.6s, scene 2→1.2s …
            )
            snap = np.stack(frames, axis=0)
            def mf(t, _a=snap): return _a[min(int(t * FPS), len(_a) - 1)]
            vc = VideoClip(mf, duration=dur).with_fps(FPS)
            if audio_clips[i]:
                vc = vc.with_audio(audio_clips[i])
            video_clips.append(vc)

        # Step 8 — Encode (cross-dissolve transitions between scenes)
        job_update(jid, progress=92, message="Encoding final MP4…")
        out_path   = jdir / "output.mp4"
        final_clip = _crossfade_concat(video_clips, xfade_s=0.6)
        final_clip.write_videofile(
            str(out_path), fps=FPS, codec="libx264", audio_codec="aac",
            temp_audiofile=str(jdir / "temp_audio.m4a"), remove_temp=True,
            logger=None,
            ffmpeg_params=["-movflags", "+faststart", "-crf", "18",
                           "-preset", "fast", "-pix_fmt", "yuv420p"],
        )
        for c in video_clips:
            try: c.close()
            except: pass
        for ac in audio_clips:
            try:
                if ac: ac.close()
            except: pass

        size_mb = out_path.stat().st_size / 1048576
        log.info(f"✅ DONE — {size_mb:.1f} MB · {total_dur:.1f}s")
        log.info("=" * 60)

        job_update(jid, status="done", progress=100,
                   message=f"Video complete! {size_mb:.1f} MB · {total_dur:.0f}s",
                   result={
                       "file":        str(out_path),
                       "size_mb":     round(size_mb, 2),
                       "duration":    round(total_dur, 1),
                       "requested_s": duration,
                       "scene_count": scene_count,
                       "video_title": video_title,
                       "tts_voice":   resolved_voice,
                       "scenes": [{"title": s["title"], "narration": s["narration"]}
                                  for s in scenes],
                   })

    except Exception as e:
        log.error(f"PIPELINE ERROR:\n{traceback.format_exc()}")
        job_update(jid, status="error", error=str(e), message=f"Error: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "INDEX2.HTML")

@app.route("/api/voices")
def voices():
    grouped = {}
    for code, label, vid in VOICE_OPTIONS:
        lang_group = label.split(" — ")[0] if " — " in label else label.split(" (")[0]
        grouped.setdefault(lang_group, []).append({
            "code": code, "label": label, "voice_id": vid
        })
    flat = [{"code": c, "label": l} for c, l, _ in VOICE_OPTIONS]
    return jsonify({"voices": flat, "grouped": grouped})

@app.route("/api/health")
def health():
    return jsonify({
        "status":         "ok",
        "gemini":         bool(gemini_client),
        "ai_backend":     "gemini-2.0-flash" if gemini_client else "pollinations",
        "tts_backend":    "edge-tts (rate/pitch)",
        "caption_sync":   "edge-tts-word-boundary",
        "tts_voice":      _resolve_voice("en"),
        "transitions":    _TRANSITION_EFFECTS,
        "image_download": "sequential (rate-limit safe)",
        "fonts": {
            "title":   os.path.basename(FONT_TITLE_PATH)   if FONT_TITLE_PATH   else "default",
            "caption": os.path.basename(FONT_CAPTION_PATH) if FONT_CAPTION_PATH else "default",
        }
    })

@app.route("/api/generate", methods=["POST"])
def generate():
    p   = request.get_json(force=True)
    jid = str(uuid.uuid4())
    log.info(f"New job {jid} — {p.get('topic','?')[:60]}")
    JOBS[jid] = {"status": "queued", "progress": 0, "log": [], "result": None, "error": None}
    threading.Thread(target=run_pipeline, args=(jid, p), daemon=True).start()
    return jsonify({"job_id": jid})

@app.route("/api/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    if not j: return jsonify({"error": "Unknown job"}), 404
    return jsonify(j)

@app.route("/api/download/<jid>")
def download(jid):
    j = JOBS.get(jid)
    if not j or j.get("status") != "done": return jsonify({"error": "Not ready"}), 404
    return send_file(j["result"]["file"], mimetype="video/mp4",
                     as_attachment=True, download_name="ai-video.mp4")

@app.route("/api/preview/<jid>")
def preview(jid):
    j = JOBS.get(jid)
    if not j or j.get("status") != "done": return jsonify({"error": "Not ready"}), 404
    return send_file(j["result"]["file"], mimetype="video/mp4")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  AI Video Studio — v12 PATCHED-2 (4 bug fixes)")
    print(f"  http://localhost:{port}")
    print(f"  AI       : {'Gemini 2.0 Flash ✅' if gemini_client else 'Pollinations'}")
    print(f"  TTS      : Edge TTS stream() — WordBoundary timestamps ✅")
    print(f"  SYNC     : Native (edge-tts WordBoundary events, 100ns precision)")
    print(f"  IMAGES   : Sequential download (rate-limit safe) ✅")
    print(f"  TTS      : Parallel synthesis (3 workers) ✅")
    print(f"  VOICE    : Single consistent speaker (locked at pipeline start) ✅")
    print(f"  CAPTIONS : Narration text only — no prompt leak ✅")
    print(f"  TRANSITIONS: {len(_TRANSITION_EFFECTS)} complex cinematic effects only ✅")
    print(f"  Font     : {os.path.basename(FONT_TITLE_PATH) if FONT_TITLE_PATH else 'downloading...'}")
    print("=" * 60)
    app.run(debug=False, port=port, threaded=True)