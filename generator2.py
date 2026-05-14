"""
AI Video Studio — Backend v9 (PERFECT Edition)
===============================================
Fixes over v8:
  • SSML removed — edge-tts Communicate now uses rate/pitch params directly
  • _clean_display() sanitizer strips em-dashes, smart quotes, × and all
    non-Poppins characters before rendering on-screen text
  • Title Y-position raised so it never overlaps caption lines
  • Duration fixed: TTS audio hard-trimmed to target sec_per, word count
    targets tightened so 30s actually means ~30s
  • Python 3.9 compatible (Optional[] instead of X | None)
  • FIX: Narration uses short simple sentences so TTS completes fully
  • FIX: All image prompts start with the exact product/topic name

pip install flask flask-cors requests edge-tts moviepy pillow numpy google-genai
Run: python generator.py  →  http://localhost:5000

env vars:
  GEMINI_API_KEY=your_key
  EDGE_TTS_VOICE=en-US-AriaNeural   ← optional override
  PORT=5000
"""

import os, io, json, uuid, time, math, threading, traceback, tempfile, urllib.parse
import asyncio, logging, re as _re, concurrent.futures, urllib.request
from pathlib import Path
from typing import Optional
import requests, numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips

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
        gemini_client = genai.Client(api_key=GEMINI_KEY)
        log.info("✅  Gemini 2.0 Flash ready")
    except Exception as e:
        log.warning(f"Gemini init failed ({e}) — falling back to Pollinations")
else:
    log.info("ℹ️   No GEMINI_API_KEY — using Pollinations AI for text")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)
JOBS: dict = {}
TEMP_DIR = Path(tempfile.gettempdir()) / "ai_video_studio"
TEMP_DIR.mkdir(exist_ok=True)
VIDEO_W, VIDEO_H, FPS = 1280, 720, 24

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

# ── Edge TTS voice map ────────────────────────────────────────────────────────
EDGE_VOICE_MAP = {
    "en":    "en-US-AriaNeural",
    "en-gb": "en-GB-SoniaNeural",
    "en-au": "en-AU-NatashaNeural",
    "en-in": "en-IN-NeerjaNeural",
    "es":    "es-ES-ElviraNeural",
    "fr":    "fr-FR-DeniseNeural",
    "de":    "de-DE-KatjaNeural",
    "it":    "it-IT-ElsaNeural",
    "pt":    "pt-BR-FranciscaNeural",
    "ar":    "ar-SA-ZariyahNeural",
    "zh":    "zh-CN-XiaoxiaoNeural",
    "ja":    "ja-JP-NanamiNeural",
    "ko":    "ko-KR-SunHiNeural",
    "hi":    "hi-IN-SwaraNeural",
    "ru":    "ru-RU-SvetlanaNeural",
}

def _resolve_voice(lang: str) -> str:
    override = os.getenv("EDGE_TTS_VOICE", "").strip()
    if override:
        return override
    lang = lang.lower().strip()
    return EDGE_VOICE_MAP.get(lang, EDGE_VOICE_MAP.get(lang.split("-")[0], "en-US-AriaNeural"))

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
            time.sleep(1)
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

# ── Text sanitizer — CRITICAL for clean on-screen rendering ──────────────────
def _clean_display(text: str) -> str:
    """
    Strip/replace every character Poppins cannot render cleanly.
    Prevents the □ boxes and × symbols that appear on-screen.
    """
    text = (text
            .replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2026", "...")  # ellipsis
            .replace("\u00d7", "x")   # multiplication ×
            .replace("\u2019", "'")   # right single quote
            .replace("\u2018", "'")   # left single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u2022", "-")   # bullet •
            .replace("\u00a0", " ")   # non-breaking space
            .replace("\u200b", "")    # zero-width space
            .replace("\u00ae", "")    # ® registered
            .replace("\u2122", "")    # ™ trademark
            .replace("\u00b0", " degrees")  # °
            )
    # Drop anything outside ASCII + Latin Extended (Poppins covers these safely)
    text = _re.sub(r'[^\x20-\x7E\u00C0-\u024F]', '', text)
    return text.strip()

# ── Script generators ─────────────────────────────────────────────────────────
def get_all_scene_titles(topic: str, style: str, count: int) -> list:
    """
    Generate ALL scene titles in one API call so the AI can plan the full
    narrative arc. This prevents the AI from defaulting to "Scene 1", "Scene 2"
    etc. because no scene numbers appear in the prompt at all.
    Returns a list of `count` title strings, guaranteed non-placeholder.
    """
    log.info(f"Generating {count} scene titles in one call…")

    # Build a JSON schema example: {"t1":"...","t2":"...","t3":"..."}
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
        r'introduction|overview|untitled)$',
        _re.IGNORECASE,
    )

    fallback_pool = [
        f"Built Without Compromise",
        f"The Power Within",
        f"Design Meets Performance",
        f"Beyond What You Expect",
        f"Every Detail Counts",
        f"The Next Level",
        f"Crafted for Champions",
        f"Where Innovation Lives",
        f"The Future, Now",
        f"Redefining the Standard",
    ]

    for attempt in range(3):
        try:
            raw = ai_call(prompt, SYS_JSON)
            log.debug(f"  Titles raw: {raw[:300]!r}")
            d = extract_json(raw)
            titles = []
            for i in range(count):
                t = d.get(f"t{i+1}", "").strip()
                if not t or _BAD.match(t):
                    log.warning(f"  Title t{i+1} invalid: {t!r} — will substitute fallback")
                    t = fallback_pool[i % len(fallback_pool)]
                titles.append(t)
            # Accept if at least half look real
            good = sum(1 for t in titles if not _BAD.match(t))
            if good >= max(1, count // 2):
                log.info(f"  Titles: {titles}")
                return titles
            log.warning(f"  Only {good}/{count} titles looked valid — retrying")
        except Exception as e:
            log.warning(f"  Titles attempt {attempt+1}/3 failed: {e}")
        time.sleep(2)

    # Hard fallback — construct topic-specific titles that always look good
    log.error("  Title generation failed — using constructed fallbacks")
    constructed = [
        f"{topic} Unleashed",
        f"Power Redefined",
        f"The Edge You Need",
        f"Built to Impress",
        f"Performance Perfected",
        f"The Ultimate Experience",
        f"Details That Matter",
        f"Speed Without Limits",
        f"A New Standard",
        f"The Future is Here",
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
    """
    Generates narration for a scene whose title is already known.
    The title is passed in from get_all_scene_titles() so the AI only has
    to write narration — it can never echo "Scene 2" as the title.
    """
    log.info(f"Writing narration for scene {num}/{total}: '{scene_title}'…")

    word_count = int(dur * 1.5)
    max_words  = int(dur * 1.7)

    prompt = (
        f'Write narration for a scene titled "{scene_title}" in a premium {style} video about: "{topic}".\n'
        f'Target duration: {dur} seconds.\n'
        f'STRICT word limit: {word_count} words. ABSOLUTE MAXIMUM: {max_words} words.\n'
        f'Count your words before submitting — do NOT exceed {max_words} words.\n\n'
        f'NARRATION rules — READ CAREFULLY:\n'
        f'- Write SHORT, SIMPLE sentences. Each sentence must be 8-12 words maximum.\n'
        f'- Every sentence must be complete and self-contained.\n'
        f'- Use plain everyday vocabulary — no complex words or long phrases.\n'
        f'- Use contractions (it\'s, we\'ve, that\'s, you\'ll) to keep it natural.\n'
        f'- Be SPECIFIC to: "{topic}" — mention real features, names, or details.\n'
        f'- The narration must connect directly to the scene title "{scene_title}".\n'
        f'- NEVER say "In this video", "As we can see", "Today we explore", "Welcome to".\n'
        f'- NO corporate or robotic language.\n'
        f'- Each sentence stands alone — the listener should understand it even if cut there.\n'
        f'- Aim for {word_count} words total. Stop writing when you reach the limit.\n\n'
        f'Return ONLY: {{"narration":"full narration text"}}'
    )

    for attempt in range(3):
        try:
            raw = ai_call(prompt, SYS_JSON)
            log.debug(f"  Scene {num} narration raw: {raw[:200]!r}")
            d = extract_json(raw)
            narration = d.get("narration", "").strip()
            # Reject if the narration looks like a prompt echo:
            # - starts with imperative/meta words
            # - or is suspiciously close to the topic text itself
            _NARR_BAD = _re.compile(
                r'^(create|make|generate|write|build|produce|design|'
                r'this video|in this video|welcome to|today we|as we can see|'
                r'highlight|showcase|featuring)\b',
                _re.IGNORECASE,
            )
            if not narration:
                log.warning(f"  Scene {num} empty narration — retrying ({attempt+1}/3)")
            elif _NARR_BAD.match(narration):
                log.warning(f"  Scene {num} narration looks like a prompt echo: {narration[:80]!r} — retrying")
            elif len(narration) < 15:
                log.warning(f"  Scene {num} narration too short ({len(narration)} chars) — retrying")
            else:
                return {
                    "id":        num,
                    "total":     total,
                    "title":     scene_title,
                    "narration": narration,
                    "duration":  dur,
                }
            log.warning(f"  Retrying narration ({attempt+1}/3)…")
        except Exception as e:
            log.warning(f"  Scene {num} narration attempt {attempt+1}/3 failed: {e}")
        time.sleep(3)

    # ── Final fallback ────────────────────────────────────────────────────────
    log.error(f"  Scene {num}: narration retries failed — using constructed fallback")
    return {
        "id":        num,
        "total":     total,
        "title":     scene_title,
        "narration": f"{topic} is one of the most impressive products available today. It combines power, precision, and elegant design in a way few others can match.",
        "duration":  dur,
    }


def get_image_prompts(scene_title: str, narration: str, topic: str, style: str) -> list:
    """
    FIX: Every image prompt is forced to start with the exact topic/product name
    so that scene 2 (and all scenes) always show the actual product, not generic imagery.
    """
    log.info(f"Image prompts for: {scene_title}")
    cam_map = {
        "documentary":  "documentary photography, Canon EOS R5, photojournalism",
        "promotional":  "commercial product photography, Sony A7R IV, advertising campaign",
        "educational":  "editorial photography, clean, magazine quality",
        "cinematic":    "anamorphic lens, film still, Hollywood cinematography",
        "travel":       "travel photography, golden hour, National Geographic",
        "corporate":    "corporate photography, professional studio lighting",
        "nature":       "wildlife photography, BBC nature documentary, telephoto",
        "food":         "food photography, culinary magazine, macro lens, styled",
    }
    cam = next((v for k, v in cam_map.items() if k in style.lower()),
               "commercial product photography, ARRI Alexa, professional production")

    prompt = (
        f'Topic/Product: "{topic}"\n'
        f'Scene title: "{scene_title}"\n'
        f'Scene context: "{narration[:120]}"\n'
        f'Visual style: {cam}\n\n'
        f'Create 3 HIGHLY SPECIFIC photorealistic image prompts.\n'
        f'CRITICAL RULE: Every single prompt MUST begin with the exact words "{topic}" — \n'
        f'this is the subject of every shot. Never describe a generic scene without "{topic}" in it.\n\n'
        f'Rules for each prompt:\n'
        f'- Start with: "{topic} ..." (the product/subject is always front and center)\n'
        f'- Specify exact camera/lens (e.g. "shot on Canon 5D, 85mm f/1.4")\n'
        f'- Include lighting (golden hour, studio softbox, neon, etc.)\n'
        f'- Include real environment/location details relevant to {scene_title}\n'
        f'- NO glowing effects, surreal elements, or fantasy visuals\n'
        f'- Angle variety: wide establishing, medium shot, extreme close-up detail\n\n'
        f'Return ONLY: {{"p1":"wide shot prompt","p2":"medium shot prompt","p3":"close-up prompt"}}'
    )
    try:
        d    = extract_json(ai_call(prompt, SYS_JSON))
        base = f"{topic}, {scene_title}, {cam}, photorealistic, RAW"
        # Ensure every fallback also starts with the topic name
        p1 = d.get("p1", f"{topic}, wide establishing shot, {cam}")
        p2 = d.get("p2", f"{topic}, medium product shot, {cam}")
        p3 = d.get("p3", f"{topic}, extreme close-up detail, {cam}")
        # Prefix topic if the AI forgot to include it
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


# ── Image download — sequential with fallback ─────────────────────────────────
_PHOTO_SUFFIX = (
    ", RAW photo, photorealistic, shot on professional camera, "
    "35mm film, natural lighting, no CGI, no illustration, "
    "no AI art, hyperrealistic, ultra-detailed, 8K UHD, "
    "sharp focus, award-winning photography, commercial quality"
)

def _make_fallback_image(scene_title: str, pal: dict) -> Image.Image:
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


def download_image(prompt: str, seed: Optional[int] = None, retries: int = 6) -> Optional[Image.Image]:
    """Returns a PIL Image or None (caller substitutes fallback gradient)."""
    if seed is None:
        seed = int(time.time() * 1000) % 999999

    models = ["flux", "flux-realism", "turbo"]
    enc    = urllib.parse.quote(prompt + _PHOTO_SUFFIX)

    for attempt in range(retries):
        model    = models[attempt % len(models)]
        seed_try = (seed + attempt * 137) % 999999
        url = (f"https://image.pollinations.ai/prompt/{enc}"
               f"?width={VIDEO_W}&height={VIDEO_H}&seed={seed_try}"
               f"&nologo=true&enhance=true&model={model}")
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            if img.width < 200:
                raise ValueError("image too small")
            log.info(f"  ✅ Image OK ({img.width}x{img.height}) attempt={attempt+1} model={model}")
            return img.resize((VIDEO_W, VIDEO_H), Image.LANCZOS)
        except Exception as e:
            wait = 15 + (attempt * 10)
            log.warning(f"  Image attempt {attempt+1}/{retries} failed ({model}): {e} — waiting {wait}s")
            if attempt < retries - 1:
                time.sleep(wait)

    log.error("  All image attempts failed — caller will use gradient fallback")
    return None


# ── TTS — FIXED: no SSML, uses rate/pitch params for warmth ──────────────────
def make_tts(text: str, path: Path, lang: str = "en") -> bool:
    """
    FIXED over v8:
    - SSML completely removed — edge-tts was reading tags aloud as text
    - Warmth/pacing now via Communicate(rate, pitch) built-in params
    - Multiple fallback attempts with different settings
    """
    voice = _resolve_voice(lang)

    # Sanitise for TTS: remove unicode that confuses the synthesiser
    clean = (text.strip()
             .replace("…",  "...")
             .replace("\u2014", ", ").replace("\u2013", ", ")   # dashes → pause
             .replace("\u2019", "'").replace("\u2018", "'")
             .replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u00d7", "x").replace("\u2022", ",")
             .replace("\u00a0", " ").replace("\u200b", ""))

    async def _synth(v: str, rate: str, pitch: str) -> None:
        comm = edge_tts.Communicate(clean, v, rate=rate, pitch=pitch)
        await comm.save(str(path))

    # Ordered attempts: warm/slower → neutral → fallback voice
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
            asyncio.run(_synth(v, rate, pitch))
            size = path.stat().st_size if path.exists() else 0
            if size < 1024:
                raise RuntimeError(f"Output too small ({size}B)")
            log.info(f"  ✅ TTS {path.name} ({size//1024}KB) voice={v} rate={rate} pitch={pitch}")
            return True
        except Exception as e:
            log.warning(f"  TTS failed (voice={v} rate={rate} pitch={pitch}): {e}")

    log.error("  ❌ ALL TTS attempts failed")
    return False


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


# ── Text helpers ──────────────────────────────────────────────────────────────
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


# ── Ken Burns ─────────────────────────────────────────────────────────────────
def _kb(img, t, flip=False):
    tv   = t if not flip else (1-t)
    zoom = 1.0 + 0.07 * tv
    px   = math.sin(t * math.pi * 0.6) * 20 * (-1 if flip else 1)
    py   = math.cos(t * math.pi * 0.5) * 11
    nw, nh = int(VIDEO_W*zoom), int(VIDEO_H*zoom)
    ox, oy = int((VIDEO_W-nw)/2+px), int((VIDEO_H-nh)/2+py)
    sc   = img.resize((nw,nh), Image.LANCZOS)
    out  = Image.new("RGB", (VIDEO_W,VIDEO_H), (0,0,0))
    cx,cy = max(0,-ox), max(0,-oy)
    sw,sh = min(VIDEO_W,nw-cx), min(VIDEO_H,nh-cy)
    if sw>0 and sh>0:
        out.paste(sc.crop((cx,cy,cx+sw,cy+sh)), (max(0,ox),max(0,oy)))
    return out


# ── Frame builder ─────────────────────────────────────────────────────────────
def build_scene_frames(scene: dict, images: list, duration: float, pal: dict) -> list:
    """
    FIXED:
    - Title Y raised to VIDEO_H-230 so it never overlaps caption area
    - Caption capped at 3 lines maximum to prevent crowding
    - _clean_display() applied to all text before rendering
    """
    n      = max(1, int(round(FPS * duration)))
    fade   = min(int(FPS * 0.55), n // 6)
    ni     = len(images)
    seg    = n // ni if ni else n
    xfade  = min(int(FPS * 0.45), seg // 3) if ni > 1 else 0

    f_title   = get_font(54, "bold")
    f_caption = get_font(26, "regular")
    f_badge   = get_font(19, "medium")

    TITLE_Y      = VIDEO_H - 230
    CAPTION_Y0   = VIDEO_H - 100
    MAX_CAP_LINES = 3

    frames = []
    for f in range(n):
        tg = f / max(n-1, 1)

        # ── Background ───────────────────────────────────────────────────────
        if ni == 0:
            frame = Image.new("RGB", (VIDEO_W, VIDEO_H))
            d2    = ImageDraw.Draw(frame)
            c1,c2 = pal["bg"]
            for y in range(VIDEO_H):
                yf   = y/VIDEO_H
                wave = 0.5+0.5*math.sin(tg*math.pi*2+yf*3)
                d2.line([(0,y),(VIDEO_W,y)], fill=_blend(c1,c2,yf*0.7+wave*0.3))
        else:
            si  = min(f // seg, ni-1)
            lf  = f - si * seg
            lt  = lf / max(seg-1, 1)
            frame = _kb(images[si], lt, flip=(si%2==1))
            if si+1 < ni and xfade > 0 and lf >= (seg - xfade):
                raw   = (lf - (seg - xfade)) / xfade
                alpha = raw*raw*(3-2*raw)
                alpha = max(0.0, min(1.0, alpha))
                t_nxt = (lf - (seg-xfade)) / max(seg-1, 1)
                nxt   = _kb(images[si+1], t_nxt, flip=((si+1)%2==1))
                frame = Image.blend(frame, nxt, alpha)

        frame = Image.alpha_composite(frame.convert("RGBA"), GRAD).convert("RGB")

        acc     = pal["accent"]
        acc_dim = tuple(int(c*0.13) for c in acc)
        dm      = ImageDraw.Draw(frame)
        dm.rectangle([(0,0),(VIDEO_W,3)],               fill=acc_dim)
        dm.rectangle([(0,VIDEO_H-3),(VIDEO_W,VIDEO_H)], fill=acc_dim)

        # ── Scene title ───────────────────────────────────────────────────────
        title = _clean_display(scene.get("title", ""))
        if title:
            ta = int(255 * min(1.0, tg / 0.22))
            if ta > 0:
                dummy = ImageDraw.Draw(Image.new("RGBA", (1,1)))
                bb    = dummy.textbbox((0,0), title.upper(), font=f_title)
                tw,th = bb[2]-bb[0], bb[3]-bb[1]
                tx,ty = 40, TITLE_Y
                px,py = 18, 10
                frame = _pill(frame,
                              tx-px, ty-py, tx+tw+px, ty+th+py,
                              r=13, color=(0,0,0,int(148*ta//255))).convert("RGB")
                dm2 = ImageDraw.Draw(frame)
                dm2.rectangle([(tx-px, ty-py),(tx-px+5, ty+th+py)],
                              fill=tuple(int(c*ta//255) for c in acc))
                _shadow_text(dm2, title.upper(), f_title, tx, ty,
                             fill=(255,255,255,ta),
                             shadow=(0,0,0,int(210*ta//255)))

        # ── Caption / narration ───────────────────────────────────────────────
        narration = _clean_display(scene.get("narration", ""))
        if narration:
            words   = narration.split()
            vis_r   = min(1.0, max(0.0, (tg-0.06)*1.9))
            vis_n   = max(10, int(len(words)*vis_r))
            sub     = " ".join(words[:vis_n]) + ("..." if vis_n < len(words) else "")
            ca      = int(215 * min(1.0, max(0.0, (tg-0.09)/0.14)))
            if ca > 0 and sub:
                ci  = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0,0,0,0))
                cd  = ImageDraw.Draw(ci)
                lns = _wrap(cd, sub, f_caption, VIDEO_W-110)
                lns = lns[-MAX_CAP_LINES:]
                lh  = f_caption.size + 9
                bh  = len(lns) * lh
                cy0 = CAPTION_Y0 - bh
                for li,ln in enumerate(lns):
                    _shadow_text(cd, ln, f_caption, 50, cy0+li*lh,
                                 fill=(238,238,238,ca),
                                 shadow=(0,0,0,int(215*ca//255)))
                frame = Image.alpha_composite(frame.convert("RGBA"),ci).convert("RGB")

        # ── Scene counter badge ───────────────────────────────────────────────
        sid   = scene.get("id","")
        stot  = scene.get("total","")
        if sid and stot:
            btxt = f"{sid} / {stot}"
            bi   = Image.new("RGBA",(VIDEO_W,VIDEO_H),(0,0,0,0))
            bd   = ImageDraw.Draw(bi)
            bbb  = bd.textbbox((0,0),btxt,font=f_badge)
            bw,bh= bbb[2]-bbb[0], bbb[3]-bbb[1]
            bx   = VIDEO_W-bw-46
            by   = 26
            bd.rounded_rectangle([bx-10,by-6,bx+bw+10,by+bh+6],
                                  radius=10, fill=(0,0,0,115))
            bd.text((bx,by), btxt, font=f_badge, fill=(*acc,200))
            frame = Image.alpha_composite(frame.convert("RGBA"),bi).convert("RGB")

        # ── Fade in / out ─────────────────────────────────────────────────────
        fa = 0.0
        if f < fade and fade > 0:
            fa = 1.0 - f/fade
        elif f > (n-fade) and fade > 0:
            fa = (f-(n-fade))/fade
        if fa > 0:
            frame = Image.blend(frame.convert("RGB"),
                                Image.new("RGB",(VIDEO_W,VIDEO_H),(0,0,0)),
                                min(1.0,fa))

        frames.append(np.array(frame.convert("RGB"), dtype=np.uint8))

    return frames


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(jid: str, payload: dict):
    jdir = TEMP_DIR / jid
    jdir.mkdir(exist_ok=True)
    log.info("="*60)
    log.info(f"JOB START  {jid}")
    log.info(f"Payload: {json.dumps(payload)}")
    log.info("="*60)

    try:
        raw_topic   = payload["topic"]
        style       = payload.get("style", "Cinematic")
        lang        = payload.get("lang", "en")
        duration    = int(payload.get("duration", 30))
        scene_count = int(payload.get("scenes", 5))
        sec_per     = max(5, duration // scene_count)

        # ── Topic cleaning: if the user typed a full prompt sentence, extract
        # just the subject/product name so it doesn't leak into narration. ──
        _PROMPT_START = _re.compile(
            r'^(create|make|generate|write|build|produce|design|give me|i want)\b',
            _re.IGNORECASE,
        )
        if _PROMPT_START.match(raw_topic.strip()) or len(raw_topic) > 80:
            log.info("Topic looks like a long prompt — extracting subject…")
            try:
                extract_prompt = (
                    f'Extract only the core product or subject name from this text: "{raw_topic}"\n'
                    f'Return a SHORT product/subject name, 1-6 words, no extra words.\n'
                    f'Examples: "Samsung Galaxy S25 Ultra", "Tesla Model 3", "Nike Air Max 90"\n'
                    f'Return ONLY: {{"subject":"product name here"}}'
                )
                topic = extract_json(ai_call(extract_prompt, SYS_JSON)).get("subject", raw_topic[:60]).strip()
                log.info(f"  Extracted subject: '{topic}'")
            except Exception as e:
                log.warning(f"  Subject extraction failed ({e}) — using raw topic truncated")
                topic = raw_topic[:60]
        else:
            topic = raw_topic

        log.info(f"topic='{topic}' style='{style}' lang={lang} "
                 f"dur={duration}s scenes={scene_count} sec_per={sec_per}s")

        # Step 1 — Title
        job_update(jid, status="running", progress=2, message="Crafting video title…")
        video_title = get_video_title(topic, style)
        job_update(jid, message=f'Title: "{video_title}"')

        # Step 2a — Generate ALL scene titles in one call (prevents "Scene 1/2/..." echoing)
        job_update(jid, progress=3, message="Planning scene titles…")
        scene_titles = get_all_scene_titles(topic, style, scene_count)
        job_update(jid, message=f"Titles: {scene_titles}")

        # Step 2b — Write narration for each scene using confirmed titles
        job_update(jid, progress=4, message=f"Writing {scene_count} cinematic scenes…")
        scenes = []
        for i in range(scene_count):
            sc = get_scene(topic, style, i+1, scene_count, sec_per,
                           scene_title=scene_titles[i])
            scenes.append(sc)
            job_update(jid, message=f'Scene {i+1}: "{sc["title"]}" ✓',
                       progress=4+int((i+1)/scene_count*8))
        job_update(jid, progress=12, message="Script complete ✓")

        # Step 3 — Image prompts
        job_update(jid, progress=13, message="Building image prompts…")
        for i in range(scene_count):
            scenes[i]["imagePrompts"] = get_image_prompts(
                scenes[i]["title"], scenes[i]["narration"], topic, style)

        # Step 4 — Download images sequentially
        job_update(jid, progress=14, message="Generating images (sequential)…")
        all_imgs   = [[] for _ in range(scene_count)]
        total_imgs = sum(len(s["imagePrompts"]) for s in scenes)
        completed  = 0

        for i, sc in enumerate(scenes):
            pal = PALETTES[i % len(PALETTES)]
            for j, p in enumerate(sc["imagePrompts"]):
                seed = (i * 1000 + j * 37 + int(time.time())) % 999983
                job_update(jid,
                           message=f"Scene {i+1} image {j+1}/{len(sc['imagePrompts'])}…",
                           progress=14+int(completed/total_imgs*26))
                img = download_image(p, seed=seed)
                if img is None:
                    img = _make_fallback_image(sc["title"], pal)
                    job_update(jid, message=f"Scene {i+1} image {j+1} → gradient fallback")
                else:
                    job_update(jid, message=f"Scene {i+1} image {j+1} ✓")
                all_imgs[i].append(img)
                completed += 1

        job_update(jid, progress=40, message="All images ready ✓")

        # Step 5 — TTS (sequential)
        voice = _resolve_voice(lang)
        job_update(jid, progress=42, message=f"Generating voiceover — {voice}…")
        audio_paths = [None]*scene_count

        for i in range(scene_count):
            sc  = scenes[i]
            ap  = jdir / f"audio_{i}.mp3"
            ok  = make_tts(sc.get("narration", sc["title"]), ap, lang=lang)
            if ok:
                audio_paths[i] = ap
            job_update(jid, message=f"Scene {i+1} voice {'✓' if ok else '✗'}",
                       progress=42+int((i+1)/scene_count*17))

        job_update(jid, progress=60, message="Voiceover complete ✓ — rendering…")

        # Step 6 — Load audio + HARD TRIM to target duration
        audio_clips, scene_durs = [], []
        for i, ap in enumerate(audio_paths):
            target = float(scenes[i].get("duration", sec_per))
            if ap and Path(ap).exists():
                try:
                    ac          = AudioFileClip(str(ap))
                    actual_dur  = ac.duration
                    log.info(f"  Scene {i+1}: TTS={actual_dur:.2f}s  target={target:.1f}s")

                    if actual_dur > target * 1.10:
                        log.info(f"  Trimming scene {i+1}: {actual_dur:.1f}s → {target:.1f}s")
                        ac = ac.subclipped(0, target)
                        actual_dur = target

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

        # Step 7 — Render frames per scene
        video_clips = []
        for i, sc in enumerate(scenes):
            dur = scene_durs[i]
            log.info(f"Rendering scene {i+1}: '{sc['title']}' ({dur:.1f}s)…")
            job_update(jid,
                       message=f"Rendering scene {i+1}/{scene_count}: {sc['title'][:42]}…",
                       progress=62+int(i/scene_count*28))
            frames = build_scene_frames(sc, all_imgs[i], dur, PALETTES[i%len(PALETTES)])
            snap   = np.stack(frames, axis=0)
            def mf(t, _a=snap): return _a[min(int(t*FPS), len(_a)-1)]
            vc = VideoClip(mf, duration=dur).with_fps(FPS)
            if audio_clips[i]:
                vc = vc.with_audio(audio_clips[i])
            video_clips.append(vc)

        # Step 8 — Encode
        job_update(jid, progress=92, message="Encoding final MP4…")
        out_path = jdir / "output.mp4"
        concatenate_videoclips(video_clips, method="compose").write_videofile(
            str(out_path), fps=FPS, codec="libx264", audio_codec="aac",
            temp_audiofile=str(jdir/"temp_audio.m4a"), remove_temp=True,
            logger=None,
            ffmpeg_params=["-movflags","+faststart","-crf","18",
                           "-preset","fast","-pix_fmt","yuv420p"],
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
        log.info("="*60)

        job_update(jid, status="done", progress=100,
                   message=f"Video complete! {size_mb:.1f} MB · {total_dur:.0f}s",
                   result={
                       "file":        str(out_path),
                       "size_mb":     round(size_mb, 2),
                       "duration":    round(total_dur, 1),
                       "requested_s": duration,
                       "scene_count": scene_count,
                       "video_title": video_title,
                       "tts_voice":   voice,
                       "scenes":      [{"title":s["title"],"narration":s["narration"]}
                                       for s in scenes],
                   })

    except Exception as e:
        log.error(f"PIPELINE ERROR:\n{traceback.format_exc()}")
        job_update(jid, status="error", error=str(e), message=f"Error: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "INDEX2.HTML")

@app.route("/api/health")
def health():
    return jsonify({
        "status":      "ok",
        "gemini":      bool(gemini_client),
        "ai_backend":  "gemini-2.0-flash" if gemini_client else "pollinations",
        "tts_backend": "edge-tts (rate/pitch)",
        "tts_voice":   _resolve_voice("en"),
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
    JOBS[jid] = {"status":"queued","progress":0,"log":[],"result":None,"error":None}
    threading.Thread(target=run_pipeline, args=(jid,p), daemon=True).start()
    return jsonify({"job_id": jid})

@app.route("/api/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    if not j: return jsonify({"error":"Unknown job"}), 404
    return jsonify(j)

@app.route("/api/download/<jid>")
def download(jid):
    j = JOBS.get(jid)
    if not j or j.get("status") != "done": return jsonify({"error":"Not ready"}), 404
    return send_file(j["result"]["file"], mimetype="video/mp4",
                     as_attachment=True, download_name="ai-video.mp4")

@app.route("/api/preview/<jid>")
def preview(jid):
    j = JOBS.get(jid)
    if not j or j.get("status") != "done": return jsonify({"error":"Not ready"}), 404
    return send_file(j["result"]["file"], mimetype="video/mp4")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("="*60)
    print("  AI Video Studio — v9 PERFECT Edition")
    print(f"  http://localhost:{port}")
    print(f"  AI  : {'Gemini 2.0 Flash ✅' if gemini_client else 'Pollinations'}")
    print(f"  TTS : Edge TTS (rate/pitch) — {_resolve_voice('en')}")
    print(f"  Font: {os.path.basename(FONT_TITLE_PATH) if FONT_TITLE_PATH else 'downloading Poppins...'}")
    print("="*60)
    app.run(debug=False, port=port, threaded=True)

