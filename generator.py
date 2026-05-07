"""
AI Video Studio — Python Backend
=================================
Requirements (pip install):
    flask flask-cors requests gtts moviepy pillow numpy

Run:
    python app.py

Then open index.html (served at http://localhost:5000) or just open the HTML file directly.
The HTML frontend calls this backend at http://localhost:5000/api/...
"""

import os
import io
import json
import uuid
import time
import math
import textwrap
import threading
import traceback
import tempfile
import shutil
import urllib.request
import urllib.parse
from pathlib import Path

import requests
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from moviepy import (
    VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip,
    concatenate_videoclips, CompositeAudioClip, AudioArrayClip
)
from moviepy import vfx

# ── Config ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)

JOBS: dict[str, dict] = {}          # job_id → {status, progress, message, result}
TEMP_DIR = Path(tempfile.gettempdir()) / "ai_video_studio"
TEMP_DIR.mkdir(exist_ok=True)

VIDEO_W, VIDEO_H = 1280, 720
FPS = 24


# ── Helpers ──────────────────────────────────────────────────────────────────

def job_update(jid, status=None, progress=None, message=None, result=None, error=None):
    j = JOBS.setdefault(jid, {})
    if status:   j["status"]   = status
    if progress is not None: j["progress"] = progress
    if message:  j["log"] = j.get("log", []) + [{"ts": time.strftime("%H:%M:%S"), "msg": message}]
    if result:   j["result"]   = result
    if error:    j["error"]    = error


from typing import Optional

def download_image(prompt: str, seed: int = None) -> Optional[Image.Image]:
    """Fetch an image from Pollinations AI."""
    if seed is None:
        seed = int(time.time() * 1000) % 99999
    enc = urllib.parse.quote(prompt + ", cinematic, photorealistic, 16:9, high quality")
    url = f"https://image.pollinations.ai/prompt/{enc}?width={VIDEO_W}&height={VIDEO_H}&seed={seed}&nologo=true"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB").resize((VIDEO_W, VIDEO_H))
    except Exception as e:
        print(f"[image] failed: {e}")
        return None


def call_ai(prompt: str) -> str:
    """Call Pollinations text API."""
    enc = urllib.parse.quote(prompt)
    url = f"https://text.pollinations.ai/{enc}?model=openai&seed={int(time.time()) % 9999}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def make_tts(text: str, path: Path, lang: str = "en") -> bool:
    """Generate TTS audio with gTTS and save to path."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(str(path))
        return True
    except Exception as e:
        print(f"[tts] failed: {e}")
        return False


# ── Visual Frame Builders ────────────────────────────────────────────────────

PALETTES = [
    {"bg": [(10, 20, 40),  (26, 58, 92)],  "accent": (79, 195, 247)},
    {"bg": [(13, 31, 45),  (26, 77, 60)],  "accent": (74, 222, 128)},
    {"bg": [(45, 13, 26),  (77, 26, 45)],  "accent": (244, 114, 182)},
    {"bg": [(26, 13, 45),  (45, 26, 77)],  "accent": (167, 139, 250)},
    {"bg": [(45, 26, 13),  (77, 58, 26)],  "accent": (251, 191, 36)},
    {"bg": [(10, 10, 30),  (30, 10, 50)],  "accent": (255, 100, 100)},
    {"bg": [(5,  35, 25),  (5,  55, 40)],  "accent": (100, 255, 180)},
]


def blend(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_text_wrapped(draw, text, font, x, y, max_width, fill, shadow_color=None, line_spacing=8):
    """Draw word-wrapped text with optional drop shadow."""
    words = text.split()
    lines, line = [], []
    for w in words:
        test = " ".join(line + [w])
        bb = draw.textbbox((0, 0), test, font=font)
        if bb[2] - bb[0] <= max_width or not line:
            line.append(w)
        else:
            lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))

    for i, ln in enumerate(lines):
        ly = y + i * (font.size + line_spacing)
        if shadow_color:
            draw.text((x + 2, ly + 2), ln, font=font, fill=shadow_color)
        draw.text((x, ly), ln, font=font, fill=fill)
    return len(lines)


def get_font(size: int, bold: bool = False):
    """Try to load a system font, fall back to default."""
    candidates = []
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_scene_frames(
    scene: dict,
    images: list[Image.Image],   # 1-3 images for this scene
    duration: float,
    scene_idx: int,
    total_scenes: int,
    total_duration: float,
    elapsed_start: float,
    pal: dict,
) -> list[np.ndarray]:
    """
    Render all frames for one scene and return as list of numpy arrays.
    Uses Ken Burns zoom+pan on each image, smooth crossfades between images,
    and rich overlay (title, subtitle, progress bar, accent bars).
    """
    n_frames = max(1, int(round(FPS * duration)))
    frames = []

    # How many images to cycle through
    n_imgs = len(images)
    # Each sub-image segment length
    seg_len = duration / n_imgs if n_imgs > 0 else duration

    font_title = get_font(52, bold=True)
    font_sub   = get_font(22, bold=False)
    font_scene = get_font(14, bold=False)

    fade_frames = min(int(FPS * 0.6), n_frames // 6)   # 0.6s fade at edges

    for f in range(n_frames):
        t = f / max(n_frames - 1, 1)   # 0→1 within scene
        elapsed = elapsed_start + f / FPS

        # Which image segment are we in?
        img_idx  = min(int(f / FPS / seg_len), n_imgs - 1) if n_imgs > 0 else 0
        img      = images[img_idx] if images else None

        # Local time within this image segment (0→1)
        seg_start_f = int(img_idx * seg_len * FPS)
        seg_f       = f - seg_start_f
        seg_total   = max(1, int(seg_len * FPS))
        lt          = seg_f / seg_total        # local t for Ken Burns

        # ── Base image with Ken Burns ────────────────────────────────
        frame = Image.new("RGB", (VIDEO_W, VIDEO_H))
        draw  = ImageDraw.Draw(frame)

        if img:
            # Ken Burns: alternate zoom directions per image
            if img_idx % 2 == 0:
                z0, z1 = 1.0, 1.18
                px_dir = 1
            else:
                z0, z1 = 1.18, 1.0
                px_dir = -1

            z   = z0 + (z1 - z0) * lt
            pan = math.sin(lt * math.pi) * 35 * px_dir
            tilt = math.cos(lt * math.pi * 0.7) * 20

            nw = int(VIDEO_W * z)
            nh = int(VIDEO_H * z)
            ox = int((VIDEO_W - nw) / 2 + pan)
            oy = int((VIDEO_H - nh) / 2 + tilt)

            scaled = img.resize((nw, nh), Image.LANCZOS)
            # Crop to canvas
            cx = max(0, -ox);  cy = max(0, -oy)
            sw = min(VIDEO_W, nw - cx);  sh = min(VIDEO_H, nh - cy)
            dx = max(0, ox);   dy = max(0, oy)
            region = scaled.crop((cx, cy, cx + sw, cy + sh))
            frame.paste(region, (dx, dy))

            # Subtle crossfade between images
            XFADE = int(FPS * 0.4)   # 0.4s crossfade
            if img_idx + 1 < n_imgs and seg_f > seg_total - XFADE:
                next_img = images[img_idx + 1]
                alpha_x  = (seg_f - (seg_total - XFADE)) / XFADE
                next_frame = Image.new("RGB", (VIDEO_W, VIDEO_H))
                nz = 1.0
                ns = next_img.resize((int(VIDEO_W * nz), int(VIDEO_H * nz)), Image.LANCZOS)
                next_frame.paste(ns, (int((VIDEO_W - ns.width)//2), int((VIDEO_H - ns.height)//2)))
                frame = Image.blend(frame, next_frame, alpha_x)
                draw  = ImageDraw.Draw(frame)

        else:
            # Animated gradient fallback
            c1 = pal["bg"][0]
            c2 = pal["bg"][1]
            for y in range(VIDEO_H):
                yf = y / VIDEO_H
                wave = 0.5 + 0.5 * math.sin(t * math.pi * 2 + yf * 3)
                c = blend(c1, c2, yf * 0.7 + wave * 0.3)
                draw.line([(0, y), (VIDEO_W, y)], fill=c)

        draw = ImageDraw.Draw(frame)

        # ── Cinematic gradient overlay ───────────────────────────────
        overlay = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
        odraw   = ImageDraw.Draw(overlay)
        # Bottom vignette (for subtitle readability)
        for y in range(VIDEO_H):
            yf = y / VIDEO_H
            if yf > 0.45:
                alpha = int(min(200, (yf - 0.45) / 0.55 * 220))
                odraw.line([(0, y), (VIDEO_W, y)], fill=(0, 0, 0, alpha))
        # Top vignette (for scene label)
        for y in range(min(120, VIDEO_H)):
            alpha = int((1 - y / 120) * 160)
            odraw.line([(0, y), (VIDEO_W, y)], fill=(0, 0, 0, alpha))
        frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        draw  = ImageDraw.Draw(frame)

        # ── Accent bars (top & bottom) ───────────────────────────────
        acc = pal["accent"]
        bar_alpha = 0.15 + 0.1 * math.sin(t * math.pi * 4)
        acc_bar = tuple(int(c * bar_alpha) for c in acc)
        draw.rectangle([(0, 0), (VIDEO_W, 5)], fill=acc_bar)
        draw.rectangle([(0, VIDEO_H - 5), (VIDEO_W, VIDEO_H)], fill=acc_bar)

        # ── Scene label (top-left) ───────────────────────────────────
        label_bg = (0, 0, 0, 140)
        label_frame = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(label_frame)
        ldraw.rounded_rectangle([(16, 14), (190, 46)], radius=6, fill=label_bg)
        frame_rgba = frame.convert("RGBA")
        frame_rgba = Image.alpha_composite(frame_rgba, label_frame)
        frame = frame_rgba.convert("RGB")
        draw = ImageDraw.Draw(frame)
        draw.text((26, 20), f"Scene {scene_idx + 1} / {total_scenes}", font=font_scene, fill=acc)

        # ── Progress bar (top-right) ─────────────────────────────────
        BAR_X, BAR_Y, BAR_W, BAR_H = VIDEO_W - 170, 24, 150, 4
        pct = min(1.0, elapsed / total_duration)
        draw.rounded_rectangle([(BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H)], radius=2, fill=(255, 255, 255, 30))
        if pct > 0:
            draw.rounded_rectangle([(BAR_X, BAR_Y), (BAR_X + int(BAR_W * pct), BAR_Y + BAR_H)], radius=2, fill=acc)

        # ── Accent line decoration ───────────────────────────────────
        line_len = int(180 + 60 * math.sin(t * math.pi * 2))
        line_y   = VIDEO_H - 130
        line_alpha = int(255 * (0.5 + 0.3 * math.sin(t * math.pi * 2)))
        acc_line = acc + (line_alpha,)
        line_frame = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
        ldraw2 = ImageDraw.Draw(line_frame)
        ldraw2.line([(30, line_y), (30 + line_len, line_y - 4)], fill=acc_line, width=3)
        frame = Image.alpha_composite(frame.convert("RGBA"), line_frame).convert("RGB")
        draw  = ImageDraw.Draw(frame)

        # ── Scene title ──────────────────────────────────────────────
        title = scene.get("title", "")
        if title:
            title_t = min(1.0, max(0.0, (t - 0.0) / 0.2))  # fade in first 20%
            title_alpha = int(255 * title_t)
            # Drop shadow
            shadow_col = (0, 0, 0, min(255, title_alpha))
            title_col  = (255, 255, 255, title_alpha)

            title_img = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
            td = ImageDraw.Draw(title_img)
            # Shadow
            draw_text_wrapped(td, title, font_title, 32 + 3, VIDEO_H - 150 + 3,
                               VIDEO_W - 80, (0, 0, 0, min(255, title_alpha)))
            # Main text
            draw_text_wrapped(td, title, font_title, 32, VIDEO_H - 150,
                               VIDEO_W - 80, title_col)
            frame = Image.alpha_composite(frame.convert("RGBA"), title_img).convert("RGB")
            draw  = ImageDraw.Draw(frame)

        # ── Subtitle (narration preview) ─────────────────────────────
        narration = scene.get("narration", "")
        if narration:
            words = narration.split()
            # Show progressively more words as scene progresses
            n_visible = max(8, int(len(words) * min(1.0, t * 2.2)))
            subtitle  = " ".join(words[:n_visible])
            if n_visible < len(words):
                subtitle += "…"

            sub_t = min(1.0, max(0.0, (t - 0.08) / 0.15))
            sub_alpha = int(200 * sub_t)

            sub_img = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sub_img)
            draw_text_wrapped(sd, subtitle, font_sub, 32 + 2, VIDEO_H - 88 + 2,
                               VIDEO_W - 80, (0, 0, 0, min(255, sub_alpha)))
            draw_text_wrapped(sd, subtitle, font_sub, 32, VIDEO_H - 88,
                               VIDEO_W - 80, (230, 230, 230, sub_alpha))
            frame = Image.alpha_composite(frame.convert("RGBA"), sub_img).convert("RGB")

        # ── Fade in/out at scene edges ───────────────────────────────
        fade_alpha = 0
        if f < fade_frames and fade_frames > 0:
            fade_alpha = 1.0 - f / fade_frames
        elif f > n_frames - fade_frames and fade_frames > 0:
            fade_alpha = (f - (n_frames - fade_frames)) / fade_frames
        if fade_alpha > 0:
            black = Image.new("RGB", (VIDEO_W, VIDEO_H), (0, 0, 0))
            frame = Image.blend(frame.convert("RGB"), black, min(1.0, fade_alpha))

        frames.append(np.array(frame.convert("RGB")))

    return frames


# ── Core Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(jid: str, payload: dict):
    """Full async pipeline: script → images → TTS → video."""
    jdir = TEMP_DIR / jid
    jdir.mkdir(exist_ok=True)
    try:
        topic      = payload["topic"]
        style      = payload.get("style", "Promotional / Marketing")
        duration   = int(payload.get("duration", 30))
        scene_count = int(payload.get("scenes", 5))

        # ── Step 1: Generate Script ──────────────────────────────────
        job_update(jid, status="running", progress=2, message="Requesting AI script…")

        script_prompt = f"""You are a world-class video director. Create a video production plan.

Brief: "{topic}"
Style: {style}
Total duration: {duration} seconds
Number of scenes: {scene_count}

Respond with ONLY a valid JSON object. No markdown. No backticks. Start with {{ end with }}.

{{
  "videoTitle": "Punchy title here",
  "concept": "Two sentence concept.",
  "scenes": [
    {{
      "id": 1,
      "title": "Scene Title Here",
      "duration": 5,
      "narration": "The voiceover text spoken here. Two to three natural sentences.",
      "imagePrompts": [
        "Detailed cinematic description: subject, setting, lighting, mood, camera angle. No text in image.",
        "Second cinematic shot of the same scene from a different angle, photorealistic.",
        "Third cinematic shot: close-up detail, dramatic lighting."
      ]
    }}
  ]
}}

Requirements:
- durations must sum to exactly {duration}
- Return exactly {scene_count} scenes
- Each scene must have exactly 3 imagePrompts (varied angles/compositions)
- JSON only"""

        raw = call_ai(script_prompt)
        job_update(jid, message="Parsing AI response…")

        plan = None
        for fn in [
            lambda: json.loads(raw.strip()),
            lambda: json.loads(raw.replace("```json", "").replace("```", "").strip()),
            lambda: json.loads(raw[raw.index("{"):raw.rindex("}") + 1]),
        ]:
            try:
                plan = fn();  break
            except Exception:
                pass

        if not plan or not plan.get("scenes"):
            raise ValueError("Could not parse AI script. Please try again.")

        job_update(jid, progress=10, message=f"Script ready: \"{plan.get('videoTitle', '')}\" · {len(plan['scenes'])} scenes")

        scenes = plan["scenes"]

        # ── Step 2: Download Images (3 per scene) ────────────────────
        job_update(jid, progress=12, message="Fetching scene images…")
        all_scene_images: list[list[Image.Image]] = []

        for i, sc in enumerate(scenes):
            prompts = sc.get("imagePrompts", [sc.get("imagePrompt", sc["title"])])
            if isinstance(prompts, str):
                prompts = [prompts]
            # Ensure we always request 3 images
            while len(prompts) < 3:
                prompts.append(prompts[-1] + ", different angle")

            scene_imgs = []
            for j, prompt in enumerate(prompts[:3]):
                job_update(jid, message=f"Scene {i+1} image {j+1}/3: {sc['title'][:30]}…")
                img = download_image(prompt, seed=i * 100 + j)
                if img:
                    scene_imgs.append(img)
                    job_update(jid, message=f"Scene {i+1} image {j+1} ✓")
                else:
                    job_update(jid, message=f"Scene {i+1} image {j+1} failed, using fallback")

            all_scene_images.append(scene_imgs)
            pct = 12 + int((i + 1) / len(scenes) * 28)
            job_update(jid, progress=pct)

        job_update(jid, progress=40, message="All images ready ✓")

        # ── Step 3: Generate TTS Audio ───────────────────────────────
        job_update(jid, progress=42, message="Generating voiceover with gTTS…")
        audio_paths: list[Path | None] = []

        for i, sc in enumerate(scenes):
            narration = sc.get("narration", sc["title"])
            apath = jdir / f"audio_{i}.mp3"
            job_update(jid, message=f"Scene {i+1}: recording narration…")
            ok = make_tts(narration, apath)
            if ok:
                audio_paths.append(apath)
                size_kb = apath.stat().st_size // 1024
                job_update(jid, message=f"Scene {i+1} audio ready ({size_kb} KB) ✓")
            else:
                audio_paths.append(None)
                job_update(jid, message=f"Scene {i+1} audio failed", )

            pct = 42 + int((i + 1) / len(scenes) * 18)
            job_update(jid, progress=pct)

        job_update(jid, progress=60, message="Voice done ✓ — starting render…")

        # ── Step 4: Render Video ─────────────────────────────────────
        job_update(jid, progress=62, message="Rendering video frames…")

        # Load audio clips and determine actual durations
        audio_clips: list[AudioFileClip | None] = []
        scene_durations: list[float] = []

        for i, apath in enumerate(audio_paths):
            if apath and apath.exists():
                try:
                    ac = AudioFileClip(str(apath))
                    audio_clips.append(ac)
                    # Add 0.5s buffer after each narration
                    scene_durations.append(ac.duration + 0.5)
                except Exception as e:
                    print(f"[audio] clip error scene {i}: {e}")
                    audio_clips.append(None)
                    scene_durations.append(float(scenes[i].get("duration", 6)))
            else:
                audio_clips.append(None)
                scene_durations.append(float(scenes[i].get("duration", 6)))

        total_dur = sum(scene_durations)
        job_update(jid, message=f"Total video duration: {total_dur:.1f}s at {FPS}fps")

        # Build video clips per scene
        video_clips = []
        elapsed = 0.0

        for i, sc in enumerate(scenes):
            sc_dur = scene_durations[i]
            pal    = PALETTES[i % len(PALETTES)]
            imgs   = all_scene_images[i]

            job_update(jid, message=f"Rendering scene {i+1}/{len(scenes)}: {sc['title'][:30]}…")

            frames = build_scene_frames(
                scene=sc,
                images=imgs,
                duration=sc_dur,
                scene_idx=i,
                total_scenes=len(scenes),
                total_duration=total_dur,
                elapsed_start=elapsed,
                pal=pal,
            )

            elapsed += sc_dur

            # frames → moviepy ImageClip sequence → video
            # Use make_frame approach for efficiency
            frame_arr = np.stack(frames, axis=0)   # (N, H, W, 3)

            def make_frame(t, _arr=frame_arr, _dur=sc_dur):
                idx = min(int(t * FPS), len(_arr) - 1)
                return _arr[idx]

            from moviepy import VideoClip
            vc = VideoClip(make_frame, duration=sc_dur).with_fps(FPS)

            # Attach audio
            if audio_clips[i] is not None:
                ac = audio_clips[i]
                # If audio is shorter than scene, that's fine; if longer, trim scene
                vc = vc.with_audio(ac)

            video_clips.append(vc)

            pct = 62 + int((i + 1) / len(scenes) * 33)
            job_update(jid, progress=pct)

        job_update(jid, progress=95, message="Concatenating scenes & encoding…")

        final = concatenate_videoclips(video_clips, method="compose")

        out_path = jdir / "output.mp4"
        final.write_videofile(
            str(out_path),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(jdir / "temp_audio.m4a"),
            remove_temp=True,
            logger=None,
            ffmpeg_params=["-movflags", "+faststart"],
        )

        # Close clips
        for c in video_clips:
            try: c.close()
            except: pass
        for ac in audio_clips:
            try:
                if ac: ac.close()
            except: pass

        size_mb = out_path.stat().st_size / 1024 / 1024
        job_update(
            jid,
            status="done",
            progress=100,
            message=f"Video complete! {size_mb:.1f} MB · {total_dur:.0f}s",
            result={"file": str(out_path), "size_mb": round(size_mb, 2), "duration": round(total_dur, 1)},
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        job_update(jid, status="error", error=str(e), message=f"Error: {e}")


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "INDEX2.HTML")


@app.route("/api/generate", methods=["POST"])
def generate():
    payload = request.get_json(force=True)
    jid = str(uuid.uuid4())
    JOBS[jid] = {"status": "queued", "progress": 0, "log": [], "result": None, "error": None}
    t = threading.Thread(target=run_pipeline, args=(jid, payload), daemon=True)
    t.start()
    return jsonify({"job_id": jid})


@app.route("/api/status/<jid>")
def status(jid):
    j = JOBS.get(jid)
    if not j:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(j)


@app.route("/api/download/<jid>")
def download(jid):
    j = JOBS.get(jid)
    if not j or j.get("status") != "done":
        return jsonify({"error": "Not ready"}), 404
    path = j["result"]["file"]
    return send_file(path, mimetype="video/mp4", as_attachment=True, download_name="ai-video.mp4")


@app.route("/api/preview/<jid>")
def preview(jid):
    j = JOBS.get(jid)
    if not j or j.get("status") != "done":
        return jsonify({"error": "Not ready"}), 404
    path = j["result"]["file"]
    return send_file(path, mimetype="video/mp4")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Video Studio — Python Backend")
    print("  http://localhost:5000")
    print()
    print("  Install deps first:")
    print("  pip install flask flask-cors requests gtts moviepy pillow numpy")
    print("=" * 60)
    app.run(debug=False, port=5000, threaded=True)