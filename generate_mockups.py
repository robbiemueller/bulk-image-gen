"""
generate_mockups.py

Opens a GUI to select an input folder, output folder, and prompt, then sends
each image to the selected provider's image-generation API (Google Gemini or
OpenAI) and saves the results.

Requirements:
    - Gemini: billing enabled on Google AI Studio (no free tier for image gen)
    - OpenAI: an account with image-generation access

Setup:
    1. Enable Gemini billing (if using Gemini): https://aistudio.google.com/plan_information
    2. Set the API key for whichever provider(s) you'll use:
           Windows (PowerShell):
               $env:GOOGLE_API_KEY = "your_gemini_key_here"
               $env:OPENAI_API_KEY = "your_openai_key_here"
           macOS/Linux:
               export GOOGLE_API_KEY="your_gemini_key_here"
               export OPENAI_API_KEY="your_openai_key_here"
    3. Run:
           python generate_mockups.py
"""

import base64
import io
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    import keyring
    import keyring.errors
    _KEYRING_AVAILABLE = True
except ImportError:
    _KEYRING_AVAILABLE = False

from google import genai
from google.genai import types

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

from PIL import Image, ImageOps, ImageTk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

# Formats both APIs accept as-is. Anything else (BMP/TIFF/GIF) is converted to
# PNG in memory before it's sent.
NATIVE_INPUT_MIME = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}

# Inputs bigger than this (file size or longest edge) are downscaled before
# sending — keeps requests inside the APIs' size limits and speeds up uploads.
MAX_INPUT_BYTES = 15 * 1024 * 1024
MAX_INPUT_EDGE = 4096

# Optional reference images sent with every request (e.g. a room scene).
# Gemini takes up to 14 images per request and OpenAI up to 16, so 10 extras
# plus the input image stays inside both limits.
MAX_REFERENCE_IMAGES = 10

# --- Providers ---
PROVIDERS = ["Gemini", "OpenAI"]
DEFAULT_PROVIDER = "Gemini"

# --- Gemini sizing controls ---
GEMINI_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "2:3", "3:2", "21:9"]
GEMINI_EXTREME_RATIOS = ["1:4", "4:1", "1:8", "8:1"]   # Nano Banana 2 / 2.1 only
DEFAULT_ASPECT_RATIO = "1:1"

# "512" is the 0.5K tier — note: no "K" suffix (API requirement).
GEMINI_SIZES = ["512", "1K", "2K", "4K"]
DEFAULT_IMAGE_SIZE = "1K"

# --- OpenAI sizing controls ---
# gpt-image-2 and newer accept any WIDTHxHEIGHT that passes
# validate_openai_size(); these are just presets for the (editable) dropdown.
# Source: https://developers.openai.com/api/docs/guides/image-generation
OPENAI_SIZES = [
    "auto", "1024x1024", "1536x1024", "1024x1536", "2048x2048",
    "2048x1152", "1152x2048", "2560x1440", "1440x2560", "3840x2160", "2160x3840",
]
DEFAULT_OPENAI_SIZE = "1024x1024"

OPENAI_QUALITIES = ["auto", "low", "medium", "high"]
OPENAI_QUALITIES_25 = OPENAI_QUALITIES + ["xhigh", "max"]   # 2.5 models only
DEFAULT_OPENAI_QUALITY = "auto"

# USD per 1M tokens — the same for gpt-image-2 and both 2.5 models.
# Source: https://developers.openai.com/api/docs/pricing (Oct 2026)
_GPT_IMAGE_TOKEN_PRICES = {"text_in": 5.00, "image_in": 8.00, "image_out": 30.00}

# --- Model registry ---
# One entry per built-in model. Dropdowns, hints, the run-mode lock and cost
# estimates are all built from this table, so adding a model is one entry.
#   sizes / ratios       Gemini resolutions + aspect ratios
#   sizes / qualities    OpenAI size presets + quality levels
#   batch                Batch API (50% off) available
#   cost                 USD per image — Gemini by size, OpenAI by "{size}/{quality}"
#   input_per_m          Gemini input price per 1M tokens (actual-spend tracking)
#   token_prices         OpenAI per-1M-token prices (actual-spend tracking)
# Sources: https://ai.google.dev/gemini-api/docs/pricing
#          https://developers.openai.com/api/docs/guides/image-generation (Oct 2026)
MODEL_SPECS = {
    "gemini-nano-banana-2.1": {
        "provider": "Gemini", "blurb": "Nano Banana 2.1 · newest, fast",
        "sizes": ["1K", "2K", "4K"], "ratios": GEMINI_RATIOS + GEMINI_EXTREME_RATIOS,
        "batch": True, "input_per_m": 1.50,
        "cost": {"1K": 0.0336, "2K": 0.0504, "4K": 0.113},
    },
    "gemini-3-pro-image": {
        "provider": "Gemini", "blurb": "Nano Banana Pro · highest quality, slower",
        "sizes": ["1K", "2K", "4K"], "ratios": GEMINI_RATIOS,
        "batch": True, "input_per_m": 2.00,
        "cost": {"1K": 0.134, "2K": 0.134, "4K": 0.24},
    },
    "gemini-3.1-flash-image": {
        "provider": "Gemini", "blurb": "Nano Banana 2 · previous gen, has 512",
        "sizes": GEMINI_SIZES, "ratios": GEMINI_RATIOS + GEMINI_EXTREME_RATIOS,
        "batch": True, "input_per_m": 0.50,
        "cost": {"512": 0.045, "1K": 0.067, "2K": 0.101, "4K": 0.151},
    },
    "gemini-3.1-flash-lite-image": {
        "provider": "Gemini", "blurb": "Nano Banana 2 Lite · fastest, 1K only",
        "sizes": ["1K"], "ratios": GEMINI_RATIOS,
        "batch": True, "input_per_m": 0.25,
        "cost": {"1K": 0.0336},
    },
    "gpt-image-2.5-sunburst": {
        "provider": "OpenAI", "blurb": "most precise edits · keeps artwork exact",
        "sizes": OPENAI_SIZES, "qualities": OPENAI_QUALITIES_25,
        "batch": True, "token_prices": _GPT_IMAGE_TOKEN_PRICES,
        "cost": {},   # no per-image table published yet — billed by tokens
    },
    "gpt-image-2.5-flare": {
        "provider": "OpenAI", "blurb": "faster everyday generation",
        "sizes": OPENAI_SIZES, "qualities": OPENAI_QUALITIES_25,
        "batch": True, "token_prices": _GPT_IMAGE_TOKEN_PRICES,
        "cost": {},
    },
    "gpt-image-2": {
        "provider": "OpenAI", "blurb": "previous generation",
        "sizes": OPENAI_SIZES, "qualities": OPENAI_QUALITIES,
        "batch": False, "token_prices": _GPT_IMAGE_TOKEN_PRICES,
        "cost": {
            "1024x1024/low": 0.006, "1024x1024/medium": 0.053, "1024x1024/high": 0.211,
            "1024x1536/low": 0.005, "1024x1536/medium": 0.041, "1024x1536/high": 0.165,
            "1536x1024/low": 0.005, "1536x1024/medium": 0.041, "1536x1024/high": 0.165,
        },
    },
}
MODELS_BY_PROVIDER = {
    p: [m for m, spec in MODEL_SPECS.items() if spec["provider"] == p] for p in PROVIDERS
}
DEFAULT_MODEL = MODELS_BY_PROVIDER[DEFAULT_PROVIDER][0]

# Retired IDs with a drop-in replacement — saved settings are migrated on load.
MODEL_MIGRATIONS = {
    "gemini-3-pro-image-preview":     "gemini-3-pro-image",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image",
    "nano-banana-pro-preview":        "gemini-3-pro-image",
}
# Still callable as a custom model ID, but on the way out — shown as a warning.
MODEL_SHUTDOWNS = {
    "gpt-image-1":            "shuts down Oct 23, 2026",
    "gpt-image-1.5":          "shuts down Dec 1, 2026",
    "gpt-image-1-mini":       "shuts down Dec 1, 2026",
    "chatgpt-image-latest":   "shuts down Dec 1, 2026",
    "gemini-2.5-flash-image": "is deprecated (shutdown listed between Oct 2026 and Mar 2027)",
}

# --- Output ---
# Display name → (file extension, MIME type, OpenAI output_format value)
OUTPUT_FORMATS = {
    "PNG":  ("png",  "image/png",  "png"),
    "JPEG": ("jpg",  "image/jpeg", "jpeg"),
    "WebP": ("webp", "image/webp", "webp"),
}
DEFAULT_OUTPUT_FORMAT = "PNG"
MAX_VARIATIONS = 4
MAX_PARALLEL = 6
DEFAULT_PARALLEL = 3
PREVIEW_SIZE = 200          # px, the latest-result thumbnail
TEST_SUBFOLDER = "_test"    # "Test 1 image" writes here so real runs don't skip it

DEFAULT_PROMPT = (
    "Create a high-resolution, photorealistic interior product mockup featuring the provided artwork. "
    "The artwork is displayed identically inside three landscape-oriented frames resting on a rustic, "
    "wide-plank wood floor and leaning against a minimalistic, warm off-white wall with a subtle "
    "baseboard. Arrange the frames in a staggered, overlapping line from left to right: the leftmost "
    "frame is dark walnut wood (positioned furthest back), the center frame is natural light oak wood "
    "(slightly overlapping the left frame), and the rightmost frame is bright white wood (positioned "
    "at the front, slightly overlapping the center frame). Use soft, natural window lighting from the "
    "side to cast a realistic glass reflection and glare over the rightmost white frame. Warm, "
    "inviting, mid-century modern interior photography style with a shallow depth of field."
)

# Minimum gap between request starts, shared by all parallel workers
MIN_REQUEST_SPACING = 1.0

# Retry settings for rate-limit (429) and service-unavailable (5xx) errors
MAX_RETRIES = 5
RETRY_BASE_DELAY = 60

# Circuit breaker: stop the run after this many consecutive items fail with 503
CIRCUIT_BREAKER_THRESHOLD = 5

# Batch API settings
BATCH_POLL_INTERVAL = 30   # seconds between status polls

# Run mode options
RUN_MODES = ["Real-time", "Batch (50% off)"]

# Sentinel value for the prompt-set dropdown
_SINGLE_PROMPT = "-- Single prompt --"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(folder: Path) -> list[Path]:
    """Return all supported image files in folder (non-recursive)."""
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def infer_provider(model: str) -> str | None:
    """Which provider a model ID belongs to (None if it can't be told)."""
    if model in MODEL_SPECS:
        return MODEL_SPECS[model]["provider"]
    m = model.lower()
    if m.startswith(("gpt-", "chatgpt-", "dall-e")):
        return "OpenAI"
    if m.startswith(("gemini", "imagen", "nano-banana")):
        return "Gemini"
    return None


def model_spec(model: str, provider: str) -> dict:
    """Spec for *model*, or a permissive generic one for a custom model ID."""
    if model in MODEL_SPECS:
        return MODEL_SPECS[model]
    if provider == "OpenAI":
        return {"provider": "OpenAI", "blurb": "custom model ID", "sizes": OPENAI_SIZES,
                "qualities": OPENAI_QUALITIES_25, "batch": False, "cost": {}}
    return {"provider": "Gemini", "blurb": "custom model ID", "sizes": GEMINI_SIZES,
            "ratios": GEMINI_RATIOS + GEMINI_EXTREME_RATIOS, "batch": True, "cost": {}}


def price_hint(spec: dict) -> str:
    costs = list((spec.get("cost") or {}).values())
    if costs:
        lo, hi = min(costs), max(costs)
        return f"${lo:.3f}/image" if lo == hi else f"${lo:.3f}–${hi:.3f}/image"
    if spec.get("token_prices"):
        return "billed by tokens"
    return "pricing unknown"


def validate_openai_size(size: str) -> str | None:
    """Return an error message if *size* isn't valid for gpt-image-2+, else None."""
    if size == "auto":
        return None
    match = re.fullmatch(r"(\d+)x(\d+)", size)
    if not match:
        return "use WIDTHxHEIGHT (e.g. 1536x1024) or 'auto'"
    w, h = int(match.group(1)), int(match.group(2))
    if w % 16 or h % 16:
        return "width and height must be multiples of 16"
    if max(w, h) > 3840:
        return "no edge can be larger than 3840 px"
    if max(w, h) > 3 * min(w, h):
        return "aspect ratio can't be wider than 3:1"
    if not 655_360 <= w * h <= 8_294_400:
        return "total pixels must be between 655,360 and 8,294,400"
    return None


def output_stems(images: list[Path]) -> dict[Path, str]:
    """Map each image to the stem used for its output files.

    `art.jpg` and `art.png` would both produce `art_mockup.png`, so stems that
    collide get the source extension appended (`art_jpg_mockup.png`).
    """
    counts = Counter(p.stem.lower() for p in images)
    return {
        p: f"{p.stem}_{p.suffix[1:].lower()}" if counts[p.stem.lower()] > 1 else p.stem
        for p in images
    }


def fill_placeholders(prompt: str, image_path: Path) -> str:
    """Replace {filename} with the image's file name (no extension)."""
    return prompt.replace("{filename}", image_path.stem)


def sanitize_error(msg: str) -> str:
    """Strip API keys and other secrets from error messages before logging."""
    msg = re.sub(r'key=[A-Za-z0-9_-]+', 'key=REDACTED', msg)
    msg = re.sub(r'sk-[A-Za-z0-9_*-]{8,}', 'sk-REDACTED', msg)
    return re.sub(r'AIza[0-9A-Za-z_-]{20,}', 'AIza-REDACTED', msg)


def extract_retry_delay(error_message: str) -> int:
    """Seconds to wait, from 'retry in 41s' (Gemini) / 'try again in 20ms' (OpenAI)."""
    match = re.search(r'(?:retry|try again)\D{0,20}?(\d+(?:\.\d+)?)\s*(ms|s)?',
                      str(error_message), re.IGNORECASE)
    if not match:
        return RETRY_BASE_DELAY
    seconds = float(match.group(1)) / (1000 if match.group(2) == "ms" else 1)
    return min(int(math.ceil(seconds)) + 2, 600)


def open_path(path: Path) -> None:
    """Open a file or folder with the OS default app."""
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _field(obj, name, default=None):
    """Read *name* from an SDK object or a plain dict (batch results are dicts)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _enum_name(value) -> str:
    return getattr(value, "name", None) or str(value)


# ---------------------------------------------------------------------------
# Input images
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime: str
    filename: str
    note: str = ""   # set when the file was converted or downscaled


def prepare_input_image(path: Path) -> PreparedImage:
    """Bytes ready to send for *path* (converted/downscaled only if needed)."""
    st = path.stat()
    return _prepare_input_cached(str(path), st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=8)
def _prepare_input_cached(path_str: str, _mtime_ns: int, size: int) -> PreparedImage:
    # Cached so a prompt set doesn't re-read (or re-convert) the same file for
    # every prompt; mtime/size are part of the key so edited files are re-read.
    path = Path(path_str)
    mime = NATIVE_INPUT_MIME.get(path.suffix.lower())
    with Image.open(path) as im:
        orig_size = im.size
        if mime and size <= MAX_INPUT_BYTES and max(orig_size) <= MAX_INPUT_EDGE:
            return PreparedImage(path.read_bytes(), mime, path.name)

        im.seek(0)   # first frame of an animated GIF
        img = ImageOps.exif_transpose(im)
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (
            img.mode == "P" and "transparency" in img.info)
        img = img.convert("RGBA" if has_alpha else "RGB")
        if max(img.size) > MAX_INPUT_EDGE:
            img.thumbnail((MAX_INPUT_EDGE, MAX_INPUT_EDGE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        data, out_mime, ext = buf.getvalue(), "image/png", "png"
        if len(data) > MAX_INPUT_BYTES:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=92)
            data, out_mime, ext = buf.getvalue(), "image/jpeg", "jpg"

    notes = []
    if not mime:
        notes.append(f"converted {path.suffix.lower()} to {ext.upper()}")
    if max(img.size) < max(orig_size):
        notes.append(f"downscaled {orig_size[0]}×{orig_size[1]} → {img.size[0]}×{img.size[1]}")
    if not notes:
        notes.append(f"re-encoded ({size / 1e6:.0f} MB) to fit API limits")
    return PreparedImage(data, out_mime, f"{path.stem}.{ext}", "; ".join(notes))


# ---------------------------------------------------------------------------
# Generation (one request each) and cost tracking
# ---------------------------------------------------------------------------

@dataclass
class GenResult:
    images: list[tuple[bytes, str]] = field(default_factory=list)   # (data, mime)
    reason: str = ""             # why fewer images than requested came back
    cost: float | None = None    # actual USD from the API's usage data, if known


def estimate_unit_cost(spec: dict, provider: str, image_size: str,
                       openai_size: str, openai_quality: str) -> tuple[float | None, str]:
    """Estimated USD per output image, plus a note when it's an upper bound."""
    cost = spec.get("cost") or {}
    if provider == "Gemini":
        return cost.get(image_size), ""
    sizes = sorted({k.split("/")[0] for k in cost}) if openai_size == "auto" else [openai_size]
    qualities = ["high"] if openai_quality == "auto" else [openai_quality]
    candidates = [cost[f"{s}/{q}"] for s in sizes for q in qualities if f"{s}/{q}" in cost]
    if not candidates:
        return None, ""
    notes = []
    if openai_quality == "auto":
        notes.append("quality is 'auto', so this assumes high")
    if openai_size == "auto":
        notes.append("size is 'auto', so this assumes the priciest size")
    return max(candidates), ("upper bound — " + "; ".join(notes)) if notes else ""


def openai_actual_cost(usage, spec: dict, batch: bool = False) -> float | None:
    """USD for one OpenAI request, from the `usage` block it returned."""
    prices = spec.get("token_prices")
    if not usage or not prices:
        return None
    details = _field(usage, "input_tokens_details")
    image_in = _field(details, "image_tokens", 0) or 0
    text_in = _field(details, "text_tokens")
    if text_in is None:
        text_in = max((_field(usage, "input_tokens", 0) or 0) - image_in, 0)
    image_out = _field(usage, "output_tokens", 0) or 0
    usd = (text_in * prices["text_in"] + image_in * prices["image_in"]
           + image_out * prices["image_out"]) / 1e6
    return usd * (0.5 if batch else 1.0)


def gemini_actual_cost(usage, spec: dict, image_size: str | None,
                       n_images: int, batch: bool = False) -> float | None:
    """Gemini bills a flat price per output image plus input tokens; this adds both."""
    per_image = (spec.get("cost") or {}).get(image_size)
    if per_image is None:
        return None
    tokens = (_field(usage, "prompt_token_count") or _field(usage, "promptTokenCount") or 0)
    usd = per_image * n_images + tokens * spec.get("input_per_m", 0) / 1e6
    return usd * (0.5 if batch else 1.0)


def gemini_result(response, spec: dict, image_size: str | None,
                  batch: bool = False) -> GenResult:
    """Pull the images out of a Gemini response (SDK object or batch-result dict).

    When no image comes back, *reason* says why — a safety block, the finish
    reason, and any text the model sent instead.
    """
    images, texts = [], []
    feedback = _field(response, "prompt_feedback") or _field(response, "promptFeedback")
    block = _field(feedback, "block_reason") or _field(feedback, "blockReason")
    candidates = _field(response, "candidates") or []
    finish = None
    if candidates:
        cand = candidates[0]
        finish = _field(cand, "finish_reason") or _field(cand, "finishReason")
        for part in _field(_field(cand, "content"), "parts") or []:
            if _field(part, "thought"):
                continue
            inline = _field(part, "inline_data") or _field(part, "inlineData")
            if inline:
                mime = _field(inline, "mime_type") or _field(inline, "mimeType") or ""
                data = _field(inline, "data")
                if mime.startswith("image/") and data:
                    images.append((base64.b64decode(data) if isinstance(data, str) else data, mime))
            elif _field(part, "text"):
                texts.append(_field(part, "text").strip())

    reason = ""
    if not images:
        if block:
            reason = f"prompt blocked ({_enum_name(block)})"
        elif not candidates:
            reason = "no candidates returned"
        elif finish and _enum_name(finish) not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            reason = f"finish reason {_enum_name(finish)}"
        else:
            reason = "no image in response"
        if texts:
            reason += f" — model said: {' '.join(texts)[:300]}"
    usage = _field(response, "usage_metadata") or _field(response, "usageMetadata")
    return GenResult(images, reason, gemini_actual_cost(usage, spec, image_size, len(images), batch))


def openai_result(body, spec: dict, out_mime: str, batch: bool = False) -> GenResult:
    """Decode an images.edit response (SDK object or batch-result dict)."""
    images = [(base64.b64decode(b64), out_mime)
              for item in _field(body, "data") or []
              if (b64 := _field(item, "b64_json"))]
    return GenResult(images, "" if images else "API returned no image data",
                     openai_actual_cost(_field(body, "usage"), spec, batch))


def make_client(provider: str, api_key: str):
    if provider == "OpenAI":
        return OpenAI(api_key=api_key)
    return genai.Client(api_key=api_key)


def gemini_generate(client, model: str, spec: dict, prompt: str,
                    inputs: list[PreparedImage], aspect_ratio: str,
                    image_size: str) -> GenResult:
    """Send the input image(s) + prompt to generate_content and return the result."""
    image_config = {"aspect_ratio": aspect_ratio}
    if image_size:
        image_config["image_size"] = image_size
    response = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=i.data, mime_type=i.mime) for i in inputs]
                 + [types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
            image_config=types.ImageConfig(**image_config),
        ),
    )
    return gemini_result(response, spec, image_size)


def openai_generate(client, model: str, spec: dict, prompt: str,
                    inputs: list[PreparedImage], options: dict,
                    out_mime: str) -> GenResult:
    """Send the input image(s) + prompt to images.edit and return the result."""
    files = [(i.filename, i.data, i.mime) for i in inputs]
    result = client.images.edit(
        model=model, prompt=prompt,
        image=files if len(files) > 1 else files[0],
        **options,
    )
    return openai_result(result, spec, out_mime)


def save_output_image(data: bytes, mime: str, path: Path, target_mime: str) -> Image.Image:
    """Save API image bytes to *path*, converting only if the format differs.

    Writes to a temp file and renames, so a crash never leaves a half-written
    file that a later run would skip as "already done". Returns a thumbnail.
    """
    tmp = path.with_name(path.name + ".part")
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        if mime == target_mime:
            tmp.write_bytes(data)
        else:
            fmt = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}[target_mime]
            out = img
            if fmt != "PNG" and out.mode not in ("RGB", "RGBA", "L"):
                out = out.convert("RGBA")
            if fmt == "JPEG" and out.mode == "RGBA":   # JPEG has no alpha: flatten on white
                flat = Image.new("RGB", out.size, "white")
                flat.paste(out, mask=out.getchannel("A"))
                out = flat
            out.save(tmp, fmt, **({"quality": 95} if fmt != "PNG" else {}))
        thumb = img.copy()
    thumb.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
    os.replace(tmp, path)
    return thumb


# ---------------------------------------------------------------------------
# Errors and retries
# ---------------------------------------------------------------------------

class _RateLimitRetry(Exception):
    """429 — wait *wait* seconds, then try again."""
    def __init__(self, wait: int):
        super().__init__(f"rate limited, retry in {wait}s")
        self.wait = wait


class _ServiceUnavailable(Exception):
    """The model is temporarily overloaded (5xx) or the network dropped."""
    def __init__(self, detail: str = ""):
        super().__init__(detail)
        self.detail = detail


class _FatalQuota(Exception):
    """Out of credits or daily quota — retrying won't help, so the run stops."""


class _Cancelled(Exception):
    """The user pressed Stop before a batch job was submitted."""


# Not "check your plan and billing": Gemini says that on ordinary per-minute 429s too.
_FATAL_QUOTA_MARKERS = ("insufficient_quota", "billing_hard_limit", "perday", "limit: 0")


def classify_api_error(exc: Exception) -> Exception:
    """Map a provider SDK exception onto the retry exceptions above.

    Returns *exc* unchanged when it isn't retryable (bad request, safety
    refusal, auth error…) so the item is marked failed straight away.
    """
    msg = str(exc)
    low = msg.lower()
    status = getattr(exc, "status_code", None)            # openai
    if status is None and isinstance(getattr(exc, "code", None), int):
        status = exc.code                                 # google-genai
    is_429 = status == 429 or (status is None and ("429" in msg or "resource_exhausted" in low))

    # 402 = Gemini prepaid credits used up ("Your prepayment credits are depleted")
    if status == 402 or (is_429 and any(marker in low for marker in _FATAL_QUOTA_MARKERS)):
        return _FatalQuota(sanitize_error(msg))
    if is_429:
        return _RateLimitRetry(extract_retry_delay(msg))
    if status in (500, 502, 503, 504) or (
            status is None and ("503" in msg or "unavailable" in low or "overloaded" in low)):
        return _ServiceUnavailable(sanitize_error(msg))
    name = type(exc).__name__
    if status is None and ("Timeout" in name or "Connect" in name
                           or isinstance(exc, (TimeoutError, ConnectionError))):
        return _ServiceUnavailable(f"network error: {sanitize_error(msg) or name}")
    return exc


class _Pacer:
    """Spaces out request starts across parallel workers.

    After a 429 every worker waits out the same cool-down, rather than each
    one hammering the API on its own schedule.
    """
    def __init__(self, spacing: float):
        self._lock = threading.Lock()
        self._next = 0.0
        self._spacing = spacing

    def wait_turn(self, should_stop) -> bool:
        """Block until this worker may send a request. False if stopped first."""
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next:
                    self._next = now + self._spacing
                    return True
                delay = self._next - now
            if should_stop():
                return False
            time.sleep(min(delay, 0.5))

    def cool_down(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


class _RunStats:
    """Progress counters shared by worker threads — read/write under `lock`."""
    def __init__(self, total: int, skipped: int = 0):
        self.lock = threading.RLock()
        self.total = total
        self.skipped = skipped
        self.succeeded = 0
        self.failed = 0
        self.spent = 0.0
        self.unpriced = 0          # saved images whose cost isn't known
        self.in_flight = 0
        self.consecutive_503 = 0
        self.started = time.monotonic()
        self.fatal = ""            # why the run had to stop early, if it did

    def record(self, succeeded: int = 0, failed: int = 0, cost: float | None = None):
        with self.lock:
            self.succeeded += succeeded
            self.failed += failed
            if cost is not None:
                self.spent += cost
            elif succeeded:
                self.unpriced += succeeded

    @property
    def done(self) -> int:
        return self.skipped + self.succeeded + self.failed

    def spent_text(self) -> str:
        with self.lock:
            if self.unpriced and self.spent:
                return f"${self.spent:.2f} + {self.unpriced} unpriced"
            if self.unpriced:
                return ""
            return f"${self.spent:.2f}" if self.spent else ""


# ---------------------------------------------------------------------------
# Work items
# ---------------------------------------------------------------------------

@dataclass
class WorkItem:
    key: str                 # unique id — batch result key and log label
    image: Path
    suffix: str              # "mockup" (single prompt) or "1", "2"… (prompt set)
    prompt: str
    outputs: list[Path]      # files this one request should produce
    first_for_image: bool    # log input-conversion notes only once per image


@dataclass
class RunJob:
    """Everything a worker thread needs — captured on the main thread at start."""
    provider: str
    model: str
    spec: dict
    api_key: str
    images: list[Path]
    prompts: list[tuple[str, str]]   # (suffix, text)
    output_dir: Path
    overwrite: bool
    aspect_ratio: str
    image_size: str
    openai_size: str
    openai_quality: str
    output_format: str               # key of OUTPUT_FORMATS
    variations: int
    parallel: int
    references: list[Path]
    batch: bool
    test: bool = False

    @property
    def ext(self) -> str:
        return OUTPUT_FORMATS[self.output_format][0]

    @property
    def out_mime(self) -> str:
        return OUTPUT_FORMATS[self.output_format][1]

    def openai_options(self, n: int) -> dict:
        """size / quality / format / n for images.edit ('auto' = let the API pick)."""
        options = {"output_format": OUTPUT_FORMATS[self.output_format][2]}
        if self.openai_size != "auto":
            options["size"] = self.openai_size
        if self.openai_quality != "auto":
            options["quality"] = self.openai_quality
        if n > 1:
            options["n"] = n
        return options


def build_work_items(job: RunJob) -> tuple[list[WorkItem], int]:
    """Expand images × prompts × variations into requests, skipping existing outputs.

    OpenAI returns several variations from one request (`n`), so it gets one
    item per image+prompt; Gemini needs a separate request per variation.
    Returns (items, number of output files skipped).
    """
    stems = output_stems(job.images)
    items: list[WorkItem] = []
    skipped = 0
    for image in job.images:
        first = True
        for suffix, prompt in job.prompts:
            base = f"{stems[image]}_{suffix}"
            if job.variations == 1:
                paths = [job.output_dir / f"{base}.{job.ext}"]
            else:
                paths = [job.output_dir / f"{base}_v{k}.{job.ext}"
                         for k in range(1, job.variations + 1)]
            todo = paths if job.overwrite else [p for p in paths if not p.exists()]
            skipped += len(paths) - len(todo)
            if not todo:
                continue
            if job.provider == "OpenAI":
                items.append(WorkItem(base, image, suffix, prompt, todo, first))
                first = False
            else:
                for path in todo:
                    items.append(WorkItem(path.stem, image, suffix, prompt, [path], first))
                    first = False
    return items, skipped


# ---------------------------------------------------------------------------
# Batch backends — one per provider, driven by App._watch_batch()
# ---------------------------------------------------------------------------

class _GeminiBatch:
    """Gemini Batch API: images go up via the File API, requests as JSONL."""
    TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
    HAS_RESULTS = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def upload_image(self, image: PreparedImage) -> tuple[str, dict]:
        """Upload one image; returns (file name for cleanup, request part)."""
        f = self.client.files.upload(
            file=io.BytesIO(image.data),
            config=types.UploadFileConfig(display_name=image.filename, mime_type=image.mime),
        )
        return f.name, {"fileData": {"fileUri": f.uri, "mimeType": image.mime}}

    def request_line(self, job: RunJob, item: WorkItem, image_parts: list[dict]) -> dict:
        image_config = {"aspectRatio": job.aspect_ratio}
        if job.image_size:
            image_config["imageSize"] = job.image_size
        return {
            "key": item.key,
            "request": {
                "contents": [{"parts": image_parts
                              + [{"text": fill_placeholders(item.prompt, item.image)}]}],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "imageConfig": image_config,
                },
            },
        }

    def submit(self, job: RunJob, lines_path: Path) -> tuple[str, list[str]]:
        jsonl = self.client.files.upload(
            file=str(lines_path),
            config=types.UploadFileConfig(display_name="batch-requests.jsonl", mime_type="jsonl"),
        )
        batch_job = self.client.batches.create(
            model=job.model, src=jsonl.name,
            config={"display_name": f"bulk-image-gen-{int(time.time())}"},
        )
        return batch_job.name, [jsonl.name]

    def poll(self, job_id: str) -> tuple[str, bool, bool, str]:
        """(state, finished, has_results, error detail)"""
        job = self.client.batches.get(name=job_id)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        detail = str(getattr(job, "error", None) or "")
        return state, state in self.TERMINAL, state in self.HAS_RESULTS, detail

    def cancel(self, job_id: str) -> None:
        self.client.batches.cancel(name=job_id)

    def results(self, job_id: str, state: dict):
        """Yield (key, GenResult) — or (key, error message) — per result line."""
        job = self.client.batches.get(name=job_id)
        data = self.client.files.download(file=job.dest.file_name)
        if not isinstance(data, bytes):
            data = data.read()
        spec = model_spec(state["model"], "Gemini")
        # Line by line: avoids extra full-size copies of a big (4K) results file.
        for raw in io.BytesIO(data):
            if not raw.strip():
                continue
            entry = json.loads(raw)
            key = entry.get("key", "unknown")
            if entry.get("error"):
                yield key, str(entry["error"])
                continue
            yield key, gemini_result(entry.get("response") or {}, spec,
                                     state.get("image_size"), batch=True)

    def cleanup(self, names: list[str]) -> None:
        for name in names:
            try:
                self.client.files.delete(name=name)
            except Exception:
                pass   # best effort — the File API expires uploads after 48h anyway


class _OpenAIBatch:
    """OpenAI Batch API: images uploaded as files, requests as JSONL on /v1/images/edits."""
    TERMINAL = {"completed", "failed", "expired", "cancelled"}

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def upload_image(self, image: PreparedImage) -> tuple[str, dict]:
        f = self.client.files.create(file=(image.filename, image.data, image.mime),
                                     purpose="vision")
        return f.id, {"file_id": f.id}

    def request_line(self, job: RunJob, item: WorkItem, image_parts: list[dict]) -> dict:
        # The JSON form of images.edit takes `images: [{"file_id": …}]` rather
        # than multipart uploads.
        return {
            "custom_id": item.key,
            "method": "POST",
            "url": "/v1/images/edits",
            "body": {
                "model": job.model,
                "prompt": fill_placeholders(item.prompt, item.image),
                "images": image_parts,
                **job.openai_options(len(item.outputs)),
            },
        }

    def submit(self, job: RunJob, lines_path: Path) -> tuple[str, list[str]]:
        with lines_path.open("rb") as f:
            jsonl = self.client.files.create(file=f, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=jsonl.id, endpoint="/v1/images/edits",
            completion_window="24h", metadata={"app": "bulk-image-gen"},
        )
        return batch.id, [jsonl.id]

    def poll(self, job_id: str) -> tuple[str, bool, bool, str]:
        batch = self.client.batches.retrieve(job_id)
        state = batch.status
        counts = batch.request_counts
        if counts and counts.total:
            state += f" ({counts.completed}/{counts.total} done, {counts.failed} failed)"
        detail = ""
        if batch.errors and batch.errors.data:
            detail = "; ".join(e.message or "" for e in batch.errors.data[:3])
        has_results = bool(batch.output_file_id or batch.error_file_id)
        return state, batch.status in self.TERMINAL, has_results, detail

    def cancel(self, job_id: str) -> None:
        self.client.batches.cancel(job_id)

    def results(self, job_id: str, state: dict):
        batch = self.client.batches.retrieve(job_id)
        spec = model_spec(state["model"], "OpenAI")
        out_mime = state.get("out_mime", "image/png")
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "results.jsonl"
                # Stream to disk — 4K results files can be very large.
                with self.client.files.with_streaming_response.content(file_id) as resp:
                    resp.stream_to_file(path)
                with path.open("rb") as f:
                    for raw in f:
                        if not raw.strip():
                            continue
                        entry = json.loads(raw)
                        key = entry.get("custom_id", "unknown")
                        response = entry.get("response") or {}
                        body = response.get("body") or {}
                        if entry.get("error") or response.get("status_code", 200) != 200:
                            err = entry.get("error") or body.get("error") or {}
                            msg = err.get("message") if isinstance(err, dict) else str(err)
                            yield key, msg or f"HTTP {response.get('status_code')}"
                            continue
                        yield key, openai_result(body, spec, out_mime, batch=True)

    def cleanup(self, file_ids: list[str]) -> None:
        for file_id in file_ids:
            try:
                self.client.files.delete(file_id)
            except Exception:
                pass


def make_batch_backend(provider: str, api_key: str):
    return _OpenAIBatch(api_key) if provider == "OpenAI" else _GeminiBatch(api_key)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path, expected: type):
    """Return (data, error). data is None if the file is missing or unreadable."""
    if not path.exists():
        return None, None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(data, expected):
        return None, f"expected a JSON {expected.__name__}"
    return data, None


def _write_json_atomic(path: Path, data, backup: bool = False) -> None:
    """Write JSON via a temp file + rename, so a crash can't leave half a file.

    With *backup*, the previous version is first copied to `<name>.bak`.
    Raises on failure — callers decide whether to tell the user.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if backup and path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:   # OneDrive / antivirus briefly holding the file
                if attempt == 4:
                    raise
                time.sleep(0.2)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Persistent config  (config.json — gitignored, never committed)
# API key is stored in the OS keychain via `keyring`, never in config.json
# ---------------------------------------------------------------------------

CONFIG_FILE = SCRIPT_DIR / "config.json"

_KEYCHAIN_SERVICES = {
    "Gemini": "gemini-bulk-image-gen",
    "OpenAI": "openai-bulk-image-gen",
}
_KEYCHAIN_USERS = {
    "Gemini": "google_api_key",
    "OpenAI": "openai_api_key",
}
_API_KEY_ENV_VARS = {
    "Gemini": "GOOGLE_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
}
_SAVE_API_KEY     = "save_api_key"   # boolean flag stored in config.json


def load_config() -> dict:
    """Load config.json, return empty dict on any error."""
    data, _ = _read_json(CONFIG_FILE, dict)
    return data or {}


def save_config(data: dict) -> None:
    """Write config.json (no secrets), silently ignore write errors."""
    try:
        _write_json_atomic(CONFIG_FILE, data)
    except Exception:
        pass


def keychain_load_api_key(provider: str = "Gemini") -> str:
    """Return the saved API key for *provider* from the OS keychain, or '' on any error."""
    if not _KEYRING_AVAILABLE:
        return ""
    try:
        return keyring.get_password(
            _KEYCHAIN_SERVICES[provider], _KEYCHAIN_USERS[provider]
        ) or ""
    except Exception:
        return ""


def keychain_save_api_key(api_key: str, provider: str = "Gemini") -> None:
    """Store the API key for *provider* in the OS keychain, silently ignore errors."""
    if not _KEYRING_AVAILABLE:
        return
    try:
        keyring.set_password(
            _KEYCHAIN_SERVICES[provider], _KEYCHAIN_USERS[provider], api_key
        )
    except Exception:
        pass


def keychain_delete_api_key(provider: str = "Gemini") -> None:
    """Remove the API key for *provider* from the OS keychain, silently ignore errors."""
    if not _KEYRING_AVAILABLE:
        return
    try:
        keyring.delete_password(
            _KEYCHAIN_SERVICES[provider], _KEYCHAIN_USERS[provider]
        )
    except keyring.errors.PasswordDeleteError:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Saved prompts  (prompts.json — gitignored, never committed)
# Stored as an ordered list of {"name": str, "text": str} objects so the
# display order is preserved across sessions.
# ---------------------------------------------------------------------------

PROMPTS_FILE = SCRIPT_DIR / "prompts.json"


def load_prompts() -> tuple[list[dict], str | None]:
    """Return (saved prompts, error). error is set if the file exists but can't be read."""
    data, err = _read_json(PROMPTS_FILE, list)
    return data or [], err


def save_prompts(prompts: list[dict]) -> None:
    """Write prompts.json atomically (previous version kept as .bak). Raises on failure."""
    _write_json_atomic(PROMPTS_FILE, prompts, backup=True)


# ---------------------------------------------------------------------------
# Prompt sets  (prompt_sets.json — gitignored, never committed)
# Each set is {"name": str, "prompts": [{"name": str, "text": str}, ...]}
# ---------------------------------------------------------------------------

PROMPT_SETS_FILE = SCRIPT_DIR / "prompt_sets.json"


def load_prompt_sets() -> tuple[list[dict], str | None]:
    """Return (prompt sets, error). error is set if the file exists but can't be read."""
    data, err = _read_json(PROMPT_SETS_FILE, list)
    return data or [], err


def save_prompt_sets(sets: list[dict]) -> None:
    """Write prompt_sets.json atomically (previous version kept as .bak). Raises on failure."""
    _write_json_atomic(PROMPT_SETS_FILE, sets, backup=True)


# ---------------------------------------------------------------------------
# Batch job state  (batch_state.json — persists active batch job across restarts)
# ---------------------------------------------------------------------------

BATCH_STATE_FILE = SCRIPT_DIR / "batch_state.json"


def load_batch_state() -> dict | None:
    """Return the saved batch job state, or None."""
    data, _ = _read_json(BATCH_STATE_FILE, dict)
    if not data or not (data.get("job_id") or data.get("job_name")):
        return None
    # Files from older versions: Gemini-only, one output path per key.
    data.setdefault("provider", "Gemini")
    data.setdefault("job_id", data.get("job_name"))
    data["key_map"] = {k: v if isinstance(v, list) else [v]
                       for k, v in data.get("key_map", {}).items()}
    return data


def save_batch_state(state: dict) -> None:
    """Persist the active batch job state to disk. Raises on failure."""
    _write_json_atomic(BATCH_STATE_FILE, state)


def clear_batch_state() -> None:
    """Remove the batch state file."""
    try:
        if BATCH_STATE_FILE.exists():
            BATCH_STATE_FILE.unlink()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bulk Image Generator (Gemini + OpenAI)")
        self.resizable(True, True)
        self.geometry("980x900")
        self.minsize(780, 660)
        self._stop_event = threading.Event()
        self._running = False
        self._batch_active = False     # worker is watching a submitted batch job
        self._batch_cancel = False     # Stop chose "cancel job" over "stop watching"
        self._force_exit = False       # window closed mid-run: don't wait for workers
        self._last_output_dir: Path | None = None
        self._preview_path: Path | None = None
        self._preview_photo = None     # keep a reference or Tk drops the image
        self._config = load_config()
        self._prompts, prompts_err = load_prompts()           # individual saved prompts
        self._prompt_sets, sets_err = load_prompt_sets()      # prompt sets
        # If a prompt file exists but can't be read, saving would overwrite it
        # with whatever is in memory — so saving is blocked until it's fixed.
        self._prompts_locked = prompts_err
        self._prompt_sets_locked = sets_err
        self._ref_paths: list[str] = []
        # Keys typed or loaded per provider, so switching providers doesn't lose one.
        self._api_keys: dict[str, str] = {}
        self._active_provider = DEFAULT_PROVIDER
        # Suppresses the API-key swap when the provider change is being driven
        # by _load_state (so the value it just populated isn't replaced).
        self._suppress_api_key_swap = False
        self._build_ui()
        self._load_state()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        for path, err in ((PROMPTS_FILE, prompts_err), (PROMPT_SETS_FILE, sets_err)):
            if err:
                self.after(300, self._warn_unreadable, path, err)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}
        small = ("TkDefaultFont", 8)

        # --- API key ---
        tk.Label(self, text="API Key:", anchor="w").grid(row=0, column=0, sticky="w", **pad)
        self.api_key_var = tk.StringVar(
            value=os.environ.get(_API_KEY_ENV_VARS[DEFAULT_PROVIDER], "")
        )
        api_row = tk.Frame(self)
        api_row.grid(row=0, column=1, columnspan=2, sticky="ew", **pad)
        tk.Entry(api_row, textvariable=self.api_key_var, show="*", width=44).pack(side="left")
        self.save_api_key_var = tk.BooleanVar(value=False)
        self._save_key_cb = tk.Checkbutton(api_row, text="Save API key",
                                           variable=self.save_api_key_var)
        self._save_key_cb.pack(side="left", padx=(10, 0))

        # --- Provider + Model ---
        tk.Label(self, text="Provider:", anchor="w").grid(row=1, column=0, sticky="w", **pad)
        self.provider_var = tk.StringVar(value=DEFAULT_PROVIDER)
        prov_frame = tk.Frame(self)
        prov_frame.grid(row=1, column=1, columnspan=2, sticky="ew", **pad)
        self._provider_combo = ttk.Combobox(prov_frame, textvariable=self.provider_var,
                                            values=PROVIDERS, width=10, state="readonly")
        self._provider_combo.pack(side="left")
        self._provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        tk.Label(prov_frame, text="  Model:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self._model_combo = ttk.Combobox(prov_frame, textvariable=self.model_var,
                                         values=MODELS_BY_PROVIDER[DEFAULT_PROVIDER],
                                         width=28)
        self._model_combo.pack(side="left", padx=(4, 0))
        # Editable, so a custom model ID can be typed; re-read it when the user leaves the field.
        for event in ("<<ComboboxSelected>>", "<FocusOut>", "<Return>"):
            self._model_combo.bind(event, self._on_model_change)
        self._model_hint = tk.Label(prov_frame, text="", fg="gray", font=small)
        self._model_hint.pack(side="left", padx=(8, 0))

        # --- Input folder ---
        tk.Label(self, text="Input folder:", anchor="w").grid(row=2, column=0, sticky="w", **pad)
        self.input_var = tk.StringVar()
        tk.Entry(self, textvariable=self.input_var, width=45).grid(
            row=2, column=1, sticky="ew", **pad)
        tk.Button(self, text="Browse…", command=self._pick_input).grid(
            row=2, column=2, sticky="w", padx=(0, 10))

        # --- Output folder ---
        tk.Label(self, text="Output folder:", anchor="w").grid(row=3, column=0, sticky="w", **pad)
        self.output_var = tk.StringVar()
        tk.Entry(self, textvariable=self.output_var, width=45).grid(
            row=3, column=1, sticky="ew", **pad)
        tk.Button(self, text="Browse…", command=self._pick_output).grid(
            row=3, column=2, sticky="w", padx=(0, 10))

        # --- Size / quality ---
        # Two dropdowns whose label/values flip based on the active model.
        # Gemini: Aspect ratio + Resolution (512/1K/2K/4K)
        # OpenAI: Size (editable — any valid WxH) + Quality
        tk.Label(self, text="Image:", anchor="w").grid(row=4, column=0, sticky="w", **pad)
        size_frame = tk.Frame(self)
        size_frame.grid(row=4, column=1, columnspan=2, sticky="w", **pad)

        self.aspect_ratio_var   = tk.StringVar(value=DEFAULT_ASPECT_RATIO)
        self.image_size_var     = tk.StringVar(value=DEFAULT_IMAGE_SIZE)
        self.openai_size_var    = tk.StringVar(value=DEFAULT_OPENAI_SIZE)
        self.openai_quality_var = tk.StringVar(value=DEFAULT_OPENAI_QUALITY)

        self._opt1_label = tk.Label(size_frame, text="Aspect ratio:")
        self._opt1_label.pack(side="left")
        self._opt1_combo = ttk.Combobox(size_frame, textvariable=self.aspect_ratio_var,
                                        width=11, state="readonly")
        self._opt1_combo.pack(side="left", padx=(4, 16))

        self._opt2_label = tk.Label(size_frame, text="Resolution:")
        self._opt2_label.pack(side="left")
        self._opt2_combo = ttk.Combobox(size_frame, textvariable=self.image_size_var,
                                        width=8, state="readonly")
        self._opt2_combo.pack(side="left", padx=(4, 0))
        self._opt_hint = tk.Label(size_frame, text="", fg="gray", font=small)
        self._opt_hint.pack(side="left", padx=(8, 0))

        # --- Output options ---
        tk.Label(self, text="Output:", anchor="w").grid(row=5, column=0, sticky="w", **pad)
        out_frame = tk.Frame(self)
        out_frame.grid(row=5, column=1, columnspan=2, sticky="w", **pad)

        tk.Label(out_frame, text="Format:").pack(side="left")
        self.output_format_var = tk.StringVar(value=DEFAULT_OUTPUT_FORMAT)
        ttk.Combobox(out_frame, textvariable=self.output_format_var,
                     values=list(OUTPUT_FORMATS), width=6,
                     state="readonly").pack(side="left", padx=(4, 16))

        tk.Label(out_frame, text="Variations:").pack(side="left")
        self.variations_var = tk.IntVar(value=1)
        ttk.Spinbox(out_frame, from_=1, to=MAX_VARIATIONS, textvariable=self.variations_var,
                    width=3, state="readonly").pack(side="left", padx=(4, 16))

        tk.Label(out_frame, text="Parallel requests:").pack(side="left")
        self.parallel_var = tk.IntVar(value=DEFAULT_PARALLEL)
        ttk.Spinbox(out_frame, from_=1, to=MAX_PARALLEL, textvariable=self.parallel_var,
                    width=3, state="readonly").pack(side="left", padx=(4, 16))

        self.overwrite_var = tk.BooleanVar(value=False)
        tk.Checkbutton(out_frame, text="Overwrite existing outputs",
                       variable=self.overwrite_var).pack(side="left")

        # --- Run mode ---
        tk.Label(self, text="Run mode:", anchor="w").grid(row=6, column=0, sticky="w", **pad)
        mode_frame = tk.Frame(self)
        mode_frame.grid(row=6, column=1, columnspan=2, sticky="w", **pad)
        self.run_mode_var = tk.StringVar(value=RUN_MODES[0])
        self._run_mode_combo = ttk.Combobox(mode_frame, textvariable=self.run_mode_var,
                                            values=RUN_MODES, width=14, state="readonly")
        self._run_mode_combo.pack(side="left", padx=(0, 12))
        self._run_mode_hint = tk.Label(mode_frame, text="", fg="gray", font=small)
        self._run_mode_hint.pack(side="left")

        # --- Reference images ---
        tk.Label(self, text="References:", anchor="w").grid(row=7, column=0, sticky="w", **pad)
        ref_frame = tk.Frame(self)
        ref_frame.grid(row=7, column=1, columnspan=2, sticky="ew", **pad)
        tk.Button(ref_frame, text="Add…", command=self._add_references).pack(side="left")
        tk.Button(ref_frame, text="Clear", command=self._clear_references).pack(
            side="left", padx=(4, 0))
        self._ref_info_var = tk.StringVar()
        tk.Label(ref_frame, textvariable=self._ref_info_var, fg="gray", font=small,
                 anchor="w", justify="left", wraplength=640).pack(side="left", padx=(8, 0))

        # --- Prompt Set selector ---
        tk.Label(self, text="Prompt set:", anchor="w").grid(row=8, column=0, sticky="w", **pad)

        set_col = tk.Frame(self)
        set_col.grid(row=8, column=1, columnspan=2, sticky="ew", **pad)

        self._prompt_set_var = tk.StringVar(value=_SINGLE_PROMPT)
        self._prompt_set_combo = ttk.Combobox(set_col, textvariable=self._prompt_set_var,
                                              state="readonly", width=34)
        self._prompt_set_combo.pack(side="left")
        self._prompt_set_combo.bind("<<ComboboxSelected>>", self._on_prompt_set_selected)

        tk.Button(set_col, text="New Set…", command=self._new_prompt_set).pack(
            side="left", padx=(8, 0))
        tk.Button(set_col, text="Edit Set…", command=self._edit_prompt_set).pack(
            side="left", padx=(4, 0))
        tk.Button(set_col, text="Delete Set", command=self._delete_prompt_set).pack(
            side="left", padx=(4, 0))

        # Info label showing set contents
        self._set_info_var = tk.StringVar(value="")
        self._set_info_label = tk.Label(self, textvariable=self._set_info_var,
                                        anchor="w", fg="gray",
                                        font=small, wraplength=700,
                                        justify="left")
        self._set_info_label.grid(row=9, column=1, columnspan=2, sticky="w",
                                  padx=10, pady=(0, 2))

        # --- Prompt (single-prompt editor) ---
        self._prompt_label = tk.Label(self, text="Prompt:", anchor="nw")
        self._prompt_label.grid(row=10, column=0, sticky="nw", **pad)

        prompt_col = tk.Frame(self)
        prompt_col.grid(row=10, column=1, columnspan=2, sticky="nsew", **pad)
        prompt_col.columnconfigure(0, weight=1)

        # Toolbar: saved-prompt picker + Save + Delete
        prompt_toolbar = tk.Frame(prompt_col)
        prompt_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        prompt_toolbar.columnconfigure(0, weight=1)

        self._saved_prompt_var = tk.StringVar()
        self._prompt_combo = ttk.Combobox(prompt_toolbar, textvariable=self._saved_prompt_var,
                                          state="readonly", width=38)
        self._prompt_combo.grid(row=0, column=0, sticky="ew")
        self._prompt_combo.bind("<<ComboboxSelected>>", self._on_prompt_selected)

        self._save_prompt_btn = tk.Button(prompt_toolbar, text="💾 Save", width=8,
                                          command=self._save_prompt)
        self._save_prompt_btn.grid(row=0, column=1, padx=(6, 0))
        self._delete_prompt_btn = tk.Button(prompt_toolbar, text="🗑 Delete", width=8,
                                            command=self._delete_prompt)
        self._delete_prompt_btn.grid(row=0, column=2, padx=(4, 0))

        # Text area
        self.prompt_text = scrolledtext.ScrolledText(prompt_col, width=60, height=5, wrap=tk.WORD)
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)
        self.prompt_text.grid(row=1, column=0, sticky="nsew")
        prompt_col.rowconfigure(1, weight=1)
        tk.Label(prompt_col, fg="gray", font=small, anchor="w", justify="left",
                 text="Tip: {filename} is replaced with each image's file name. "
                      "Your image is sent first, then any reference images in order."
                 ).grid(row=2, column=0, sticky="w", pady=(2, 0))
        # Remember the theme's colours so disabling/enabling the editor restores them.
        self._prompt_label_fg = self._prompt_label.cget("fg")
        self._prompt_text_bg = self.prompt_text.cget("bg")

        # --- Progress ---
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.grid(row=11, column=0, columnspan=3, sticky="ew", **pad)

        self.status_label = tk.Label(self, text="", anchor="w", fg="gray")
        self.status_label.grid(row=12, column=0, columnspan=3, sticky="w", **pad)

        # --- Log + latest-result preview ---
        log_frame = tk.Frame(self)
        log_frame.grid(row=13, column=0, columnspan=3, sticky="nsew", **pad)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(log_frame, width=70, height=8, state="disabled",
                                             wrap=tk.WORD, bg="#1e1e1e", fg="#d4d4d4",
                                             font=("TkFixedFont", 9))
        self.log.grid(row=0, column=0, sticky="nsew")

        preview = tk.Frame(log_frame)
        preview.grid(row=0, column=1, sticky="n", padx=(8, 0))
        box = tk.Frame(preview, width=PREVIEW_SIZE, height=PREVIEW_SIZE,
                       relief="groove", borderwidth=2)
        box.pack()
        box.pack_propagate(False)
        self._preview_label = tk.Label(box, text="Latest result\nappears here",
                                       fg="gray", cursor="hand2")
        self._preview_label.pack(expand=True, fill="both")
        self._preview_label.bind("<Button-1>", self._open_preview)
        self._preview_caption = tk.Label(preview, text="", fg="gray", font=small,
                                         wraplength=PREVIEW_SIZE)
        self._preview_caption.pack(pady=(2, 0))

        # --- Buttons ---
        # Native ttk themes (vista, aqua) ignore fg/bg on buttons, so switch to
        # the built-in cross-platform "clam" theme, which honours colours. Note
        # this restyles every ttk widget (comboboxes, progress bar…), not just
        # the coloured buttons.
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=14, column=0, columnspan=3, pady=10)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Run.TButton",
                        foreground="white", background="#0078d7",
                        font=("TkDefaultFont", 10, "bold"),
                        padding=6)
        style.map("Run.TButton",
                  background=[("active", "#005fa3"), ("disabled", "#a0a0a0")],
                  foreground=[("disabled", "#d0d0d0")])
        style.configure("Plain.TButton", padding=6)
        style.configure("Resume.TButton",
                        foreground="white", background="#2d8a4e",
                        font=("TkDefaultFont", 10, "bold"),
                        padding=6)
        style.map("Resume.TButton",
                  background=[("active", "#1e6b3a"), ("disabled", "#a0a0a0")],
                  foreground=[("disabled", "#d0d0d0")])
        self.run_btn = ttk.Button(btn_frame, text="Run", width=14,
                                  style="Run.TButton", command=self._start)
        self.run_btn.pack(side="left", padx=6)
        self.test_btn = ttk.Button(btn_frame, text="Test 1 image…", width=14,
                                   style="Plain.TButton", command=self._test_one)
        self.test_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(btn_frame, text="Stop", width=14,
                                   style="Plain.TButton",
                                   state="disabled", command=self._request_stop)
        self.stop_btn.pack(side="left", padx=6)
        self.resume_btn = ttk.Button(btn_frame, text="Resume Batch", width=14,
                                     style="Resume.TButton",
                                     state="disabled",
                                     command=self._resume_batch)
        self.resume_btn.pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Open output folder", width=18,
                   style="Plain.TButton",
                   command=self._open_output_folder).pack(side="left", padx=6)

        # Make columns/rows resize gracefully.
        self.columnconfigure(1, weight=1)
        self.rowconfigure(10, weight=1)   # prompt editor
        self.rowconfigure(13, weight=2)   # log

        # Disable "Save API key" if keyring is not installed
        if not _KEYRING_AVAILABLE:
            self._save_key_cb.configure(state="disabled")

    def _center(self, dlg: tk.Toplevel, width: int | None = None, height: int | None = None):
        """Centre a dialog over the main window."""
        self.update_idletasks()
        w = width or dlg.winfo_reqwidth()
        h = height or dlg.winfo_reqheight()
        x = self.winfo_x() + (self.winfo_width() - w) // 2
        y = self.winfo_y() + (self.winfo_height() - h) // 2
        dlg.geometry(f"+{x}+{y}")

    def _choice_dialog(self, title: str, message: str, buttons: list[str]) -> str | None:
        """Modal dialog with one button per choice. Returns the label clicked (None on Esc)."""
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.resizable(False, False)
        dlg.transient(self)
        choice = {"value": None}

        def pick(label):
            choice["value"] = label
            dlg.destroy()

        tk.Label(dlg, text=message, justify="left", wraplength=440).pack(padx=20, pady=(16, 12))
        row = tk.Frame(dlg)
        row.pack(pady=(0, 12))
        for label in buttons:
            tk.Button(row, text=label, width=14,
                      command=lambda lbl=label: pick(lbl)).pack(side="left", padx=4)
        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        self._center(dlg)
        dlg.grab_set()
        self.wait_window(dlg)
        return choice["value"]

    # ------------------------------------------------------------------
    # Provider / model switch
    # ------------------------------------------------------------------

    def _stored_api_key(self, provider: str) -> str:
        """Key for *provider*: typed this session > OS keychain (if saving) > env var."""
        return (self._api_keys.get(provider)
                or (keychain_load_api_key(provider) if self.save_api_key_var.get() else "")
                or os.environ.get(_API_KEY_ENV_VARS[provider], ""))

    def _on_provider_change(self, _event=None):
        """Swap the model list and API key for the currently selected provider."""
        provider = self.provider_var.get()

        models = MODELS_BY_PROVIDER.get(provider, [])
        self._model_combo.configure(values=models)
        model = self.model_var.get().strip()
        # Keep a custom model ID unless it clearly belongs to the other provider.
        if not model or infer_provider(model) not in (None, provider):
            self.model_var.set(models[0] if models else "")

        # API key field — remember what was typed for the provider we're leaving,
        # then show the new provider's key. Skipped during _load_state.
        if not self._suppress_api_key_swap and provider != self._active_provider:
            self._api_keys[self._active_provider] = self.api_key_var.get().strip()
            self.api_key_var.set(self._stored_api_key(provider))
        self._active_provider = provider

        self._on_model_change()

    def _on_model_change(self, _event=None):
        """Rebuild the size/quality choices, hints and run-mode lock for the model."""
        provider = self.provider_var.get()
        model = self.model_var.get().strip()
        spec = model_spec(model, provider)

        def keep_valid(var, values, default):
            if var.get() not in values:
                var.set(default if default in values else values[0])

        if provider == "OpenAI":
            self._opt1_label.configure(text="Size:")
            # Editable: gpt-image-2+ accept any valid WIDTHxHEIGHT.
            self._opt1_combo.configure(textvariable=self.openai_size_var,
                                       values=spec["sizes"], state="normal")
            self._opt2_label.configure(text="Quality:")
            self._opt2_combo.configure(textvariable=self.openai_quality_var,
                                       values=spec["qualities"])
            keep_valid(self.openai_quality_var, spec["qualities"], DEFAULT_OPENAI_QUALITY)
            self._opt_hint.configure(
                text="type any size: multiples of 16 · max 3840 px · up to 3:1")
        else:
            self._opt1_label.configure(text="Aspect ratio:")
            self._opt1_combo.configure(textvariable=self.aspect_ratio_var,
                                       values=spec["ratios"], state="readonly")
            keep_valid(self.aspect_ratio_var, spec["ratios"], DEFAULT_ASPECT_RATIO)
            self._opt2_label.configure(text="Resolution:")
            self._opt2_combo.configure(textvariable=self.image_size_var,
                                       values=spec["sizes"])
            keep_valid(self.image_size_var, spec["sizes"], DEFAULT_IMAGE_SIZE)
            self._opt_hint.configure(text="")

        shutdown = MODEL_SHUTDOWNS.get(model)
        if shutdown:
            self._model_hint.configure(text=f"⚠ {model} {shutdown}", fg="#c0392b")
        else:
            self._model_hint.configure(text=f"{spec['blurb']} · {price_hint(spec)}", fg="gray")

        # Run mode — Batch only for models with Batch API support
        if spec["batch"]:
            self._run_mode_combo.configure(state="readonly")
            self._run_mode_hint.configure(
                text="Batch sends all work as one async job · 50% cheaper · "
                     "results within 24h")
        else:
            self.run_mode_var.set(RUN_MODES[0])  # Real-time
            self._run_mode_combo.configure(state="disabled")
            self._run_mode_hint.configure(
                text=f"Batch isn't available for {model or 'this model'} — runs in real-time.")

    def _pick_input(self):
        folder = filedialog.askdirectory(title="Select input folder")
        if folder:
            self.input_var.set(folder)
            if not self.output_var.get():
                subfolder = "openai" if self.provider_var.get() == "OpenAI" else "gemini"
                self.output_var.set(str(Path(folder) / subfolder))

    def _pick_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_var.set(folder)

    # ------------------------------------------------------------------
    # Reference images
    # ------------------------------------------------------------------

    def _add_references(self):
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Choose reference images (sent with every request)",
            filetypes=[("Images", patterns), ("All files", "*.*")])
        for p in paths:
            if p not in self._ref_paths:
                self._ref_paths.append(p)
        if len(self._ref_paths) > MAX_REFERENCE_IMAGES:
            self._ref_paths = self._ref_paths[:MAX_REFERENCE_IMAGES]
            self._log(f"Only the first {MAX_REFERENCE_IMAGES} reference images are kept.")
        self._update_ref_info()

    def _clear_references(self):
        self._ref_paths = []
        self._update_ref_info()

    def _update_ref_info(self):
        if not self._ref_paths:
            self._ref_info_var.set("None — optional images sent with every request "
                                   "(e.g. a room scene to keep mockups consistent)")
        else:
            names = ", ".join(Path(p).name for p in self._ref_paths)
            self._ref_info_var.set(f"{len(self._ref_paths)} image(s): {names}")

    # ------------------------------------------------------------------
    # Saving prompts / prompt sets
    # ------------------------------------------------------------------

    def _warn_unreadable(self, path: Path, err: str):
        messagebox.showwarning(
            "Couldn't read saved prompts",
            f"{path.name} exists but couldn't be read:\n{err}\n\n"
            f"To protect it, changes to these prompts won't be saved this session. "
            f"Fix or restore the file (the previous version may be in "
            f"{path.name}.bak or OneDrive's version history), then restart.",
            parent=self)

    def _persist(self, kind: str) -> bool:
        """Save prompts ("prompts") or prompt sets ("sets"); tell the user if it fails."""
        if kind == "sets":
            locked, path, saver, data = (self._prompt_sets_locked, PROMPT_SETS_FILE,
                                         save_prompt_sets, self._prompt_sets)
        else:
            locked, path, saver, data = (self._prompts_locked, PROMPTS_FILE,
                                         save_prompts, self._prompts)
        if locked:
            messagebox.showerror(
                "Not saved",
                f"{path.name} couldn't be read when the app started, so saving is "
                f"disabled to avoid overwriting it. Fix or restore the file, then restart.",
                parent=self)
            return False
        try:
            saver(data)
            return True
        except Exception as exc:
            messagebox.showerror("Not saved", f"Couldn't write {path.name}:\n{exc}", parent=self)
            return False

    # ------------------------------------------------------------------
    # Prompt set management
    # ------------------------------------------------------------------

    def _refresh_prompt_set_combo(self):
        """Sync the prompt-set combobox with self._prompt_sets."""
        names = [_SINGLE_PROMPT] + [s["name"] for s in self._prompt_sets]
        self._prompt_set_combo["values"] = names
        if self._prompt_set_var.get() not in names:
            self._prompt_set_var.set(_SINGLE_PROMPT)
        self._update_set_info()

    def _update_set_info(self):
        """Update the info label below the prompt-set selector."""
        name = self._prompt_set_var.get()
        if name == _SINGLE_PROMPT:
            self._set_info_var.set("Using the single prompt editor below.")
            return
        for s in self._prompt_sets:
            if s["name"] == name:
                prompts = s.get("prompts", [])
                n = len(prompts)
                if n == 0:
                    self._set_info_var.set("Set is empty — add prompts via Edit Set.")
                else:
                    preview = ", ".join(p["name"] for p in prompts[:8])
                    if n > 8:
                        preview += f", … ({n} total)"
                    self._set_info_var.set(f"{n} prompt(s): {preview}")
                return
        self._set_info_var.set("")

    def _on_prompt_set_selected(self, _event=None):
        self._update_set_info()
        self._set_prompt_editor_enabled(self._prompt_set_var.get() == _SINGLE_PROMPT)

    def _set_prompt_editor_enabled(self, enabled: bool):
        """Enable or disable the single-prompt editor area."""
        state = "normal" if enabled else "disabled"
        self._prompt_label.configure(fg=self._prompt_label_fg if enabled else "gray")
        self._prompt_combo.configure(state="readonly" if enabled else "disabled")
        self._save_prompt_btn.configure(state=state)
        self._delete_prompt_btn.configure(state=state)
        self.prompt_text.configure(state=state,
                                   bg=self._prompt_text_bg if enabled else self.cget("bg"))

    def _new_prompt_set(self):
        """Open the set editor dialog to create a new prompt set."""
        self._open_set_editor(None)

    def _edit_prompt_set(self):
        """Open the set editor dialog for the currently selected set."""
        name = self._prompt_set_var.get()
        if name == _SINGLE_PROMPT:
            self._log("Select a prompt set first, or click 'New Set…' to create one.")
            return
        for s in self._prompt_sets:
            if s["name"] == name:
                self._open_set_editor(s)
                return

    def _delete_prompt_set(self):
        """Delete the currently selected prompt set after confirmation."""
        name = self._prompt_set_var.get()
        if name == _SINGLE_PROMPT:
            return
        if not messagebox.askyesno("Delete Prompt Set",
                                   f'Delete set "{name}"?\n\nThis cannot be undone.',
                                   icon="warning", parent=self):
            return
        self._prompt_sets = [s for s in self._prompt_sets if s["name"] != name]
        self._persist("sets")
        self._prompt_set_var.set(_SINGLE_PROMPT)
        self._refresh_prompt_set_combo()
        self._set_prompt_editor_enabled(True)

    # ------------------------------------------------------------------
    # Prompt set editor dialog
    # ------------------------------------------------------------------

    def _open_set_editor(self, existing_set: dict | None):
        """Open a dialog to create or edit a prompt set.

        If *existing_set* is None a new empty set is created; otherwise the
        existing set is edited in-place.
        """
        dlg = tk.Toplevel(self)
        dlg.title("Edit Prompt Set" if existing_set else "New Prompt Set")
        dlg.resizable(True, True)
        dlg.geometry("720x480")
        dlg.minsize(520, 350)
        dlg.grab_set()
        dlg.transient(self)

        # Working copy of prompts (list of dicts)
        if existing_set:
            work_prompts: list[dict] = [dict(p) for p in existing_set.get("prompts", [])]
        else:
            work_prompts = []
        # Index of the prompt currently loaded in the editor. Tracked separately
        # from the listbox selection, which has already moved on by the time
        # <<ListboxSelect>> fires — so edits are saved back to the right prompt.
        current = {"idx": None}

        # --- Set name ---
        name_frame = tk.Frame(dlg)
        name_frame.pack(fill="x", padx=12, pady=(12, 6))
        tk.Label(name_frame, text="Set name:").pack(side="left")
        set_name_var = tk.StringVar(value=existing_set["name"] if existing_set else "")
        tk.Entry(name_frame, textvariable=set_name_var, width=40).pack(side="left", padx=(6, 0))

        # --- Main pane: list + editor ---
        pane = tk.Frame(dlg)
        pane.pack(fill="both", expand=True, padx=12, pady=4)
        pane.columnconfigure(1, weight=1)
        pane.rowconfigure(0, weight=1)

        # Left: prompt listbox
        list_frame = tk.LabelFrame(pane, text="Prompts")
        list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        list_frame.rowconfigure(0, weight=1)

        listbox = tk.Listbox(list_frame, width=24, exportselection=False)
        listbox.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        list_sb = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        list_sb.grid(row=0, column=1, sticky="ns", pady=4)
        listbox.configure(yscrollcommand=list_sb.set)

        # Right: prompt text editor
        edit_frame = tk.LabelFrame(pane, text="Prompt text")
        edit_frame.grid(row=0, column=1, sticky="nsew")
        edit_frame.rowconfigure(1, weight=1)
        edit_frame.columnconfigure(0, weight=1)

        # Prompt name field
        pname_row = tk.Frame(edit_frame)
        pname_row.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
        tk.Label(pname_row, text="Name:").pack(side="left")
        prompt_name_var = tk.StringVar()
        prompt_name_entry = tk.Entry(pname_row, textvariable=prompt_name_var, width=30)
        prompt_name_entry.pack(side="left", padx=(4, 0))

        # Prompt text area
        prompt_edit = scrolledtext.ScrolledText(edit_frame, width=40, height=8, wrap=tk.WORD)
        prompt_edit.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        # ---- Helper functions ----

        def _refresh_listbox():
            listbox.delete(0, "end")
            for i, p in enumerate(work_prompts, 1):
                listbox.insert("end", f"{i}. {p['name']}")

        def _store_editor():
            """Copy the editor contents back into the prompt they belong to."""
            idx = current["idx"]
            if idx is None or idx >= len(work_prompts):
                return
            work_prompts[idx]["name"] = prompt_name_var.get().strip() or f"Prompt {idx+1}"
            work_prompts[idx]["text"] = prompt_edit.get("1.0", "end").strip()
            listbox.delete(idx)
            listbox.insert(idx, f"{idx+1}. {work_prompts[idx]['name']}")

        def _load_editor(idx):
            """Show prompt *idx* in the editor (None clears it)."""
            current["idx"] = idx
            prompt_edit.delete("1.0", "end")
            if idx is None:
                prompt_name_var.set("")
                return
            listbox.selection_clear(0, "end")
            listbox.selection_set(idx)
            listbox.see(idx)
            prompt_name_var.set(work_prompts[idx]["name"])
            prompt_edit.insert("1.0", work_prompts[idx]["text"])

        def _on_listbox_select(_event=None):
            sel = listbox.curselection()
            if not sel or sel[0] == current["idx"]:
                return
            _store_editor()
            _load_editor(sel[0])

        listbox.bind("<<ListboxSelect>>", _on_listbox_select)

        def _add_prompt():
            _store_editor()
            work_prompts.append({"name": f"Prompt {len(work_prompts) + 1}", "text": ""})
            _refresh_listbox()
            _load_editor(len(work_prompts) - 1)
            prompt_name_entry.focus_set()
            prompt_name_entry.select_range(0, "end")

        def _remove_prompt():
            idx = current["idx"]
            if idx is None:
                return
            work_prompts.pop(idx)
            current["idx"] = None
            _refresh_listbox()
            _load_editor(min(idx, len(work_prompts) - 1) if work_prompts else None)

        def _move(delta):
            idx = current["idx"]
            if idx is None or not 0 <= idx + delta < len(work_prompts):
                return
            _store_editor()
            work_prompts[idx], work_prompts[idx + delta] = work_prompts[idx + delta], work_prompts[idx]
            _refresh_listbox()
            _load_editor(idx + delta)

        # --- Toolbar buttons ---
        tb = tk.Frame(dlg)
        tb.pack(fill="x", padx=12, pady=(4, 2))
        tk.Button(tb, text="+ Add", command=_add_prompt).pack(side="left", padx=(0, 4))
        tk.Button(tb, text="- Remove", command=_remove_prompt).pack(side="left", padx=(0, 4))
        tk.Button(tb, text="↑ Up", command=lambda: _move(-1)).pack(side="left", padx=(0, 4))
        tk.Button(tb, text="↓ Down", command=lambda: _move(1)).pack(side="left")

        # --- Save / Cancel ---
        def _do_save():
            sname = set_name_var.get().strip()
            if not sname:
                messagebox.showwarning("Name required",
                                       "Please enter a name for this prompt set.",
                                       parent=dlg)
                return
            _store_editor()

            clash = next((s for s in self._prompt_sets
                          if s["name"] == sname and s is not existing_set), None)
            if clash and not messagebox.askyesno(
                    "Replace set?",
                    f'A set named "{sname}" already exists. Replace its prompts with these?',
                    parent=dlg):
                return

            if existing_set:
                if clash:
                    self._prompt_sets.remove(clash)
                existing_set["name"] = sname
                existing_set["prompts"] = work_prompts
            elif clash:
                clash["prompts"] = work_prompts
            else:
                self._prompt_sets.append({"name": sname, "prompts": work_prompts})

            self._persist("sets")
            self._refresh_prompt_set_combo()
            self._prompt_set_var.set(sname)
            self._on_prompt_set_selected()
            dlg.destroy()

        bottom = tk.Frame(dlg)
        bottom.pack(fill="x", padx=12, pady=(6, 12))
        tk.Button(bottom, text="Save Set", width=12, command=_do_save).pack(side="right", padx=(6, 0))
        tk.Button(bottom, text="Cancel", width=12, command=dlg.destroy).pack(side="right")

        # Populate
        _refresh_listbox()
        if work_prompts:
            _load_editor(0)

        self._center(dlg, 720, 480)

    # ------------------------------------------------------------------
    # Saved prompts (individual — existing feature)
    # ------------------------------------------------------------------

    def _refresh_prompt_combo(self):
        """Sync the combobox values with self._prompts."""
        names = [p["name"] for p in self._prompts]
        self._prompt_combo["values"] = names
        if self._saved_prompt_var.get() not in names:
            self._saved_prompt_var.set("")

    def _on_prompt_selected(self, _event=None):
        """Load the selected saved prompt into the text area."""
        name = self._saved_prompt_var.get()
        for p in self._prompts:
            if p["name"] == name:
                self.prompt_text.delete("1.0", "end")
                self.prompt_text.insert("1.0", p["text"])
                break

    def _save_prompt(self):
        """Ask for a name and save the current prompt text."""
        current_text = self.prompt_text.get("1.0", "end").strip()
        if not current_text:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Save Prompt")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self)

        tk.Label(dlg, text="Name for this prompt:").pack(padx=16, pady=(14, 4))
        name_var = tk.StringVar()

        if self._saved_prompt_var.get():
            name_var.set(self._saved_prompt_var.get())

        name_entry = tk.Entry(dlg, textvariable=name_var, width=36)
        name_entry.pack(padx=16)
        name_entry.select_range(0, "end")
        name_entry.focus_set()

        def _confirm():
            name = name_var.get().strip()
            if not name:
                return
            for p in self._prompts:
                if p["name"] == name:
                    p["text"] = current_text
                    break
            else:
                self._prompts.append({"name": name, "text": current_text})
            self._persist("prompts")
            self._refresh_prompt_combo()
            self._saved_prompt_var.set(name)
            dlg.destroy()

        btn_row = tk.Frame(dlg)
        btn_row.pack(pady=12)
        tk.Button(btn_row, text="Save", width=10, command=_confirm).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", width=10, command=dlg.destroy).pack(side="left", padx=6)

        dlg.bind("<Return>", lambda _e: _confirm())
        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        self._center(dlg)

    def _delete_prompt(self):
        """Delete the currently selected saved prompt after confirmation."""
        name = self._saved_prompt_var.get()
        if not name:
            return
        if not messagebox.askyesno("Delete Prompt",
                                   f'Delete "{name}"?\n\nThis cannot be undone.',
                                   icon="warning", parent=self):
            return
        self._prompts = [p for p in self._prompts if p["name"] != name]
        self._persist("prompts")
        self._refresh_prompt_combo()

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_state(self):
        """Populate widgets from saved config and OS keychain."""
        c = self._config

        # Restore provider first so we know which keychain slot to read.
        if c.get("provider") in PROVIDERS:
            self.provider_var.set(c["provider"])
        provider = self.provider_var.get()
        self._active_provider = provider
        if c.get(_SAVE_API_KEY) and _KEYRING_AVAILABLE:
            self.save_api_key_var.set(True)
        self.api_key_var.set(self._stored_api_key(provider))

        if c.get("input_dir"):
            self.input_var.set(c["input_dir"])
        if c.get("output_dir"):
            self.output_var.set(c["output_dir"])
        model = c.get("model", "")
        if model in MODEL_MIGRATIONS:
            self.after(200, self._log,
                       f"NOTE: {model} has been retired — switched to {MODEL_MIGRATIONS[model]}.")
            model = MODEL_MIGRATIONS[model]
        elif model in MODEL_SHUTDOWNS:
            self.after(200, self._log, f"NOTE: {model} {MODEL_SHUTDOWNS[model]}.")
        if model:
            self.model_var.set(model)
        if c.get("aspect_ratio") in GEMINI_RATIOS + GEMINI_EXTREME_RATIOS:
            self.aspect_ratio_var.set(c["aspect_ratio"])
        if c.get("image_size") in GEMINI_SIZES:
            self.image_size_var.set(c["image_size"])
        if isinstance(c.get("openai_size"), str) and validate_openai_size(c["openai_size"]) is None:
            self.openai_size_var.set(c["openai_size"])
        if c.get("openai_quality") in OPENAI_QUALITIES_25:
            self.openai_quality_var.set(c["openai_quality"])
        if "overwrite" in c:
            self.overwrite_var.set(c["overwrite"])
        if c.get("run_mode") in RUN_MODES:
            self.run_mode_var.set(c["run_mode"])
        if c.get("output_format") in OUTPUT_FORMATS:
            self.output_format_var.set(c["output_format"])
        if c.get("variations") in range(1, MAX_VARIATIONS + 1):
            self.variations_var.set(c["variations"])
        if c.get("parallel") in range(1, MAX_PARALLEL + 1):
            self.parallel_var.set(c["parallel"])
        refs = c.get("reference_images")
        if isinstance(refs, list):
            self._ref_paths = [p for p in refs if isinstance(p, str)][:MAX_REFERENCE_IMAGES]
        self._update_ref_info()

        # Now apply the provider's UI changes (model list, label swaps, etc.).
        # Suppress the API-key swap since we just populated it above.
        self._suppress_api_key_swap = True
        try:
            self._on_provider_change()
        finally:
            self._suppress_api_key_swap = False

        if not _KEYRING_AVAILABLE:
            self.save_api_key_var.set(False)
            self.after(200, self._log,
                       "NOTE: 'keyring' package not found — "
                       "run 'pip install keyring' to enable secure API key saving.")

        # Populate the saved-prompts combobox
        self._refresh_prompt_combo()

        # Populate the prompt-set combobox and restore selection
        self._refresh_prompt_set_combo()
        saved_set = c.get("selected_prompt_set", _SINGLE_PROMPT)
        set_names = [s["name"] for s in self._prompt_sets]
        if saved_set in set_names:
            self._prompt_set_var.set(saved_set)
        else:
            self._prompt_set_var.set(_SINGLE_PROMPT)
        self._update_set_info()
        self._set_prompt_editor_enabled(self._prompt_set_var.get() == _SINGLE_PROMPT)

        # Check for a pending batch job from a previous session
        batch = load_batch_state()
        if batch:
            self.resume_btn.configure(state="normal")
            self.after(200, self._log,
                       f"Pending {batch['provider']} batch job found: {batch['job_id']}\n"
                       f"  Output folder: {batch.get('output_dir', '?')}\n"
                       f"  Items: {len(batch['key_map'])}\n"
                       "Click 'Resume Batch' to reconnect and download results.")

    def _save_state(self):
        """Persist settings to config.json; API keys go to the OS keychain only."""
        save_api = self.save_api_key_var.get()
        provider = self.provider_var.get()
        self._api_keys[provider] = self.api_key_var.get().strip()

        if save_api:
            for p, key in self._api_keys.items():
                if key:
                    keychain_save_api_key(key, p)
        else:
            for p in PROVIDERS:
                keychain_delete_api_key(p)

        save_config({
            "provider":             provider,
            "input_dir":            self.input_var.get(),
            "output_dir":           self.output_var.get(),
            "model":                self.model_var.get().strip(),
            "aspect_ratio":         self.aspect_ratio_var.get(),
            "image_size":           self.image_size_var.get(),
            "openai_size":          self.openai_size_var.get().strip(),
            "openai_quality":       self.openai_quality_var.get(),
            "overwrite":            self.overwrite_var.get(),
            _SAVE_API_KEY:          save_api,
            "selected_prompt_set":  self._prompt_set_var.get(),
            "run_mode":             self.run_mode_var.get(),
            "output_format":        self.output_format_var.get(),
            "variations":           self.variations_var.get(),
            "parallel":             self.parallel_var.get(),
            "reference_images":     self._ref_paths,
        })

    # ------------------------------------------------------------------
    # Logging, progress, preview
    # ------------------------------------------------------------------

    def _ui(self, fn, *args):
        """Run *fn* on the Tk main thread — the only safe way for workers to touch widgets."""
        try:
            self.after(0, fn, *args)
        except (RuntimeError, tk.TclError):
            pass   # window already closed

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, msg: str, color: str = "gray"):
        self.status_label.configure(text=msg, fg=color)

    def _set_progress(self, percent: float, text: str):
        self.progress_var.set(percent)
        self._set_status(text, "blue")

    def _report_progress(self, stats: _RunStats):
        """Worker-side: push a progress/ETA/spend line to the status bar."""
        with stats.lock:
            done, total, in_flight = stats.done, stats.total, stats.in_flight
            processed = stats.succeeded + stats.failed
            elapsed = time.monotonic() - stats.started
        remaining = total - done
        parts = [f"{done}/{total} done"]
        if in_flight:
            parts.append(f"{in_flight} in flight")
        if processed and remaining:
            parts.append(f"~{self._format_eta(remaining * elapsed / processed)} left")
        spent = stats.spent_text()
        if spent:
            parts.append(f"spent ≈ {spent}")
        self._ui(self._set_progress, 100 * done / total if total else 0, "  ·  ".join(parts))

    def _show_preview(self, path: Path, thumb: Image.Image):
        self._preview_photo = ImageTk.PhotoImage(thumb)
        self._preview_label.configure(image=self._preview_photo, text="")
        self._preview_caption.configure(text=f"{path.name}  (click to open)")
        self._preview_path = path

    def _open_preview(self, _event=None):
        if self._preview_path and self._preview_path.exists():
            open_path(self._preview_path)

    def _open_output_folder(self):
        typed = self.output_var.get().strip()
        for folder in (self._last_output_dir, Path(typed) if typed else None):
            if folder and folder.is_dir():
                open_path(folder)
                return
        self._log("The output folder doesn't exist yet — it's created when a run starts.")

    @staticmethod
    def _format_eta(seconds: float) -> str:
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"

    @staticmethod
    def _sleep(seconds: float, should_stop) -> bool:
        """Sleep in short steps so Stop stays responsive. False if stopped early."""
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            if should_stop():
                return False
            time.sleep(min(0.5, left))
        return not should_stop()

    # ------------------------------------------------------------------
    # Run / Test / Stop
    # ------------------------------------------------------------------

    def _selected_prompts(self) -> list[tuple[str, str]] | None:
        """[(suffix, text), …] for the selected prompt set or single prompt (None if empty)."""
        selected_set = self._prompt_set_var.get()
        if selected_set != _SINGLE_PROMPT:
            prompt_set = next((s for s in self._prompt_sets if s["name"] == selected_set), None)
            # Numbering follows the set order even if a prompt is blank and skipped.
            prompts = [
                (str(i), p.get("text", ""))
                for i, p in enumerate((prompt_set or {}).get("prompts", []), 1)
                if p.get("text", "").strip()
            ]
            if not prompts:
                self._log("ERROR: Selected prompt set is empty. Add prompts via Edit Set.")
                return None
            self._log(f'Using prompt set "{selected_set}" ({len(prompts)} prompts)')
            return prompts
        prompt_text = self.prompt_text.get("1.0", "end").strip()
        if not prompt_text:
            self._log("ERROR: Prompt cannot be empty.")
            return None
        return [("mockup", prompt_text)]

    def _gather_job(self, test_image: Path | None = None) -> RunJob | None:
        """Validate the form and capture it as a RunJob (logs an error and returns None)."""
        api_key = self.api_key_var.get().strip()
        input_str = self.input_var.get().strip()
        output_str = self.output_var.get().strip()
        provider = self.provider_var.get()
        model = self.model_var.get().strip()
        spec = model_spec(model, provider)

        def fail(msg):
            self._log(f"ERROR: {msg}")
            return None

        if not api_key:
            return fail("API key is required.")
        if provider == "OpenAI" and not _OPENAI_AVAILABLE:
            return fail("OpenAI SDK not installed. Run: pip install openai")
        if not model:
            return fail("Choose a model.")
        other = infer_provider(model)
        if other not in (None, provider):
            return fail(f"{model} is an {other} model — switch Provider to {other}.")
        # Check the raw text: Path("") is "." — the current folder — not empty.
        if not output_str:
            return fail("Select an output folder.")
        if test_image is None:
            if not input_str or not Path(input_str).is_dir():
                return fail("Select a valid input folder.")
            images = collect_images(Path(input_str))
            if not images:
                return fail("No supported images found in the input folder.")
        else:
            images = [test_image]
        if provider == "OpenAI":
            size_error = validate_openai_size(self.openai_size_var.get().strip())
            if size_error:
                return fail(f"Invalid size '{self.openai_size_var.get()}': {size_error}.")
        references = [Path(p) for p in self._ref_paths]
        missing = [p for p in references if not p.is_file()]
        if missing:
            return fail(f"Reference image not found: {missing[0]}")
        prompts = self._selected_prompts()
        if prompts is None:
            return None
        batch = test_image is None and self.run_mode_var.get().startswith("Batch")
        if batch and not spec["batch"]:
            return fail(f"Batch mode isn't available for {model}. Switch to Real-time.")

        return RunJob(
            provider=provider, model=model, spec=spec, api_key=api_key,
            images=images, prompts=prompts,
            output_dir=Path(output_str) / TEST_SUBFOLDER if test_image else Path(output_str),
            overwrite=bool(test_image) or self.overwrite_var.get(),
            aspect_ratio=self.aspect_ratio_var.get(),
            image_size=self.image_size_var.get(),
            openai_size=self.openai_size_var.get().strip(),
            openai_quality=self.openai_quality_var.get(),
            output_format=self.output_format_var.get(),
            variations=self.variations_var.get(),
            parallel=self.parallel_var.get(),
            references=references,
            batch=batch,
            test=bool(test_image),
        )

    def _start(self):
        job = self._gather_job()
        if job:
            self._confirm_and_launch(job)

    def _test_one(self):
        """Run one chosen image through the current prompt(s) into <output>/_test."""
        path = filedialog.askopenfilename(
            title="Pick one image to test the prompt(s) on",
            initialdir=self.input_var.get().strip() or None,
            filetypes=[("Images", " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))),
                       ("All files", "*.*")])
        if path:
            job = self._gather_job(test_image=Path(path))
            if job:
                self._confirm_and_launch(job)

    def _confirm_and_launch(self, job: RunJob):
        """Show the cost estimate, then start the worker thread if confirmed."""
        items, skipped = build_work_items(job)
        n_outputs = sum(len(i.outputs) for i in items)
        if not items:
            self._log(f"Nothing to do — all {skipped} output image(s) already exist. "
                      "Tick 'Overwrite existing outputs' to redo them.")
            return

        if job.batch:
            pending = load_batch_state()
            if pending and not messagebox.askyesno(
                    "Pending batch job",
                    f"{pending['provider']} batch job {pending['job_id']} hasn't been "
                    "downloaded yet.\n\nStarting a new batch replaces its saved state, so "
                    "this app won't be able to download its results.\n\nStart anyway?",
                    icon="warning", parent=self):
                return

        unit_cost, note = estimate_unit_cost(job.spec, job.provider, job.image_size,
                                             job.openai_size, job.openai_quality)
        if unit_cost is not None:
            if job.batch:
                unit_cost *= 0.5
            cost_line = (f"Estimated cost: {n_outputs} image(s) × ${unit_cost:.3f} "
                         f"= ${unit_cost * n_outputs:.2f}")
            if job.batch:
                cost_line += "  (50% batch discount)"
            if note:
                cost_line += f"\n({note})"
        else:
            cost_line = (f"Images to generate: {n_outputs}\n"
                         "(No per-image price on file for this model/setting — "
                         "actual spend is shown as the run goes.)")
        if job.provider == "OpenAI":
            cost_line += ("\nOpenAI also bills for the input image(s) sent with each "
                          "request, which isn't included above.")

        if job.provider == "OpenAI":
            size_label = f"{job.openai_size} · {job.openai_quality}"
        else:
            size_label = f"{job.aspect_ratio} · {job.image_size}"
        mode_label = "Test (real-time)" if job.test else ("Batch" if job.batch else "Real-time")
        lines = [
            f"Provider: {job.provider}",
            f"Mode: {mode_label}",
            f"Model: {job.model}",
            f"Output: {size_label} · {job.output_format}",
            f"Images: {len(job.images)}  ×  Prompts: {len(job.prompts)}"
            + (f"  ×  Variations: {job.variations}" if job.variations > 1 else ""),
        ]
        if skipped:
            lines.append(f"Skipping {skipped} that already exist")
        if job.references:
            lines.append(f"Reference images: {len(job.references)}")
        shutdown = MODEL_SHUTDOWNS.get(job.model)
        if shutdown:
            lines.append(f"\n⚠ {job.model} {shutdown}.")
        if not messagebox.askyesno("Cost estimate",
                                   "\n".join(lines) + f"\n\n{cost_line}\n\nContinue?",
                                   parent=self):
            self._log("Run cancelled by user.")
            return

        self._save_state()
        self._launch(job, items, skipped)

    def _launch(self, job: RunJob, items: list[WorkItem], skipped: int):
        self._stop_event.clear()
        self._batch_cancel = False
        self._last_output_dir = job.output_dir
        self.progress_var.set(0)
        self._set_running(True)
        target = self._run_batch if job.batch else self._run_realtime
        threading.Thread(target=target, args=(job, items, skipped), daemon=True).start()

    def _set_running(self, running: bool):
        self._running = running
        idle = "disabled" if running else "normal"
        self.run_btn.configure(state=idle)
        self.test_btn.configure(state=idle)
        self.stop_btn.configure(state="normal" if running else "disabled")
        # Never offer Resume while another run is going — two workers would
        # share the same stop signal and batch state.
        has_pending = not running and load_batch_state() is not None
        self.resume_btn.configure(state="normal" if has_pending else "disabled")

    def _request_stop(self):
        if self._batch_active:
            choice = self._choice_dialog(
                "Stop batch job",
                "The batch job is still running on the server.\n\n"
                "Stop watching — close this run but keep the job. Click Resume Batch "
                "any time to download the results.\n\n"
                "Cancel job — cancel it on the server. Unfinished items are discarded.",
                ["Stop watching", "Cancel job", "Keep running"])
            if choice in (None, "Keep running"):
                return
            self._batch_cancel = choice == "Cancel job"
            self._set_status("Cancelling batch job…" if self._batch_cancel
                             else "Detaching from batch job…", "orange")
        else:
            self._set_status("Stopping — letting requests already in flight finish…", "orange")
        self._stop_event.set()

    def _on_close(self):
        if self._running:
            if self._batch_active:
                msg = ("A batch job is being watched. Quit anyway?\n\nThe job keeps running "
                       "on the server — click Resume Batch next time to download the results.")
            else:
                msg = ("A run is in progress. Quit anyway?\n\nImages already saved are "
                       "kept; requests still in flight are lost.")
            if not messagebox.askyesno("Quit?", msg, icon="warning", parent=self):
                return
            self._stop_event.set()
            self._force_exit = True
        try:
            self._save_state()
        except Exception:
            pass
        self.destroy()

    def _finish(self, stats: _RunStats, output_dir: Path, note: str = ""):
        self._batch_active = False
        self._set_running(False)
        not_run = stats.total - stats.done
        summary = (f"\nDone.  Succeeded: {stats.succeeded}  |  Skipped: {stats.skipped}  |  "
                   f"Failed: {stats.failed}")
        if not_run > 0:
            summary += f"  |  Not run: {not_run}"
        self._log(summary)
        spent = stats.spent_text()
        if spent:
            self._log(f"Spent ≈ {spent} (from the API's usage data)")
        if note:
            self._log(note)
        self._log(f"Output folder: {output_dir}")
        status = (f"Done — {stats.succeeded} succeeded, {stats.skipped} skipped, "
                  f"{stats.failed} failed" + (f", {not_run} not run" if not_run > 0 else ""))
        if spent:
            status += f"  ·  spent ≈ {spent}"
        color = "red" if stats.failed or stats.fatal else ("orange" if not_run > 0 else "green")
        self._set_status(status, color)

    # ------------------------------------------------------------------
    # Real-time worker (runs in a background thread)
    # ------------------------------------------------------------------

    def _log_run_header(self, job: RunJob, items: list[WorkItem], skipped: int):
        n_outputs = sum(len(i.outputs) for i in items)
        if job.test:
            self._ui(self._log, f"\n=== Test run: {job.images[0].name} → {job.output_dir} ===")
        else:
            self._ui(self._log, f"\nFound {len(job.images)} image(s) in {job.images[0].parent}")
        if job.provider == "OpenAI":
            options = f"Size: {job.openai_size}  |  Quality: {job.openai_quality}"
        else:
            options = f"Aspect ratio: {job.aspect_ratio}  |  Resolution: {job.image_size}"
        mode = "BATCH" if job.batch else f"real-time, {job.parallel} parallel"
        self._ui(self._log, f"Provider: {job.provider}  |  Model: {job.model}  |  {options}  |  "
                            f"Format: {job.output_format}  |  Mode: {mode}")
        if len(job.prompts) > 1 or job.variations > 1:
            self._ui(self._log, f"{len(job.prompts)} prompt(s) × {job.variations} "
                                f"variation(s) per image")
        if job.references:
            self._ui(self._log, "Reference images: "
                                + ", ".join(p.name for p in job.references))
        if skipped:
            self._ui(self._log, f"Skipping {skipped} image(s) that already exist.")
        self._ui(self._log, f"Generating {n_outputs} image(s).\n")

    def _run_realtime(self, job: RunJob, items: list[WorkItem], skipped: int):
        """Send requests (up to job.parallel at once) and save the results."""
        stats = _RunStats(total=skipped + sum(len(i.outputs) for i in items), skipped=skipped)
        try:
            job.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._ui(self._log, f"ERROR creating output folder: {exc}")
            self._ui(self._finish, stats, job.output_dir)
            return
        self._log_run_header(job, items, skipped)

        try:
            refs = [prepare_input_image(p) for p in job.references]
        except Exception as exc:
            self._ui(self._log, f"ERROR reading reference image: {exc}")
            self._ui(self._finish, stats, job.output_dir)
            return

        client = make_client(job.provider, job.api_key)
        pacer = _Pacer(MIN_REQUEST_SPACING)
        halt = threading.Event()   # set by workers: out of credits, or too many 503s

        def should_stop():
            return self._stop_event.is_set() or halt.is_set()

        workers = max(1, min(job.parallel, len(items)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gen") as pool:
            for n, item in enumerate(items, 1):
                pool.submit(self._process_item, job, client, refs, item,
                            f"[{n:>3}/{len(items)}]", pacer, stats, should_stop, halt)
        # Leaving the with-block waits for every submitted item.

        if stats.fatal:
            note = f"*** Run stopped early: {stats.fatal}"
        elif self._stop_event.is_set():
            note = "Stopped by user."
        else:
            note = ""
        self._ui(self._finish, stats, job.output_dir, note)

    def _process_item(self, job, client, refs, item: WorkItem, counter: str,
                      pacer: _Pacer, stats: _RunStats, should_stop, halt):
        """Generate one request's image(s), retrying 429/5xx; never raises."""
        try:
            self._process_item_inner(job, client, refs, item, counter,
                                     pacer, stats, should_stop, halt)
        except Exception as exc:   # a bug here must not silently drop the item
            self._ui(self._log, f"{counter} ERROR (unexpected): {sanitize_error(str(exc))}")
            stats.record(failed=len(item.outputs))
            self._report_progress(stats)

    def _process_item_inner(self, job, client, refs, item, counter,
                            pacer, stats, should_stop, halt):
        n = len(item.outputs)
        if should_stop():
            return
        label = item.image.name
        if len(job.prompts) > 1:
            label += f" · prompt {item.suffix}"
        if job.provider == "Gemini" and job.variations > 1:
            label += f" · {item.outputs[0].stem.rsplit('_', 1)[-1]}"
        log = lambda msg: self._ui(self._log, f"{counter} {msg}")

        try:
            main_image = prepare_input_image(item.image)
        except Exception as exc:
            log(f"ERROR reading {item.image.name}: {exc}")
            stats.record(failed=n)
            self._report_progress(stats)
            return
        if item.first_for_image and main_image.note:
            log(f"{item.image.name}: {main_image.note}")
        prompt = fill_placeholders(item.prompt, item.image)
        inputs = [main_image] + refs

        attempt = 0
        while True:
            if not pacer.wait_turn(should_stop):
                return   # stopped — counted as "not run"
            log(f"Processing  {label} …")
            with stats.lock:
                stats.in_flight += 1
            self._report_progress(stats)
            try:
                if job.provider == "OpenAI":
                    result = openai_generate(client, job.model, job.spec, prompt, inputs,
                                             job.openai_options(n), job.out_mime)
                else:
                    result = gemini_generate(client, job.model, job.spec, prompt, inputs,
                                             job.aspect_ratio, job.image_size)
                error = None
            except Exception as exc:
                error = classify_api_error(exc)
            finally:
                with stats.lock:
                    stats.in_flight -= 1

            if error is None:
                break
            if isinstance(error, _FatalQuota):
                with stats.lock:
                    stats.fatal = stats.fatal or (
                        "the account is out of credits or quota — check billing. "
                        f"({str(error)[:200]})")
                halt.set()
                log(f"FAILED  {label}: out of credits/quota — stopping the run.")
                stats.record(failed=n)
                self._report_progress(stats)
                return
            if isinstance(error, _RateLimitRetry):
                attempt += 1
                if attempt >= MAX_RETRIES:
                    log(f"FAILED  {label}: still rate-limited after {MAX_RETRIES} attempts.")
                    stats.record(failed=n)
                    self._report_progress(stats)
                    return
                log(f"Rate limited — pausing all requests {error.wait}s "
                    f"(retry {attempt}/{MAX_RETRIES - 1})…")
                pacer.cool_down(error.wait)
                continue
            if isinstance(error, _ServiceUnavailable):
                attempt += 1
                if attempt >= MAX_RETRIES:
                    log(f"FAILED  {label}: model unavailable after {MAX_RETRIES} attempts. "
                        f"Last error: {error.detail[:300]}")
                    stats.record(failed=n)
                    self._report_progress(stats)
                    # Circuit breaker: stop if too many consecutive 503 failures
                    with stats.lock:
                        stats.consecutive_503 += 1
                        tripped = stats.consecutive_503 >= CIRCUIT_BREAKER_THRESHOLD
                        if tripped and not stats.fatal:
                            stats.fatal = (f"{CIRCUIT_BREAKER_THRESHOLD} items in a row failed "
                                           "with 503 — the service looks degraded. "
                                           "Try again later.")
                    if tripped:
                        halt.set()
                    return
                base = min(30 * (2 ** (attempt - 1)), 300)
                wait = int(base + random.uniform(0, base * 0.5))
                log(f"Unavailable: {error.detail[:200]}")
                log(f"Waiting {wait}s (retry {attempt}/{MAX_RETRIES - 1})…")
                if not self._sleep(wait, should_stop):
                    return
                continue
            log(f"ERROR  {label}: {sanitize_error(str(error))}")
            stats.record(failed=n)
            self._report_progress(stats)
            return

        with stats.lock:
            stats.consecutive_503 = 0   # reset on any successful API call
        saved = 0
        for (data, mime), path in zip(result.images, item.outputs):
            try:
                thumb = save_output_image(data, mime, path, job.out_mime)
            except Exception as exc:
                log(f"ERROR saving {path.name}: {exc}")
                continue
            saved += 1
            log(f"saved → {path.name}")
            self._ui(self._show_preview, path, thumb)
        missing = n - saved
        if missing and not result.images:
            log(f"WARNING  {label}: {result.reason}")
        elif missing and len(result.images) < n:
            log(f"WARNING  {label}: only {len(result.images)} of {n} image(s) came back.")
        stats.record(succeeded=saved, failed=missing, cost=result.cost if saved else None)
        self._report_progress(stats)

    # ------------------------------------------------------------------
    # Batch worker (runs in a background thread)
    # ------------------------------------------------------------------

    def _run_batch(self, job: RunJob, items: list[WorkItem], skipped: int):
        """Upload inputs, submit everything as one Batch API job, then watch it."""
        stats = _RunStats(total=skipped + sum(len(i.outputs) for i in items), skipped=skipped)
        try:
            job.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._ui(self._log, f"ERROR creating output folder: {exc}")
            self._ui(self._finish, stats, job.output_dir)
            return
        self._log_run_header(job, items, skipped)

        backend = make_batch_backend(job.provider, job.api_key)
        uploaded: list[str] = []   # file names/ids to delete once results are in
        try:
            # --- Phase 1: upload each distinct image (and reference) once ---
            ref_parts = []
            for path in job.references:
                name, part = backend.upload_image(prepare_input_image(path))
                uploaded.append(name)
                ref_parts.append(part)
            unique_images = list(dict.fromkeys(item.image for item in items))
            image_parts = {}
            for i, path in enumerate(unique_images, 1):
                if self._stop_event.is_set():
                    raise _Cancelled
                prepared = prepare_input_image(path)
                if prepared.note:
                    self._ui(self._log, f"{path.name}: {prepared.note}")
                name, image_parts[path] = backend.upload_image(prepared)
                uploaded.append(name)
                self._ui(self._set_progress, i / len(unique_images) * 25,
                         f"Uploading images… ({i}/{len(unique_images)})")
            self._ui(self._log, f"Uploaded {len(unique_images)} image(s).")

            # --- Phase 2: build the JSONL and create the batch job ---
            self._ui(self._set_status, "Submitting batch job…", "blue")
            with tempfile.TemporaryDirectory() as tmp:
                lines_path = Path(tmp) / "requests.jsonl"
                with lines_path.open("w", encoding="utf-8") as f:
                    for item in items:
                        line = backend.request_line(job, item, [image_parts[item.image]] + ref_parts)
                        f.write(json.dumps(line) + "\n")
                if self._stop_event.is_set():
                    raise _Cancelled
                job_id, extra_files = backend.submit(job, lines_path)
                uploaded += extra_files
        except _Cancelled:
            self._ui(self._log, "Cancelled before the batch job was submitted.")
            backend.cleanup(uploaded)
            self._ui(self._finish, stats, job.output_dir)
            return
        except Exception as exc:
            self._ui(self._log, f"ERROR submitting batch job: {sanitize_error(str(exc))}")
            backend.cleanup(uploaded)
            stats.record(failed=stats.total - skipped)
            self._ui(self._finish, stats, job.output_dir)
            return

        state = {
            "provider": job.provider,
            "job_id": job_id,
            "model": job.model,
            "image_size": job.image_size,
            "out_mime": job.out_mime,
            "output_dir": str(job.output_dir),
            "key_map": {item.key: [str(p) for p in item.outputs] for item in items},
            "skipped": skipped,
            "uploaded_files": uploaded,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._ui(self._log, f"Batch job created: {job_id}")
        try:
            save_batch_state(state)
            self._ui(self._log, "Job state saved — you can safely close the app and "
                                "resume later via 'Resume Batch'.")
        except Exception as exc:
            self._ui(self._log, f"WARNING: couldn't save the batch state ({exc}) — keep "
                                "the app open until the job finishes.")
        self._watch_batch(backend, state, stats)

    def _resume_batch(self):
        """Resume a previously submitted batch job from saved state."""
        state = load_batch_state()
        if not state:
            self._log("No pending batch job found.")
            self._set_running(False)
            return
        provider = state["provider"]
        if provider == self.provider_var.get():
            api_key = self.api_key_var.get().strip()
        else:
            api_key = self._stored_api_key(provider)
        if not api_key:
            messagebox.showwarning(
                "API Key Required",
                f"This batch job was submitted to {provider}. Switch Provider to "
                f"{provider} and enter its API key before resuming.", parent=self)
            return
        if provider == "OpenAI" and not _OPENAI_AVAILABLE:
            self._log("ERROR: OpenAI SDK not installed. Run: pip install openai")
            return

        key_map = state["key_map"]
        self._log(f"\nResuming {provider} batch job: {state['job_id']}")
        self._log(f"  Output folder: {state['output_dir']}")
        self._log(f"  Items: {len(key_map)}")

        self._stop_event.clear()
        self._batch_cancel = False
        self._last_output_dir = Path(state["output_dir"])
        self._set_running(True)
        stats = _RunStats(total=state.get("skipped", 0) + sum(len(v) for v in key_map.values()),
                          skipped=state.get("skipped", 0))
        backend = make_batch_backend(provider, api_key)
        threading.Thread(target=self._watch_batch, args=(backend, state, stats),
                         daemon=True).start()

    def _watch_batch(self, backend, state: dict, stats: _RunStats):
        """Poll a submitted batch job, then download and save its results.

        The saved batch state is cleared only once there's nothing left to
        fetch — results saved, the job ended without results, or the user
        cancelled it — so a dropped connection never orphans a paid-for job.
        """
        job_id = state["job_id"]
        output_dir = Path(state["output_dir"])
        uploaded = state.get("uploaded_files", [])
        self._batch_active = True
        self._ui(self.progress_var.set, 30)
        self._ui(self._log, f"Watching batch job — checking every {BATCH_POLL_INTERVAL}s…")

        poll_failures = 0
        last_status = None
        started = time.monotonic()
        while True:
            if self._stop_event.is_set():
                if self._batch_cancel:
                    try:
                        backend.cancel(job_id)
                    except Exception as exc:
                        self._ui(self._finish, stats, output_dir,
                                 f"Couldn't cancel the job ({sanitize_error(str(exc))}). "
                                 "It may still be running — Resume Batch to check on it.")
                        return
                    clear_batch_state()
                    backend.cleanup(uploaded)
                    self._ui(self._finish, stats, output_dir, "Batch job cancelled.")
                else:
                    self._ui(self._finish, stats, output_dir,
                             "Stopped watching — the job is still running. "
                             "Click 'Resume Batch' any time to download the results.")
                return

            try:
                status, finished, has_results, detail = backend.poll(job_id)
                poll_failures = 0
            except Exception as exc:
                poll_failures += 1
                self._ui(self._log, f"  Poll error ({poll_failures}): "
                                    f"{sanitize_error(str(exc))}")
                if poll_failures >= 10:
                    self._ui(self._finish, stats, output_dir,
                             "Too many consecutive poll failures. The job may still be "
                             "running — click 'Resume Batch' to try again later.")
                    return
                self._sleep(BATCH_POLL_INTERVAL, self._stop_event.is_set)
                continue

            if status != last_status:
                self._ui(self._log, f"  Status: {status}")
                last_status = status
            self._ui(self._set_status, f"Batch status: {status}  ·  watching for "
                                       f"{self._format_eta(time.monotonic() - started)}", "blue")
            if finished:
                break
            self._sleep(BATCH_POLL_INTERVAL, self._stop_event.is_set)

        if not has_results:
            self._ui(self._log, f"\nBatch job ended without results ({status})."
                                + (f"\n  Error: {sanitize_error(detail)}" if detail else ""))
            stats.record(failed=stats.total - stats.done)
            clear_batch_state()
            backend.cleanup(uploaded)
            self._ui(self._finish, stats, output_dir)
            return

        # --- Download and save results ---
        self._ui(self._set_status, "Downloading results…", "blue")
        self._ui(self._log, "\nDownloading results…")
        key_map = state["key_map"]
        out_mime = state.get("out_mime", "image/png")
        seen = set()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            for key, outcome in backend.results(job_id, state):
                paths = [Path(p) for p in key_map.get(key, [])]
                if not paths or key in seen:
                    continue
                seen.add(key)
                if isinstance(outcome, str):
                    self._ui(self._log, f"  FAILED [{key}]: {sanitize_error(outcome)[:300]}")
                    stats.record(failed=len(paths))
                else:
                    saved = 0
                    for (data, mime), path in zip(outcome.images, paths):
                        try:
                            thumb = save_output_image(data, mime, path, out_mime)
                        except Exception as exc:
                            self._ui(self._log, f"  ERROR saving {path.name}: {exc}")
                            continue
                        saved += 1
                        self._ui(self._log, f"  Saved: {path.name}")
                        self._ui(self._show_preview, path, thumb)
                    if saved < len(paths):
                        self._ui(self._log, f"  WARNING [{key}]: "
                                            f"{outcome.reason or 'fewer images than requested'}")
                    stats.record(succeeded=saved, failed=len(paths) - saved,
                                 cost=outcome.cost if saved else None)
                self._ui(self.progress_var.set, 30 + len(seen) / max(len(key_map), 1) * 70)
        except Exception as exc:
            self._ui(self._finish, stats, output_dir,
                     f"ERROR downloading results: {sanitize_error(str(exc))}\n"
                     "The results are still on the server — click 'Resume Batch' to try again.")
            return

        missing = [k for k in key_map if k not in seen]
        if missing:
            self._ui(self._log, f"  {len(missing)} request(s) came back with no result.")
            stats.record(failed=sum(len(key_map[k]) for k in missing))
        clear_batch_state()
        backend.cleanup(uploaded)
        self._ui(self._finish, stats, output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = App()
    app.mainloop()
    if app._force_exit:
        # Closed mid-run: don't wait for in-flight requests (saves are atomic,
        # so nothing is left half-written).
        os._exit(0)


if __name__ == "__main__":
    main()
