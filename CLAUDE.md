# CLAUDE.md

## Project overview

Single-file Python GUI tool (`generate_mockups.py`) that batch-processes a folder of images through Google Gemini *or* OpenAI image-generation models and saves the results. Built with `tkinter` (stdlib) + `google-genai` + `openai` + `Pillow`.

## Architecture

Everything lives in one file — no package structure, no modules. Keep it that way unless the scope grows significantly.

- **Model registry** — `MODEL_SPECS` is the single source of truth per model: provider, blurb, sizes/ratios (Gemini) or sizes/qualities (OpenAI), `batch` support, per-image `cost`, and token prices for actual-spend tracking. `MODELS_BY_PROVIDER` / `DEFAULT_MODEL` derive from it. `model_spec()` returns a permissive generic spec for custom model IDs. `MODEL_MIGRATIONS` auto-replaces retired IDs in saved config; `MODEL_SHUTDOWNS` shows a warning for IDs that still work but are on the way out.
- **`collect_images()`** — scans input folder for supported image types
- **`prepare_input_image()`** — returns bytes ready to send (`PreparedImage`); BMP/TIFF/GIF are converted to PNG and anything over 15 MB / 4096 px is downscaled. `lru_cache`d so prompt sets don't re-read the same file.
- **`build_work_items()`** — expands images × prompts × variations into `WorkItem`s, skipping outputs that exist. OpenAI gets one item per image+prompt (variations via `n`); Gemini gets one item per variation. `output_stems()` disambiguates colliding stems (`art.jpg` + `art.png`).
- **`gemini_generate()` / `openai_generate()`** — one request each; return a `GenResult` (images, reason when none came back, actual cost). `gemini_result()` / `openai_result()` parse both SDK objects and batch-result dicts via `_field()`.
- **`classify_api_error()`** — maps SDK exceptions to `_RateLimitRetry` (429), `_ServiceUnavailable` (5xx / network), or `_FatalQuota` (OpenAI `insufficient_quota`, Gemini 402 credits depleted or per-day quota) — fatal stops the whole run. Don't key fatal detection on "check your plan and billing": Gemini says that on ordinary per-minute 429s.
- **`save_output_image()`** — writes atomically (`.part` + `os.replace`), converts format only when needed, returns a thumbnail for the preview.
- **`App(tk.Tk)`** — the entire GUI. `_on_provider_change()` swaps the model list and API key; `_on_model_change()` rebuilds the size/quality dropdowns, hint, and run-mode lock from the model's spec.
- **Real-time worker** — `_run_realtime()` runs `_process_item()` on a `ThreadPoolExecutor` (`parallel` workers). `_Pacer` spaces request starts and applies a shared cool-down after a 429. `_RunStats` holds counters under an `RLock`. Circuit breaker: 5 consecutive items failing with 503 stops the run.
- **Run modes** — *Real-time* or *Batch* (async job at 50% cost). Batch is offered only when the model's spec has `batch: True` (all Gemini models; `gpt-image-2.5-*` on OpenAI).
- **Batch backends** — `_GeminiBatch` (File API uploads + JSONL, `batches.create`) and `_OpenAIBatch` (files with `purpose="vision"`, JSONL on `/v1/images/edits` with `images: [{"file_id": …}]`, `purpose="batch"`). Same interface: `upload_image`, `request_line`, `submit`, `poll`, `cancel`, `results`, `cleanup`. `_run_batch()` submits; `_watch_batch()` polls/downloads for both new and resumed jobs.
- **Batch resilience** — after job creation, `batch_state.json` stores provider, job id, key→output paths, uploaded file ids. It's cleared only when results are saved, the job ended without results, or the user cancelled — never on poll/download failure. Stop during a batch asks "Stop watching" vs "Cancel job". `load_batch_state()` upgrades files from older versions.
- **Prompt sets** — named, ordered groups of prompts stored in `prompt_sets.json`; each image is processed through every prompt in the selected set. The set editor tracks the prompt shown in the editor (`current["idx"]`) separately from the listbox selection.
- **Output naming** — single-prompt mode: `{stem}_mockup.{ext}`; prompt-set mode: `{stem}_1.{ext}`, `{stem}_2.{ext}`… (numbered by prompt order); variations add `_v1`, `_v2`…
- **Test 1 image** — runs one chosen image in real-time into `<output>/_test` (always overwrites).

## Key constraints

- **`threading.Event` for stop signal** — use `self._stop_event` (not a plain bool) to signal workers to stop; check it before every request and during every wait
- **`self._ui(fn, *args)`** for all UI updates from worker threads — it wraps `self.after(0, …)` and swallows the error if the window is gone. Never touch tkinter widgets directly from a background thread
- **`ttk` widgets for coloured buttons** — `tk.Button` ignores `bg`/`fg` on macOS. `style.theme_use("clam")` restyles every ttk widget, not just buttons
- **`TkFixedFont`** for the log widget — cross-platform monospace; don't hardcode `Consolas` or `Menlo`
- **API key** — read from the active provider's env var (`GOOGLE_API_KEY` for Gemini, `OPENAI_API_KEY` for OpenAI), the keychain, or the GUI field; never log or print it. `sanitize_error()` redacts `key=`, `sk-…` and `AIza…`
- **Keychain** — when "Save API key" is checked, keys are stored via `keyring` under a per-provider service name (`gemini-bulk-image-gen` / `openai-bulk-image-gen`). Unchecking it deletes both. Never written to `config.json` or any file on disk.
- **JSON files** — written with `_write_json_atomic()`. Prompt files keep a `.bak`, and if one can't be parsed at startup, saving is blocked (`_prompts_locked` / `_prompt_sets_locked`) so it isn't overwritten.

## Dependencies

```
google-genai>=1.73,<2
openai>=2.32,<3
Pillow>=10.0.0
keyring>=24.0.0
tkinter  # stdlib, no install needed
```

Upper bounds are deliberate: both SDKs have newer major versions (openai 3.x, google-genai 2.x) that haven't been tested with this code.

## Running locally

```bash
# install deps
pip install -r requirements.txt

# set API key (whichever provider you'll use)
export GOOGLE_API_KEY="your_gemini_key_here"   # macOS/Linux
export OPENAI_API_KEY="your_openai_key_here"
$env:GOOGLE_API_KEY = "your_gemini_key_here"   # Windows PowerShell
$env:OPENAI_API_KEY = "your_openai_key_here"

# run
python generate_mockups.py
```

## Model notes

### Gemini
- `gemini-nano-banana-2.1` — newest (Oct 2026), 1K/2K/4K (no 512), extra aspect ratios (1:4, 4:1, 1:8, 8:1)
- `gemini-3-pro-image` — highest quality, 1K/2K/4K, standard ratios only
- `gemini-3.1-flash-image` — 512/1K/2K/4K, extra aspect ratios
- `gemini-3.1-flash-lite-image` — fastest, 1K only
- The `-preview` IDs (`gemini-3-pro-image-preview`, `gemini-3.1-flash-image-preview`) passed their shutdown date (June 25, 2026) — migrated via `MODEL_MIGRATIONS`
- Image size `"512"` has no `K` suffix (API requirement); all others use `"1K"`, `"2K"`, `"4K"` (uppercase)
- All use `client.models.generate_content()` with `types.ImageConfig` for aspect ratio and resolution; all support `batchGenerateContent`. Google's docs now show the Interactions API, but `generate_content` still works for every model above
- Gemini 3 image models may return `thought` parts — skipped when collecting images
- Imagen models use a different API (`generate_images`), don't accept a reference image, and are shut down — not supported

### OpenAI
- `gpt-image-2.5-sunburst` (most precise edits) and `gpt-image-2.5-flare` (faster) — Sept 2026, support Batch and quality `xhigh` / `max`
- `gpt-image-2` — previous generation, no Batch here, quality up to `high`
- `gpt-image-1` shuts down Oct 23, 2026; `gpt-image-1.5`, `gpt-image-1-mini`, `chatgpt-image-latest` Dec 1, 2026 (in `MODEL_SHUTDOWNS`)
- Uses `client.images.edit(image=..., model=..., prompt=..., size=..., quality=..., output_format=..., n=...)`; `image` may be a list (input first, then references)
- Sizes: `auto` or any `WIDTHxHEIGHT` passing `validate_openai_size()` (multiples of 16, ≤3840 px edge, ≤3:1, 655,360–8,294,400 px)
- Don't send `input_fidelity` to gpt-image-2+ — inputs are always high fidelity
- Response is base64 (`result.data[i].b64_json`) plus `usage` tokens, used for actual spend

### Cost estimate
`MODEL_SPECS[model]["cost"]` — Gemini keyed by image size (`"1K"`); OpenAI keyed by `"{size}/{quality}"` (`"1024x1024/medium"`). `estimate_unit_cost()` labels `auto` size/quality as an upper bound and returns None when nothing is on file (shown as "cost unknown"). Actual spend comes from usage data: `openai_actual_cost()` (token prices) and `gemini_actual_cost()` (per-image price + input tokens); both halve for Batch.
