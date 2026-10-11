# Bulk Image Generator

A simple desktop GUI for batch-processing images through Google Gemini *or* OpenAI image-generation models. Point it at a folder of images, write a prompt, and it generates a new AI image for each one — saving the results to an output folder of your choice.

Originally built to create framed wall art product mockups from print designs (walnut, oak, and white wood frames in a mid-century modern interior), but works for any image transformation prompt.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## Features

- Browse to select input and output folders
- Write any prompt — or use the built-in default
- **Two providers** — switch between Google Gemini and OpenAI from a single dropdown; size/quality controls only offer what the selected model supports
- **Prompt sets** — group multiple prompts together and run every image through all of them in one go
- **`{filename}` placeholder** — replaced in each prompt with the image's file name
- **Reference images** — optional extra images (e.g. a room scene) sent with every request, so a whole product line gets consistent mockups
- **Test 1 image** — try your prompt(s) on a single image before paying for a whole folder
- **Parallel requests** — run up to 6 requests at once (real-time mode); a shared pacer backs every worker off together when the API rate-limits
- **Two run modes** (both providers, model permitting):
  - **Real-time** — live progress, ETA, running spend, and a thumbnail of the latest result
  - **Batch (50% off)** — submits all work as one async job via the [Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch-api) or [OpenAI Batch API](https://developers.openai.com/api/docs/guides/batch)
- **Output format** — PNG, JPEG or WebP, plus up to 4 **variations** per prompt
- **Cost estimate** before each run, and **actual spend** (from the APIs' usage data) as it runs
- Remembers your folders, provider, and settings between sessions
- Optional secure API key saving via the OS keychain (Windows Credential Manager / macOS Keychain / Linux Secret Service) — never stored in plaintext, with separate slots per provider
- Skips already-processed images (with optional overwrite); output files are written atomically, so a crash never leaves a half-written image that gets skipped later
- Handles rate limits (429) and server overload (503) automatically with exponential backoff, jitter, and a circuit breaker — and stops the run straight away if the account is out of credits

## Requirements

- Python 3.10 or later
- Whichever provider(s) you want to use:
  - **Gemini** — a Google AI Studio account with **billing enabled** (no free tier for image gen — [enable billing here](https://aistudio.google.com/plan_information)) and a `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/apikey)
  - **OpenAI** — an OpenAI account with image-generation access and an `OPENAI_API_KEY` from the [OpenAI dashboard](https://platform.openai.com/api-keys)

## Installation

```bash
git clone https://github.com/robbiemueller/bulk-image-gen.git
cd bulk-image-gen
pip install -r requirements.txt
```

> `tkinter` is part of the Python standard library. On some Linux distros you may need to install it separately: `sudo apt install python3-tk`

## Setup

Set the API key(s) for the provider(s) you'll use as environment variables before running:

```bash
# macOS / Linux
export GOOGLE_API_KEY="your_gemini_key_here"
export OPENAI_API_KEY="your_openai_key_here"

# Windows (PowerShell)
$env:GOOGLE_API_KEY = "your_gemini_key_here"
$env:OPENAI_API_KEY = "your_openai_key_here"
```

You can also paste the key directly into the **API Key** field in the GUI. Check **Save API key** to store it securely in your OS keychain so you don't need to re-enter it each time — it is never written to disk in plaintext. Each provider has its own keychain slot, and a key you type for one provider is kept if you switch to the other.

## Usage

```bash
python generate_mockups.py
```

1. Enter your API key (pre-filled if saved or if the matching environment variable is set)
2. Pick a **provider** (Gemini or OpenAI), then a **model** — the hint next to it shows what it's for and what it costs
3. Set the **image** controls: aspect ratio + resolution (Gemini) or size + quality (OpenAI)
4. Browse to your **input folder** (containing the images to process)
5. Browse to your **output folder** (where results will be saved)
6. Optionally set the **format**, **variations**, **parallel requests**, and **reference images**
7. Choose a **prompt set** or write a single prompt (see below)
8. Click **Test 1 image…** to check the result on one image, then **Run**

### Single prompt mode

Leave the prompt set dropdown on **"-- Single prompt --"** and write your prompt in the text area. Output files are saved as `<original_name>_mockup.png`.

### Prompt sets (multi-prompt batch)

Prompt sets let you run every image through multiple prompts in one go — useful for generating several different mockup styles per design.

1. Click **New Set…** to open the prompt set editor
2. Give the set a name (e.g. "Etsy Mockups")
3. Click **+ Add** to add prompts — give each a short name and write the prompt text
4. Use **↑ Up** / **↓ Down** to reorder prompts (the order determines output numbering)
5. Click **Save Set**
6. Select your set from the dropdown and click **Run**

Output files are numbered by prompt order: `<original_name>_1.png`, `<original_name>_2.png`, etc. With variations, each gets a `_v1`, `_v2`… suffix. If two inputs share a name (`art.jpg` and `art.png`), the source extension is added so neither overwrites the other (`art_jpg_1.png`).

Prompt files are saved atomically and the previous version is kept as `prompt_sets.json.bak` / `prompts.json.bak`. If a prompt file can't be read at startup, the app warns you and refuses to save over it.

### Reference images and `{filename}`

Click **Add…** next to **References** to attach up to 10 extra images to every request — for example a room photo so every mockup uses the same scene. Your input image is always sent first, then the references in order, so you can write prompts like *"Place the artwork from the first image into the room shown in the second image."*

`{filename}` anywhere in a prompt is replaced with the input image's file name (without extension).

### Test 1 image

**Test 1 image…** asks you to pick one image, runs it through the current prompt or prompt set in real-time, and saves the results to an `_test` subfolder of the output folder (always overwriting), so a later full run doesn't skip them.

### Run modes

#### Real-time (default)

Requests run in parallel (set **Parallel requests**, 1–6). You see live progress, an ETA, actual spend so far, and a thumbnail of the latest result (click it to open the file). If the API rate-limits, every worker pauses together; if the account runs out of credits, the run stops immediately instead of retrying each image.

#### Batch (50% off)

All work is submitted as a single asynchronous job. The app uploads your images, submits the job, then polls until it completes and downloads the results. Available for every Gemini model and for `gpt-image-2.5-sunburst` / `gpt-image-2.5-flare` on OpenAI; the dropdown locks to Real-time for models without Batch support.

**Advantages:**
- **50% cheaper** — batch pricing is half the real-time rate
- **Higher rate limits** — avoids the 503/429 errors common in large real-time runs
- **Fire and forget** — submit a large job and let it run

**Trade-offs:**
- No per-image progress during processing — you see the job status, not individual images
- Turnaround is variable — usually minutes, but up to 24 hours
- Results arrive all at once when the job completes

**Stopping and resuming:**
Once a batch job is submitted, its state is saved locally. You can safely close the app, restart your computer, or come back later — click **Resume Batch** to reconnect and download results. Pressing **Stop** while a batch is running asks whether to **stop watching** (the job keeps running; resume later) or **cancel the job** on the server. The saved state is only cleared once the results are downloaded or the job has ended, so a dropped connection never loses a paid-for job.

## Models & pricing

### Gemini

| Model | Notes | Resolutions | Cost per image |
|---|---|---|---|
| `gemini-nano-banana-2.1` | Newest (Nano Banana 2.1), fast — recommended | 1K, 2K, 4K | $0.034 / $0.050 / $0.113 |
| `gemini-3-pro-image` | Nano Banana Pro — highest quality, slower | 1K, 2K, 4K | $0.134 (1K/2K) · $0.24 (4K) |
| `gemini-3.1-flash-image` | Nano Banana 2 — previous generation | 512, 1K, 2K, 4K | $0.045–$0.151 |
| `gemini-3.1-flash-lite-image` | Nano Banana 2 Lite — fastest | 1K | $0.034 |

All support 10 standard aspect ratios (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `4:5`, `5:4`, `2:3`, `3:2`, `21:9`); Nano Banana 2 and 2.1 also support `1:4`, `4:1`, `1:8`, `8:1`. Batch mode halves these prices.

The old `-preview` model IDs have been retired; saved settings that use them are switched to the replacement automatically. See [Google AI pricing](https://ai.google.dev/gemini-api/docs/pricing) for current rates.

### OpenAI

| Model | Notes | Batch |
|---|---|---|
| `gpt-image-2.5-sunburst` | Most capable — best when the artwork must stay exact | ✓ |
| `gpt-image-2.5-flare` | Faster everyday generation | ✓ |
| `gpt-image-2` | Previous generation | — |

`gpt-image-1` (shuts down Oct 23, 2026), `gpt-image-1.5` and `gpt-image-1-mini` (Dec 1, 2026) are no longer listed; you can still type them as a custom model ID until then, and the app shows a warning.

`gpt-image-2` cost per image at the common sizes:

| Size | Low | Medium | High |
|---|---|---|---|
| `1024x1024` | $0.006 | $0.053 | $0.211 |
| `1024x1536` | $0.005 | $0.041 | $0.165 |
| `1536x1024` | $0.005 | $0.041 | $0.165 |

The 2.5 models are billed by tokens ($8/1M image input, $30/1M image output) and OpenAI hasn't published a per-image table yet, so the estimate shows "unknown" — the running spend (computed from each response's usage data) shows what you're actually paying. OpenAI also bills for the input image(s) sent with each edit, which the up-front estimate doesn't include.

**Size:** pick a preset or type any `WIDTHxHEIGHT` — both edges multiples of 16, max 3840 px, no wider than 3:1, between 655,360 and 8,294,400 total pixels. **Quality:** `auto`, `low`, `medium`, `high`, plus `xhigh` and `max` on the 2.5 models.

See [OpenAI image-generation docs](https://developers.openai.com/api/docs/guides/image-generation) for current rates.

Before each run, the tool shows a **cost estimate** based on the number of images still to generate, model, and size/quality — and asks you to confirm. When quality or size is `auto`, the estimate is labelled as an upper bound.

### Custom model IDs

The model box is editable — type any model ID and it's remembered between sessions. Custom models get every size/quality option and no price estimate.

## Tips

- **Aspect ratio / size** — set this to match what you want in the output, not the input. The model will crop/compose accordingly.
- **Resolution / quality** — `1K` (Gemini) or `1024x1024` low/medium (OpenAI) is fine for web use. Step up for print or large displays. Higher resolutions/qualities cost more and take longer.
- **Prompt** — be specific about style, framing, lighting, and context. The model uses both your image and the prompt together.
- **Failed images** — re-run with *Overwrite* unticked to retry only what failed; everything already saved is skipped.
- **Refusals** — if a model declines or a safety filter blocks an image, the log shows the reason and any text the model returned.
- **Rate limits / 503 errors** — handled automatically. If 5 consecutive items fail with 503 after all retries, the run stops (circuit breaker). For large runs, **Batch mode** avoids rate limits and costs 50% less.

## Supported input formats

`.jpg`, `.jpeg`, `.png`, `.webp` are sent as-is. `.bmp`, `.tif`/`.tiff` and `.gif` (first frame) are converted to PNG in memory first. Images larger than 15 MB or 4096 px on the long edge are downscaled before sending.

## License

MIT
