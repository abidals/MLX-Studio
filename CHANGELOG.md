# Changelog

All notable changes to Alis Studio are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version lives in exactly one place — `studio/__version__` (in `studio/__init__.py`).
`pyproject.toml` reads it via `[tool.setuptools.dynamic]`, the server injects it into the
web UI, and the DMG build stamps it into the app bundle.

## [Unreleased]

### Added
- **Viggle Turbo (Qwen-Image-2.1)** — Viggle's DMD2-distilled 6-step student of Qwen-Image-2.1
  (`Viggle/Qwen-Image-2.1-viggle-turbo` v0.2.1), shipped as a rank-256 LoRA that is applied at
  runtime — **never merged** into the (4/8-bit quantized) transformer, exactly as diffusers does —
  and samples with the model's distilled sigma schedule: **6 steps, no CFG** (8 steps for dense
  small text). Text-to-image and strength-based img2img ship now; multi-reference instruction
  editing follows when mflux's 2.1 edit variant learns LoRA mappings. Build options: **8-bit**
  and **4-bit** (the 17.5 GB Qwen3-VL text encoder is never quantized, so both peak ~47 GB at
  1024² — comfortable on 64 GB; measured). The ~33 GB base weights auto-download via mflux on
  first Generate; the 1.3 GB adapter is downloadable and deletable in the model manager.
  **License note:** both the adapter and the base model are under the Qwen RESEARCH LICENSE —
  **non-commercial use only**.
- **Live preview for the new backend** — the in-progress-latent decoder in `mflux_common` now
  understands `QwenImage21` (unpack via `Qwen21LatentCreator`), so the eye toggle streams frames
  for it like the other mflux models.

### Changed
- **mflux is pinned to a main-HEAD commit** (PRs #756 + #768) until a release ships Qwen-Image-2.1
  LoRA support — the runtime-LoRA path and the common adapter key formats the Viggle backend
  depends on landed one day after 0.20.0. The Krea 2 backend installs with `--no-deps`, because
  its v0.3.1 pins (`mflux<0.19`, `mlx<0.32`) would otherwise clash with the new pin (v0.3.1 was
  verified to run fine on mflux 0.20 / mlx 0.32). See `requirements.txt`.

## [0.9.1]
## [0.9.1] — 2026-07-14

### Added
- **LoRA on Krea 2 Turbo.** The LoRA library now feeds the Krea 2 backend too, via `krea2-alis-mlx`
  0.3.0's runtime adapters — the adapter is applied as a low-rank branch over the (quantized)
  transformer, never merged into the weights, so LoRAs **stack** and **revert cleanly with no
  reload**. Same shared library and per-LoRA strength as the mflux models; pick LoRAs built for
  Krea 2 (a wrong-base LoRA is rejected with a clear message before any denoising). The `krea2-alis-mlx`
  dependency is pinned to **v0.3.0** (the release that added `set_loras`).

### Changed
- The "Applying LoRA(s)…" status and the LoRA hint no longer say "re-fuses the model" — accurate for
  the mflux backends but not for Krea 2's runtime-branch swap; the wording now fits both.

## [0.9.0] — 2026-07-07

### Added
- **ERNIE-Image Turbo** — Baidu's 8B text-to-image model (Apache-2.0, April 2026), the strongest
  option here for **text rendering and structured layouts** (posters, comics, labels) where the
  letters have to be right. Distilled — clean images in ~8 steps. Best with English or Chinese
  prompts. img2img and LoRA included. Offered in **8-bit** and **bf16**; downloads ~32 GB on first
  use. It's a roomy-Mac model: mflux loads the full weights before quantizing, so even 8-bit peaks
  near full precision at load (there's no pre-quantized repo to stream onto a small Mac), hence the
  ~48 GB floor and the RAM gate. *4-bit is deliberately not offered* — it would shrink the resident
  footprint but not the load peak, so it buys a constrained Mac nothing while implying it can run.
  (Resolves the ERNIE part of #26.)
- **Live preview** — watch the image form. When on (a new eye toggle next to the safety shield, and
  the default), the app decodes a few downscaled in-progress frames during a run and streams them
  under the progress bar — the "preview mode" people know from CivitAI, for iterating on a prompt
  without waiting for the full render. Throttled to ~3 frames per run (and only the first image of a
  batch) so the cost is bounded; turn it off for maximum speed. Available on the models whose
  in-progress latents can be decoded — **Z-Image, CyberRealistic Z, ERNIE-Image, Qwen-Image, and
  FLUX** (the toggle hides itself for the others). Fully best-effort: a decode hiccup silently drops
  the preview and never affects the actual generation or Stop. (Resolves #24.)

## [0.8.3] — 2026-07-03

### Fixed
- **Top bar was clipped under the titlebar** in the native window (v0.8.2 regression): macOS can
  still draw titlebar material over the first ~28 px even with a transparent titlebar, so the top
  bar's content was sliced. The app now injects a same-colour titlebar spacer that owns that strip
  (and doubles as the window drag area) — nothing real renders under the material anymore.

## [0.8.2] — 2026-07-03

### Changed
- **The app now looks like a Mac app, not a wrapped web page.** In the native window: the title
  bar is transparent and hidden — the traffic lights sit directly on the app's own top bar
  (unified-toolbar chrome, applied on the main thread), the top bar's passive areas drag the
  window, a native **View** menu (Toggle Gallery / Edit an Image / Focus Prompt) appears in the
  menu bar, the browser context menu is suppressed on chrome (kept for text fields and images),
  and the window dims its chrome when inactive.
- Web-tea removed everywhere (browser too): UI chrome is no longer selectable text (inputs stay
  selectable), cursors are native (pointer on controls, default elsewhere), focus rings are the
  app's clay accent instead of the browser default, buttons have a subtle press state, and
  scrollbars are thin macOS-style overlays.
- **⌘G** toggles the gallery; **⌘Enter** still generates.

## [0.8.1] — 2026-07-03

### Added
- **FLUX.2 klein 4B** — BFL's January-2026 fast model (Apache-2.0, **ungated** — no HF login,
  unlike FLUX.1): ~4-step distilled, ~15 GB download, 4-bit build runs on a 16 GB Mac. img2img and
  LoRA included. The 9B sibling stays out (gated, restrictive license).
- **Generation queue** — while a run is in flight, a **Queue** button appears next to Stop: it
  snapshots the current prompt + model + settings as the next job. Queued jobs run back-to-back
  (each gets its own seed when auto-seed is on); the badge shows how many are waiting, and **Stop
  clears the queue**.

## [0.8.0] — 2026-07-03

### Added
- **LoRA support** — the capability people run ComfyUI for, without the node graph. Every mflux
  model (Z-Image, CyberRealistic Z, Qwen-Image, Qwen-Image Edit, FLUX) gets a **LoRA** section:
  a shared library (add by **Civitai/HF download URL** or a local `.safetensors` path), check the
  ones to apply, and set a per-LoRA strength (0–2). Multiple LoRAs stack; Civitai's
  ComfyUI/kohya/diffusers key formats are recognized automatically (mflux). Changing the set
  reloads the model (seconds — LoRA is fused at load). Auth-gated Civitai downloads work with
  `CIVITAI_API_TOKEN` set.
- **Restore settings** — every gallery item now stores its full recipe (model, build, size, steps,
  seed, guidance, LoRAs + strengths). A new lightbox button restores all of it — model switched,
  controls repopulated, prompt refilled — so any past image is one click from a re-run or a tweak.
  The recipe is also embedded in the PNG itself (`tEXt "alis"`), like ComfyUI's workflow-in-PNG.
- **Auto-seed** — a new toggle next to the seed dice rolls a fresh random seed on every Generate
  (the seed box always shows what was used, so results stay reproducible).

## [0.7.5] — 2026-07-03

### Added
- **CyberRealistic Z** — the first community model adopted from **Civitai**: Cyberdelia's
  photorealism finetune of Z-Image Turbo (v4.0, taken from the creator's own HF mirror,
  CreativeML OpenRAIL-M). Converted from the ComfyUI single-file checkpoint to the mflux layout
  (fused-qkv split + key remap, verified key-complete both ways) and re-hosted pre-quantized —
  **4-bit (~5.5 GB) runs on a 16 GB Mac**, 8-bit (~10 GB) for bigger machines. Same ~9-step /
  no-CFG / multilingual recipe as Z-Image Turbo, img2img included.
  - Candidates that didn't make the cut: Moody Pro Mix (no-derivatives license), RedCraft KREA2
    (INT8-ConvRot only, no bf16 source — would double-quantize).

### Changed
- **Memory honesty across models** (3-lens review): on Macs with ≤ 32 GB, loading a model now frees
  the one another backend had cached — A/B-ing Z-Image against its finetune used to stack two ~6 GB
  pipelines and could OOM a 16 GB machine. Stock Z-Image 8-bit/bf16 gained explicit RAM floors
  (24 / 32 GB — they quantize on the fly from the ~33 GB bf16 repo); CyberRealistic 8-bit wants
  ≥ 24 GB.
- An unknown build id now errors instead of silently substituting (and mislabeling) the default
  build; a failed NSFW-filter load is logged instead of silently passing images through unfiltered.
- Model backends touch the repo's `config.json` on first download so Hugging Face counts the
  download (mflux itself never fetches it).

## [0.7.4] — 2026-07-02

### Added
- **Bring your own image** — the canvas editor now takes *your* pictures, not just generated ones:
  an **Edit** button in the top bar opens a file picker, and you can **drag-drop** an image onto the
  editor or **paste (⌘V)** one from the clipboard anywhere in the app. If marks are already on the
  canvas, replacing the image asks first ("Replace the image?").
- **Krea 2 Turbo learns img2img** — the flagship model now takes an **Input image + Strength** like
  the rest: the picture is encoded with the sampler's own VAE and the rectified-flow schedule is
  entered at the matching timestep (lower strength = fewer steps = faster). Via `krea2-alis-mlx`
  0.2.0; picked up automatically by the DMG build.
- **Paste routes to the right place** — with the editor open, a pasted image replaces the editing
  base; with it closed, it lands in the **Input image** (img2img) of the selected model and the drop
  zone highlights; a text-to-image-only model would open the editor instead.
- The settings panel now **says why** the Input image control would be missing on a
  text-to-image-only model (none ship today — Krea 2 Turbo was the last, and it just learned).

## [0.7.3] — 2026-07-02

### Added
- **Shape tools** — the canvas editor grows from Sketch/Text to five tools: **Sketch, Circle, Box,
  Arrow, Text**. Shapes are drag-drawn with a live preview; a bare click with a shape tool is
  ignored (no accidental dots).
- **Stroke sizes** — Thin / Medium / Thick, applied to strokes, shapes, and text.
- **Redo** — full undo/redo for every mark (including text), with **⌘Z / ⇧⌘Z** while the editor
  is open (text fields keep their native undo).
- **Step history** — every edit result joins a thumbnail strip (*Original → Step 1 → …*); click a
  step to put it back on the canvas and continue from there (results also live in the gallery).
- **Fast / Fine quality** — 4-step (default) or 8-step edits, one tap.
- Each run now uses a **fresh random seed**, so "Edit" again on the same marks gives a new take.
- **Discard guards** — closing the editor (✕ / backdrop / Esc) or switching steps now asks before
  throwing away unsaved marks; Esc during a drag abandons just that drag; Clear can be brought back
  with Redo. ⌘Z/⇧⌘Z use the physical key, so undo works under a Korean input source.

### Changed
- Editor side panel reorganized ("Quiet Workshop" pass): sectioned Mark it / Describe the change /
  Steps, an icon toolbar, size dots, and disabled-state Undo/Redo. The base image is pre-rendered
  once, so long strokes stay smooth on big images; coordinates are guarded against a not-yet-laid-out
  canvas.

## [0.7.2] — 2026-07-02

### Added
- **Canvas editor (draw + instruct)** — hit **Edit** on any image (a fresh result or a gallery item)
  to open a Gemini-style editor: **Sketch** freehand in a chosen color, drop **Text** labels (click a spot →
  a transparent-background text box), then describe the change ("make the circled area blue"). The
  marks are baked into the image sent to **Qwen-Image Edit**, which follows them and paints the
  drawing out. The result replaces the canvas so you can **keep editing**, and every step is saved
  to the gallery. Colour palette, Undo, and Clear included.
  - Requires the Qwen-Image Edit backend (≥ 64 GB Mac); the same RAM confirm/gate applies before a run.

## [0.7.1] — 2026-07-02

### Added
- **Qwen-Image Edit** — instruction-based image editing (Tongyi, Apache-2.0): attach an image and
  describe the change ("make the hat red", understands Korean and other languages). Unlike
  strength-based img2img this follows an edit instruction; the output keeps the input's aspect ratio,
  normalized to ~1 MP (≈1024²). Offered in **8-bit** (needs ~64 GB RAM) and **bf16** (~96 GB);
  downloads ~54 GB on first use. The model picker now **warns** — and the app refuses with a confirm
  override — when a chosen build's RAM floor exceeds this Mac's memory (measured peaks: 8-bit ~39 GB,
  bf16 ~58 GB).
  - *4-bit is deliberately not offered* — mflux's 4-bit quantization of this model decodes to grainy
    noise regardless of step count (reproduced with the stock mflux CLI; same upstream issue that
    removed Qwen-Image 4-bit in #9).

### Changed
- The generation result now reports the **actual output size** (edit models keep the input's size;
  others may round to a multiple of 16) instead of the requested size.
- **Krea 2 Turbo** 2K sampling now pins the timestep shift to `mu = 1.15` at 2048² (matches Krea's
  own 2K recipe) instead of extrapolating past it (`krea2-alis-mlx` 0.1.2).

### Fixed
- Upscale status now hints at the first-use download size, and the gallery-save path was
  de-duplicated (single `_gallery_write` primitive).

## [0.7.0] — 2026-07-01

### Added
- **Image-to-image** — attach an image and transform it with a prompt. Z-Image Turbo, Qwen-Image,
  and FLUX get a drag-and-drop **Input image** control plus a **Strength** slider (Krea 2 Turbo
  stays text-to-image). The uploaded image never leaves your Mac; the NSFW filter still runs on the
  output.
- **Upscale (2× / 3×)** — a diffusion super-resolution step powered by **SeedVR2** (3B, Apache-2.0).
  Open any gallery image and upscale it to a crisp 2K–4K; the result is saved back to the gallery.
  Downloads on first use; auto-hidden on Macs under 24 GB.
- **Gallery lightbox** — click an image (or its prompt) to open a full view with the **full,
  editable prompt** and icon actions: Use (send the edited prompt to Generate), Copy, Download,
  Delete. Cards also get a quick Copy-prompt button.

### Changed
- **Larger resolutions per model** — Krea 2 Turbo up to **2048²** (it's a native 1K–2K model),
  Qwen-Image up to 1536², Z-Image / FLUX up to 1280². On a low-memory Mac the resolution control
  now **warns** when a chosen size may not fit (≤16 GB → above 1024², ≤24 GB → above 1536²).

## [0.6.3] — 2026-06-29

### Fixed
- **Qwen-Image 4-bit produced noisy/grainy images** ([#9](https://github.com/avlp12/alis-studio/issues/9)).
  Qwen-Image's ~20B transformer is too sensitive to 4-bit quantization — mflux blanket-quantizes its
  AdaLN modulation and output-projection layers, so the composition is right but the image is grainy
  (mflux's own docs warn that ≤6-bit "degrades a lot more compared to Flux"). Reproduced and confirmed
  4-bit noisy vs 8-bit clean. **Removed the 4-bit option for Qwen-Image; it now defaults to 8-bit**
  (bf16 also available). FLUX 4-bit is unaffected and stays.

## [0.6.2] — 2026-06-29

### Added
- **Recommended model for your Mac.** On launch, Alis Studio detects your unified memory and marks
  the best-fitting model **★ Recommended** in the picker — and selects it by default: **Z-Image Turbo**
  on a 16 GB Mac, **Krea 2 Turbo** on ≥ 24 GB. Each backend declares a `min_ram_gib` floor and
  `/api/system` now returns `recommended` / `recommended_label`.

### Changed
- README updated for Z-Image Turbo, 16 GB-Mac support, the low-memory tiling, and the recommendation.

## [0.6.1] — 2026-06-29

### Added
- **Lower memory on small Macs**: the mflux models (Z-Image Turbo, Qwen-Image, FLUX) now use mflux's
  VAE tiling for large (≥1024²) renders on ≤24 GB Macs, cutting Z-Image 4-bit's 1024² peak from
  ~12.9 GB to ~8.5 GB with no visible quality change (verified across every supported resolution).
  Smaller renders (≤768², which already fit) and big-RAM Macs keep the exact untiled decode; force it
  on or off with `ALIS_VAE_TILING=1` / `0`.
- A **"Loading…" status** now shows while a model is being loaded for the first time (and, on the very
  first use, while it downloads its weights) — so the wait isn't a blank progress bar.

### Changed
- The progress bar now advances **cumulatively across a multi-image batch** instead of restarting at
  step 1 for each image.

## [0.6.0] — 2026-06-28

### Added
- **Z-Image-Turbo model** (Tongyi-MAI / Alibaba, Apache-2.0): a ~6B single-stream DiT with a
  Qwen3-4B text encoder (understands Korean and other languages natively) and the FLUX VAE,
  distilled to ~9 steps with no CFG. At **4-bit (~6 GB) it runs on a 16 GB Mac** (comfortable at
  512–768px; 1024px is tight) — the first backend that doesn't effectively need a big-RAM machine
  the way the 12.9B Krea 2 Turbo default does. Powered by mflux (no model port), it downloads on
  first use: the 4-bit pre-quantized build (~6 GB) by default, or 8-bit/bf16 (~33 GB) from the
  official repo. Adds `mflux>=0.18` as a direct dependency.
- **Live progress bar + responsive Stop for the mflux backends** (Z-Image, Qwen-Image, FLUX):
  generation now reports each denoising step to the UI, and the Stop button interrupts
  mid-generation instead of only taking effect after the current image finishes.

## [0.5.4] — 2026-06-28

### Added
- **Gallery**: generated images are now saved to disk (`~/Library/Application Support/Alis Studio/
  gallery/`) with their prompt and settings, and a new **Gallery** view (top-bar toggle) shows your
  past work as a grid — reuse a prompt, download, or delete each one. Adds `GET /api/gallery`,
  `GET /api/gallery/<id>.png`, and `POST /api/gallery/delete`. (History starts now — images from
  earlier sessions weren't persisted.)

### Fixed
- Scrolling no longer rubber-bands the whole window into blank space (WKWebView elastic overscroll).
  The header is now a fixed bar and only the content area scrolls.

## [0.5.3] — 2026-06-28

### Changed
- **New brand icon**: a pine tree (echoing the project's profile imagery) with a generative
  sparkle and a "STUDIO" wordmark on the clay squircle — generated by Alis Studio itself. The
  in-app top-bar logo now uses the same pine mark, so the app icon, the in-app logo, and the
  profile image share one identity.

## [0.5.2] — 2026-06-28

### Fixed
- **Image generation crashed with `There is no Stream(gpu, 0) in current thread`** — on recent MLX
  this hit every generation through the app (and always after using the prompt enhancer), making
  0.5.0–0.5.1 unable to generate. MLX keeps its default GPU stream per-thread, but the stdlib
  threaded server ran model work on whichever request thread arrived, and the enhancer's mlx-lm
  corrupted that state further. Fix: all MLX work (generation, NSFW filter, model-cache scan, and
  registry build) now runs on a single dedicated GPU thread warmed up at startup, and the prompt
  enhancer's mlx-lm runs in its own isolated subprocess so it can't touch the image model's GPU state.

## [0.5.1] — 2026-06-28

### Added
- **Prompt enhancer (optional, fully on-device)**: a Settings toggle enables a local LLM
  (`Qwen3-4B-Instruct-2507`, via mlx-lm) that rewrites the prompt — translating non-English
  prompts (e.g. Korean) into English and enriching them into vivid image captions. Click the
  **Enhance** button to rewrite the prompt in place; you review/edit the English before
  generating, so seeds stay reproducible. Deterministic (greedy + cached). Adds
  `GET /api/system` and `POST /api/enhance`.
- **Memory-aware gating**: on Macs with ≤ 24 GB the toggle defaults off and asks for
  confirmation before enabling (the ~2.3 GB enhancer model competes with the image model for
  unified memory). The self-contained `.dmg` bundles mlx-lm; the model downloads on first use.

## [0.5.0] — 2026-06-27

### Added
- **Per-model prompt-language hint**: a one-line note under the prompt shows whether the active
  model understands non-English prompts natively (Krea 2 Turbo / Qwen-Image, via their Qwen text
  encoders) or works best in English (FLUX, via T5/CLIP) — so Korean users know Krea 2 Turbo and
  Qwen-Image take Korean directly, no translation needed.
- **Stop button**: while an image is generating, the Generate button becomes a Stop button.
  Clicking it cancels the run — the frontend aborts the request and `POST /api/cancel` flags the
  server, which halts at the next denoise step and reports `cancelled`. (Per-step backends like
  Krea 2 Turbo stop promptly; a fresh generation afterwards is unaffected.)
- App version is now shown in the UI — a small tag next to the brand in the top bar — and
  in the native window's title bar. New `GET /api/version` endpoint exposes it programmatically.
- **Self-contained `.dmg`**: `packaging/build_dmg.sh` builds an `Alis Studio.app` that bundles
  its own standalone Python interpreter **and every runtime dependency** (mlx, mflux, transformers,
  pywebview, the Krea 2 Turbo backend, …) under `Contents/Resources`. Double-click to run — no
  system Python, no `pip install`. The bundle is ad-hoc signed and wrapped in a drag-to-Applications
  disk image. `packaging/make_icon.py` renders the app's sparkle mark into `AppIcon.icns`.
  (Model **weights** are not bundled — they download from Hugging Face on the first generation.)

### Changed
- Single source of truth for the version: `pyproject.toml` no longer hard-codes it
  (`dynamic = ["version"]`), so the package version can never drift from `studio.__version__`.
- `pyproject.toml` uses the PEP 639 SPDX license expression (`license = "MIT"`) and requires
  `setuptools>=77`, replacing the deprecated `license = { text = "MIT" }` table form.

## [0.4.0] — 2026-06-27

### Added
- Native desktop app: runs in its own window via pywebview (WKWebView on macOS), no browser tab.

## [0.3.0] — 2026-06-27

### Added
- Multi-model support: Qwen-Image and FLUX (schnell/dev) backends alongside Krea 2 Turbo.
- Model names are shown in the UI; model picker reworked into a button + popover.

## [0.2.0] — 2026-06-27

### Added
- LM Studio-style detailed settings panel and an in-app model manager/downloader.

## [0.1.0] — 2026-06-27

### Added
- Initial release: a local, model-agnostic, standard-library image-generation studio for
  Apple silicon (MLX), with the pure-MLX Krea 2 Turbo backend.

[0.5.4]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.4
[0.5.3]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.3
[0.5.2]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.2
[0.5.1]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.1
[0.5.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.5.0
[0.4.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.4.0
[0.3.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.3.0
[0.2.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.2.0
[0.1.0]: https://github.com/avlp12/alis-studio/releases/tag/v0.1.0
