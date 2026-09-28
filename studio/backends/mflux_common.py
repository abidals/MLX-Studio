"""Shared infrastructure for the mflux-backed backends (FLUX, Qwen-Image, Z-Image).

Kept in a neutral module so no backend imports another: both ``mflux_models`` and ``z_image``
import the live-progress bridge and the low-memory VAE-tiling policy from here.
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

# --- low-memory policy: VAE tiling for large decodes on constrained Macs ----------------------
# A large VAE decode is the memory spike: measured on Z-Image 4-bit, the 1024² peak is ~12.9 GB
# untiled vs ~8.5 GB tiled (output visually identical; ≤768² already fits untiled at ≤9.8 GB).
# So tile only (a) on a Mac that needs it and (b) for genuinely large outputs.
_TILE_RAM_GIB = 24                 # ≤24 GiB Macs (16/24 GB) — where an untiled 1024² risks pressure.
_TILE_MIN_PIXELS = 1024 * 1024     # only tile ≥1024²; ≤768² keeps the exact untiled decode.
_constrained: bool | None = None   # memoized hardware check (RAM doesn't change at runtime)


def _total_ram_gib() -> float:
    try:  # canonical on macOS
        import subprocess
        out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
        except Exception:
            return 0.0


def _is_constrained() -> bool:
    global _constrained
    if _constrained is None:
        ram = _total_ram_gib()
        _constrained = 0 < ram <= _TILE_RAM_GIB
    return _constrained


def _tiling_enabled() -> bool:
    """Whether VAE tiling is wanted. ``ALIS_VAE_TILING=1/0`` forces it on/off (lets a big-RAM user
    opt in, or a constrained user keep the exact decode); otherwise it auto-enables on low-RAM Macs."""
    v = os.environ.get("ALIS_VAE_TILING")
    if v is not None:
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return _is_constrained()


def _apply_memory_policy(model, width: int, height: int) -> None:
    """Turn mflux's VAE tiling on for a large decode on a memory-constrained Mac, off otherwise.
    ``tiling_config`` is just a flag read at decode time (no weight change), so it is safe to toggle
    per generation on a cached model. Big-RAM Macs and ≤768² outputs keep the exact untiled decode.
    Applies to any mflux backend, though in practice only Z-Image runs on such Macs. Best-effort —
    a tiling-config change must never break generation."""
    try:
        from mflux.models.common.vae.tiling_config import TilingConfig
        tile = _tiling_enabled() and (width * height) >= _TILE_MIN_PIXELS
        model.tiling_config = TilingConfig() if tile else None
        if tile:
            _log.info("VAE tiling enabled for %dx%d decode (low-memory policy)", width, height)
    except Exception as e:
        _log.warning("VAE tiling policy not applied: %s", e)


# --- live progress bridge ---------------------------------------------------------------------
_PREVIEW_MAX_PX = 384   # longest side of a streamed preview frame — small, it's a thumbnail


def _decode_preview_latents(model, latents, config):
    """Decode in-loop latents to a small preview PIL image. Best-effort: returns None on anything
    unexpected and NEVER raises — a preview is a nicety and must not affect generation.

    mflux's decode path isn't uniform across models, so we dispatch: Z-Image / CyberRealistic-Z /
    ERNIE expose ``_decode_latents``; Qwen-Image and FLUX unpack with their own latent creator and
    the shared VAE util. Every path reuses the model's own ``tiling_config`` (set by the memory
    policy), so a preview decode has the same peak as the final one — safe on constrained Macs.
    Models not handled here (FLUX.2 klein, Qwen-Image-Edit, Krea 2) simply get no preview."""
    try:
        from mflux.utils.image_util import ImageUtil
        decoded = None
        dec = getattr(model, "_decode_latents", None)
        if dec is not None:
            import inspect
            # Z-Image / CyberRealistic-Z take (latents, config); ERNIE takes (latents) only. Pick by
            # signature rather than catching TypeError, so a real error inside decode isn't masked.
            if "config" in inspect.signature(dec).parameters:
                decoded = dec(latents=latents, config=config)
            else:
                decoded = dec(latents=latents)
        else:
            cls = type(model).__name__
            from mflux.models.common.vae.vae_util import VAEUtil
            tiling = getattr(model, "tiling_config", None)
            if cls == "QwenImage":
                from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
                lat = QwenLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
                decoded = VAEUtil.decode(vae=model.vae, latent=lat, tiling_config=tiling)
            elif cls == "QwenImage21":
                from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
                lat = Qwen21LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
                decoded = VAEUtil.decode(vae=model.vae, latent=lat, tiling_config=tiling)
            elif cls == "Flux1":
                from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
                lat = FluxLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
                decoded = VAEUtil.decode(vae=model.vae, latent=lat, tiling_config=tiling)
        if decoded is None:
            return None
        gen = ImageUtil.to_image(
            decoded_latents=decoded, config=config, seed=0, prompt="",
            quantization=getattr(model, "bits", None), generation_time=0,
            lora_paths=getattr(model, "lora_paths", None), lora_scales=getattr(model, "lora_scales", None),
        )
        im = gen.image.copy()
        im.thumbnail((_PREVIEW_MAX_PX, _PREVIEW_MAX_PX))
        return im
    except Exception as e:
        _log.info("live preview decode unavailable for %s: %s", type(model).__name__, e)
        return None


class _StepProgress:
    """Bridges mflux's per-step callback to Alis Studio's step_callback(step, total).

    Registered on an mflux model's callback registry, it fires once per denoise step so the
    UI can show a live progress bar — and, because step_callback raises when the user clicks
    Stop, it also lets a generation be interrupted mid-loop instead of only after it finishes.
    The Stop signal (server._Cancelled) is a plain Exception, NOT a KeyboardInterrupt, so it
    escapes mflux's `except KeyboardInterrupt` denoise loop and surfaces as a cancel — don't
    change _Cancelled to subclass KeyboardInterrupt or Stop becomes a silently-completed run.

    When ``preview_cb`` is set it also decodes a downscaled in-progress frame every few steps and
    hands it to ``preview_cb(step, total, pil_image)`` — the "Live preview" feature. Preview decode
    is throttled (≤ a few per run) and only for the first image of a batch, so its cost is bounded;
    it is fully best-effort and disables itself for the run on the first failure.
    """

    def __init__(self, step_callback, base=0, batches=1, preview_cb=None, model=None):
        # base = index of the current image in a multi-image batch; batches = batch size.
        # Reporting (base*total + step, batches*total) makes the bar advance monotonically across
        # the whole batch instead of restarting from 1 for each image.
        self._cb = step_callback
        self._base = base
        self._batches = batches
        self._preview_cb = preview_cb
        self._model = model
        self._preview_dead = False   # a failed decode disables further preview attempts this run

    def call_in_loop(self, *, t, seed, prompt, latents, config, time_steps):
        total = getattr(config, "num_inference_steps", 0) or len(time_steps)
        step = t + 1
        self._cb(self._base * total + step, self._batches * total)   # progress + Stop first (cheap, always)
        if self._preview_cb and not self._preview_dead and self._base == 0 and self._wants_preview(step, total):
            im = _decode_preview_latents(self._model, latents, config)
            if im is None:
                self._preview_dead = True   # unsupported model or decode failed — stop trying for this run
            else:
                try:
                    self._preview_cb(step, total, im)
                except Exception as e:
                    self._preview_dead = True
                    _log.info("live preview emit failed: %s", e)

    @staticmethod
    def _wants_preview(step, total):
        """Throttle to ~3–4 frames spread across the run, skipping the final step (the real decode
        happens there anyway). `round(total/4)` keeps low step-counts from decoding every step (e.g.
        a 6-step run previews at 2 and 4, not 1–5) while big step-counts stay bounded."""
        if step >= total:
            return False
        every = max(1, round(total / 4))
        return step % every == 0


def _wire_progress(model, step_callback, base=0, batches=1) -> None:
    """Attach a single per-step progress callback to an mflux model, replacing any previous one
    (the model is cached across generations, so we must not stack subscribers; re-wired per image
    in a batch so progress is cumulative). Best-effort: progress is a nicety, so a change in mflux
    internals must never break generation — but leave a trace, since a silent failure would degrade
    Stop to job-boundary granularity.

    If ``step_callback`` carries a ``preview`` attribute (set by the server when the user enabled
    Live preview), in-progress frames are streamed too — see ``_StepProgress``."""
    try:
        preview_cb = getattr(step_callback, "preview", None)
        model.callbacks.in_loop = [_StepProgress(step_callback, base, batches,
                                                 preview_cb=preview_cb, model=model)]
    except Exception as e:
        _log.warning("mflux progress/Stop wiring failed: %s", e)


# --- image-to-image (shared by every mflux backend; mflux's generate_image takes image_path/strength) ---
def _img2img_params():
    """Optional input-image + strength controls. The UI renders type 'image' as a drop zone and
    puts the picked image (data URI) in params['init_image']; the server decodes it to a temp file
    and sets params['image_path'] before generate()."""
    return [
        {"key": "init_image", "label": "Input image", "type": "image", "group": "Image-to-image",
         "hint": "Optional — attach (or paste) an image to transform it with your prompt (img2img). "
                 "It's scaled to the Resolution setting, so match the aspect ratio to avoid stretching."},
        {"key": "strength", "label": "Strength", "type": "float", "group": "Image-to-image",
         "min": 0.1, "max": 1.0, "step": 0.05, "default": 0.6,
         "hint": "How much to change the input — higher = more change; lower also runs fewer steps "
                 "(faster), and with few steps nearby values can land on the same step. 1.0 ignores "
                 "the input entirely."},
    ]


def _img2img_args(params):
    """Resolve (image_path, image_strength) for a generate_image() call from params; (None, None)
    means plain txt2img. image_path is set by the server after decoding the uploaded image."""
    path = params.get("image_path")
    if not path:
        return None, None
    try:
        strength = float(params.get("strength", 0.6))
    except (TypeError, ValueError):
        strength = 0.6
    return path, strength


def _lora_params():
    """The LoRA control. The UI renders type 'loras' as the LoRA library (checkbox + scale per
    entry, add-by-URL/path); the server validates the picked names against the library directory
    and rewrites params['loras'] to [{'path': <abs path>, 'scale': float}, ...]."""
    return [
        {"key": "loras", "label": "LoRA", "type": "loras", "group": "LoRA",
         "hint": "Style/subject adapters applied on top of the model — pick ones made for THIS "
                 "model family. On Civitai copy the Download button's link (not the page URL). "
                 "Changing the set is applied before the run (a short pause before the first step)."},
    ]


def _lora_args(params):
    """(lora_paths, lora_scales) for an mflux constructor, or (None, None). Server-resolved."""
    loras = params.get("loras") or []
    paths = [entry.get("path") for entry in loras if entry.get("path")]
    if not paths:
        return None, None
    scales = []
    for entry in loras:
        if not entry.get("path"):
            continue
        try:
            scales.append(float(entry.get("scale", 1.0)))
        except (TypeError, ValueError):
            scales.append(1.0)
    return paths, scales


def _lora_sig(params):
    """Hashable signature of the LoRA selection — part of the model-cache key (mflux applies LoRA
    at construction, so a different set means a reload). NOTE: only meaningful AFTER the server's
    _resolve_loras rewrote library names to paths."""
    paths, scales = _lora_args(params)
    if not paths:
        return ()
    return tuple(zip(paths, scales))


def _construct_checking_lora(builder, lora_paths):
    """Run a model constructor; when LoRAs are requested, capture mflux's fuse log and FAIL LOUDLY
    if any LoRA matched 0 keys — mflux silently no-ops on a wrong-base file (e.g. a FLUX LoRA on
    Z-Image), which would otherwise look like "LoRA does nothing". The captured log is re-printed
    so server logs keep mflux's own output."""
    if not lora_paths:
        return builder()
    import contextlib
    import io
    import re
    import sys
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = builder()
    text = buf.getvalue()
    sys.stdout.write(text)
    for m in re.finditer(r"\((\d+)/(\d+) keys matched\)", text):
        if m.group(1) == "0":
            raise ValueError("A selected LoRA didn't match this model (0 keys fused) — it's probably "
                             "made for a different base model. Uncheck it, or switch to the model it "
                             "was trained for.")
    return model
