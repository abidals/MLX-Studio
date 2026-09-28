"""Viggle Turbo — a few-step distilled student of Qwen-Image-2.1 (Tongyi / Qwen).

Production checkpoint: ``Viggle/Qwen-Image-2.1-viggle-turbo`` v0.2.1 — a DMD2-distilled LoRA
(rank 256; an SVD-truncated rank 128 is also shipped) that sits on top of the base Qwen-Image-2.1
transformer and samples in **6 denoising steps with no classifier-free guidance**, covering
text-to-image and strength-based img2img (multi-reference instruction editing is a follow-up —
mflux's 2.1 edit variant doesn't wire LoRA mappings yet).

Three rules from the model card shape this backend, and each one is load-bearing:

  1. **The sigma schedule.** Raw nodes ``[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]`` sampled in 6
     steps (8 steps for dense small text). Keep ``0.875/0.75/0.5/0.25`` fixed and only move the
     high-noise end ``1 → 0.875``. ViggleTurboScheduler applies the same resolution-dependent
     shift the base pipeline applies to its default nodes.
  2. **shift_terminal must be null.** The base model's scheduler config stretches sigma to a
     0.02 terminal, which "would wreck the last step" for the turbo student — so the model
     config override below sets ``sigma_shift_terminal = None`` (Viggle's repo ships its own
     scheduler config with exactly that change).
  3. **The LoRA is never merged.** Merging into bf16 is lossy (round-to-nearest keeps ~70% of
     the update on average), so the adapter is applied at runtime — mflux's ``bake_lora=False``
     wraps the (possibly quantized) linear layers in ``LoRALinear`` and computes ``W·x + s·B·A·x``
     exactly as diffusers does. Scale stays at 1.0 (alpha equals rank).

License: the adapter and the base model are under the **Qwen RESEARCH LICENSE — non-commercial
use only** (research or evaluation). See the NOTICE file in the Viggle repo for attribution.

Needs mflux with Qwen-Image-2.1 LoRA support (unreleased at the time of writing — the
requirements pin a commit). Space: ~33 GB base weights (mflux-managed, fetched on first use,
then quantized to 4/8-bit at load — the adapter is tiny, 1.3 GB). Measured peak at 1024² is
~47 GB for both builds — the 17.5 GB Qwen3-VL text encoder is never quantized and dominates —
so this is a big-memory-Mac model: comfortable on 64 GB, like mflux's own guidance.
"""

from __future__ import annotations

import os

from .base import Backend
from .mflux_common import (_apply_memory_policy, _construct_checking_lora, _img2img_args,
                           _img2img_params, _lora_args, _lora_params, _lora_sig, _wire_progress)

VIGGLE_REPO = "Viggle/Qwen-Image-2.1-viggle-turbo"
# The shipped v0.2.1 adapters (the card marks v0.2.1 "use this one"; the 4-step v0.1/v0.2 files
# are kept upstream for reproducibility only and are not offered here).
SHIPPED_RANK = "r256"   # every build uses the shipped adapter; the r128 cut stays available upstream
ADAPTER_FILES = {
    "r256": "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
    "r128": "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
}
ADAPTER_SIZE_GB = {"r256": 1.3, "r128": 0.68}

# Raw sigma nodes per supported step count — the v0.2.1 6-step schedule plus the flagship's
# documented neighbors: 4 is the training schedule; 5/6/7 split ONLY the high-noise segment
# 1 → 0.875 evenly (the low-noise nodes the student was trained on never move); 8 is the card's
# dense-text schedule (adds 0.625 and 0.125). Schedules outside this table are rejected rather
# than invented — uniform linspace with other step counts makes every image visibly softer.
_RAW_SIGMAS = {
    4: (1.0, 0.75, 0.5, 0.25),
    5: (1.0, 0.875, 0.75, 0.5, 0.25),
    6: (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25),
    7: (1.0, 11 / 12, 5 / 6, 0.875, 0.75, 0.5, 0.25),
    8: (1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125),
}

_SCHEDULE_NAME = "viggle-turbo"
_SCHEDULE_CLASS = None   # built once, on first need (nested def keeps mflux imports lazy)


def _scheduler_class():
    """The scheduler for the turbo schedule, built lazily so this module imports (and the app
    boots) even without mflux. Mirrors mflux's LinearScheduler — whose resolution-dependent
    exponential shift (base 0.5 @ seq 256 → max 0.9 @ seq 8192) is exactly the shift the base
    pipeline applies to its raw nodes — except the starting nodes are the turbo schedule, not
    uniform linspace, and there is no terminal stretch (shift_terminal = null, rule #2)."""
    global _SCHEDULE_CLASS
    if _SCHEDULE_CLASS is None:
        import mlx.core as mx
        from mflux.models.common.schedulers.base_scheduler import BaseScheduler

        class ViggleTurboScheduler(BaseScheduler):
            def __init__(self, config):
                self.config = config
                self._sigmas = self._get_sigmas()
                # the loop index doubles as the sigma index (qwen21's transformer resolves
                # the timestep from config.scheduler.sigmas[t]) — same contract as the
                # LinearScheduler the default generate path uses
                self._timesteps = mx.arange(config.num_inference_steps, dtype=mx.float32)

            @property
            def sigmas(self) -> "mx.array":
                return self._sigmas

            @property
            def timesteps(self) -> "mx.array":
                return self._timesteps

            def _get_sigmas(self) -> "mx.array":
                from mflux.models.common.config import ModelConfig
                mc = ModelConfig.qwen_image_21()
                nodes = mx.array(list(_RAW_SIGMAS[self.config.num_inference_steps]), dtype=mx.float32)
                m = (mc.sigma_max_shift - mc.sigma_base_shift) / (mc.sigma_max_seq_len - mc.sigma_base_seq_len)
                b = mc.sigma_base_shift - m * mc.sigma_base_seq_len
                mu = mx.array(m * self.config.width * self.config.height / 256 + b)
                shifted = mx.exp(mu) / (mx.exp(mu) + (1 / nodes - 1))
                return mx.concatenate([shifted, mx.zeros(1)])

            def step(self, noise, timestep: int, latents, **kwargs):
                dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
                return latents + noise.astype(latents.dtype) * dt

        _SCHEDULE_CLASS = ViggleTurboScheduler
    return _SCHEDULE_CLASS


def _register_scheduler() -> None:
    """Make the turbo schedule resolvable by Config via generate_image(scheduler=...).
    Best-effort and idempotent — a scheduler registration failure must never break generation."""
    try:
        from mflux.models.common.schedulers import SCHEDULER_REGISTRY
        if _SCHEDULE_NAME not in SCHEDULER_REGISTRY:
            SCHEDULER_REGISTRY[_SCHEDULE_NAME] = _scheduler_class()
    except Exception:
        pass


def _cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "alis-studio", "viggle-qwen21-turbo")


def _download_adapter(rank: str, progress) -> str:
    """Fetch the rank's adapter into our cache with the shared resumable HTTP bridge."""
    from studio.download import download_files
    fname = ADAPTER_FILES[rank]
    url = f"https://huggingface.co/{VIGGLE_REPO}/resolve/main/{fname}"
    dest = os.path.join(_cache_dir(), fname)
    if not os.path.exists(dest):
        os.makedirs(_cache_dir(), exist_ok=True)
        # the pre-flight size is authoritative; the table value only seeds the bar
        download_files([(url, dest)], progress or (lambda done, total: None))
    return dest


class ViggleQwen21TurboBackend(Backend):
    id = "qwen21-viggle-turbo"
    label = "Viggle Turbo (Qwen-Image-2.1)"
    supports_preview = True   # in-loop latents decode via Qwen21LatentCreator → live preview works
    min_ram_gib = 48          # measured: ~47 GB peak at 1024² (mflux: "a 64 GB machine is the comfortable default")
    prompt_note = "Understands Korean and other languages natively (Qwen3-VL encoder)."
    info = ("Qwen RESEARCH License (non-commercial) · 6-step DMD-distilled turbo of Qwen-Image-2.1 · "
            "~33 GB base + ~1.3 GB adapter on first use · ~47 GB peak — comfortable on 64 GB")
    variants = [
        {"id": "8bit", "label": "8-bit · best quality", "min_ram": 48},
        {"id": "4bit", "label": "4-bit · lighter transformer", "min_ram": 48},
    ]
    params = [
        {"key": "resolution", "label": "Resolution", "type": "resolution", "group": "Output",
         "sizes": [512, 768, 1024, 1536, 2048], "default_size": 1024,
         "aspects": ["1:1", "3:2", "2:3", "16:9", "9:16", "4:3", "3:4"], "default_aspect": "1:1",
         "min": 256, "max": 2048, "multiple": 32},
        {"key": "steps", "label": "Steps", "type": "int", "group": "Output", "min": 4, "max": 8, "default": 6,
         "hint": "Viggle's fixed sigma schedule — 6 is the shipped one; 8 prints dense small text cleaner."},
        {"key": "num_images", "label": "Images", "type": "int", "group": "Output", "min": 1, "max": 4, "default": 1},
        {"key": "seed", "label": "Seed", "type": "seed", "group": "Sampling", "default": 0},
        {"key": "guidance", "label": "Guidance (CFG)", "type": "float", "group": "Sampling",
         "min": 1, "max": 1, "step": 0.5, "default": 1.0, "fixed": True,
         "hint": "The turbo student samples without classifier-free guidance — leave at 1.0."},
        {"key": "negative", "label": "Negative prompt", "type": "text", "group": "Advanced",
         "default": "", "enabled": False,
         "hint": "Sampling runs without CFG (guidance 1.0), so a negative prompt has no effect."},
        *_img2img_params(),
        *_lora_params(),
    ]
    # Listed per build so the model manager can download/delete the adapter inline. The base
    # model (~33 GB) is mflux-managed: it auto-downloads (and quantizes in-place for 4-bit) on
    # the first Generate. The text encoder is never quantized; the adapter file is shared by
    # both builds — deleting one build leaves it for the other.
    catalog = [
        {"variant": "8bit", "label": "8-bit · best quality", "size_gb": ADAPTER_SIZE_GB[SHIPPED_RANK],
         "note": "adapter · ~33 GB base fetched by mflux on first use · ~47 GB peak"},
        {"variant": "4bit", "label": "4-bit · lighter transformer", "size_gb": ADAPTER_SIZE_GB[SHIPPED_RANK],
         "note": "adapter · ~33 GB base fetched by mflux (then quantized to 4-bit) · ~47 GB peak"},
    ]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mflux.models.qwen21.variants.txt2img.qwen_image_21  # noqa: F401
            from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping  # noqa: F401
            _register_scheduler()
            return True
        except Exception:
            return False

    def __init__(self):
        self._model = None
        self._key = None     # (variant, adapter path, extra-LoRA sig) — the model-cache key

    @staticmethod
    def _parts(variant: str) -> tuple[str, str]:
        if variant not in ("8bit", "4bit"):
            raise ValueError(f"Unknown build '{variant}' for Viggle Turbo (Qwen-Image-2.1)")
        return variant, SHIPPED_RANK

    def will_load(self, variant: str) -> bool:
        return self._model is None or self._key is None or self._key[0] != variant

    def _get(self, variant, params=None):
        import gc
        import copy as _copy
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        quant, rank = self._parts(variant)
        adapter = _download_adapter(rank, None)   # no-op when cached; self-heals on first use
        extra_paths, extra_scales = _lora_args(params or {})
        key = (variant, adapter, _lora_sig(params or {}))
        if self._model is None or self._key != key:
            self._model, self._key = None, None
            gc.collect()
            mx.clear_cache()
            # rule #2: the turbo student needs shift_terminal = null — the base config's 0.02
            # terminal stretch would wreck the last step. The classmethod is lru-cached and
            # shared, so patch a throwaway copy instead of the global instance.
            model_config = _copy.copy(ModelConfig.qwen_image_21())
            model_config.sigma_shift_terminal = None
            lora_paths = [adapter]
            lora_scales = [1.0]   # rule #3: alpha == rank, so scale stays exactly 1.0
            if extra_paths:
                lora_paths += extra_paths
                lora_scales += extra_scales
            self._model = _construct_checking_lora(
                lambda: QwenImage21(quantize=8 if quant == "8bit" else 4,
                                    model_config=model_config,
                                    lora_paths=lora_paths, lora_scales=lora_scales,
                                    bake_lora=False),   # rule #3: apply unmerged, exactly as diffusers
                lora_paths)
            self._key = key
        return self._model

    def generate(self, *, prompt, variant, params, step_callback):
        _register_scheduler()
        steps = int(params.get("steps", 6) or 6)
        if steps not in _RAW_SIGMAS:
            raise ValueError(f"Viggle Turbo samples with its distilled sigma schedule — steps must be "
                             f"one of {sorted(_RAW_SIGMAS)} (6 is the shipped schedule), not {steps}.")
        model = self._get(variant, params)
        w, h = int(params.get("width", 1024)), int(params.get("height", 1024))
        _apply_memory_policy(model, w, h)
        img_path, strength = _img2img_args(params)
        n = int(params.get("num_images", 1))
        out = []
        for i in range(n):
            _wire_progress(model, step_callback, base=i, batches=n)
            img = model.generate_image(
                seed=int(params.get("seed", 0)) + i, prompt=prompt,
                num_inference_steps=steps,
                height=h, width=w,
                guidance=1.0,                 # no CFG — hardcoded, the negative pass never runs
                scheduler=_SCHEDULE_NAME,
                image_path=img_path, image_strength=strength,
            )
            out.append(img.image)
        return out

    # --- model management (the adapter file; the base is mflux-managed) ------------------------
    def is_installed(self, variant: str) -> bool:
        try:
            _quant, rank = self._parts(variant)
            return os.path.exists(os.path.join(_cache_dir(), ADAPTER_FILES[rank]))
        except ValueError:
            return False

    def download(self, variant: str, progress) -> None:
        _quant, rank = self._parts(variant)
        _download_adapter(rank, progress)

    def delete(self, variant: str) -> None:
        _quant, rank = self._parts(variant)
        path = os.path.join(_cache_dir(), ADAPTER_FILES[rank])
        if os.path.exists(path):
            os.remove(path)
        if self._key and self._key[0] == variant:   # drop the loaded build if we just deleted it
            self._model, self._key = None, None