"""Backend registry — the list of image-generation models Alis Studio can run.

Add a model by importing its Backend subclass and appending it to BACKENDS. Backends whose
dependencies aren't installed are skipped automatically, so the app runs with whatever you have.
"""

from __future__ import annotations

from .backends.cyber_z import CyberRealisticZBackend
from .backends.ernie import ErnieImageTurboBackend
from .backends.flux2 import Flux2KleinBackend
from .backends.krea2 import Krea2Backend
from .backends.mflux_models import FluxDevBackend, FluxSchnellBackend, QwenImageBackend
from .backends.qwen21_viggle import ViggleQwen21TurboBackend
from .backends.qwen_edit import QwenImageEditBackend
from .backends.z_image import ZImageTurboBackend

# register additional models here — a backend whose deps aren't importable is skipped automatically
BACKENDS = [Krea2Backend, ZImageTurboBackend, CyberRealisticZBackend, Flux2KleinBackend, QwenImageBackend,
            ErnieImageTurboBackend, QwenImageEditBackend, ViggleQwen21TurboBackend, FluxSchnellBackend, FluxDevBackend]


class Registry:
    def __init__(self):
        self.backends = {}
        for cls in BACKENDS:
            if cls.is_available():
                self.backends[cls.id] = cls()

    def models(self) -> list[dict]:
        return [b.meta() for b in self.backends.values()]

    def resolve(self, value: str):
        """'backend_id:variant_id' -> (backend, variant_id)."""
        bid, _, vid = value.partition(":")
        backend = self.backends.get(bid)
        if backend is None:
            raise ValueError(f"unknown model '{value}'")
        return backend, (vid or (backend.variants[0]["id"] if backend.variants else ""))
