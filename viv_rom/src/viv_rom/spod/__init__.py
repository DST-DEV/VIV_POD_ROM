"""Spectral proper orthogonal decomposition (SPOD) of the sectional forces.

Numerics are imported eagerly; the pipeline (and with it matplotlib) is
loaded on first access of ``analyse``, ``run`` or ``write_figures``.
"""
from viv_rom.spod.analysis import (HarmonicContent, SpectralPeaks,
                                   fundamental_mode_lines, harmonic_content,
                                   spectral_peaks)
from viv_rom.spod.blocks import (FourierBlocks, SegmentSpec, fourier_blocks)
from viv_rom.spod.config import SPODConfig
from viv_rom.spod.core import SPODResult, compute_spod
from viv_rom.spod.injection import SPODInjection, spod_injection
from viv_rom.spod.locking import PhaseLockedResponse, phase_locked_response
from viv_rom.spod.weights import (InnerProduct, available_inner_products,
                                  make_inner_product, register_inner_product)

__all__ = ["SPODConfig", "SegmentSpec", "FourierBlocks", "fourier_blocks",
           "SPODResult", "compute_spod", "InnerProduct",
           "available_inner_products", "make_inner_product",
           "register_inner_product", "HarmonicContent", "SpectralPeaks",
           "harmonic_content", "spectral_peaks", "fundamental_mode_lines",
           "PhaseLockedResponse", "phase_locked_response", "SPODInjection",
           "spod_injection",
           "SPODAnalysisResult", "analyse", "run", "write_figures"]

_LAZY = {"SPODAnalysisResult", "analyse", "run", "write_figures"}


def __getattr__(name):
    if name in _LAZY:
        from viv_rom.spod import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
