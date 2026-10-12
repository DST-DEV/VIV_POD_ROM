"""Spectral analysis (PSD, phase, spectrogram, lock-in sweep).

Numerics are imported eagerly; the pipeline (and with it matplotlib) is
loaded on first access of ``analyse``, ``run`` or ``write_figures``.
"""
from viv_rom.spectral.config import SpectralConfig
from viv_rom.spectral.core import (compute_psd, compute_spanwise_amplitude,
                                   compute_spanwise_phase,
                                   compute_spanwise_psd, compute_spectrogram,
                                   dominant_peaks,
                                   integrate_over_span)
from viv_rom.spectral.cycle_energy import (CycleEnergy, CycleStatistics,
                                           cycle_energy_analysis,
                                           cycle_statistics,
                                           per_cycle_integral)
from viv_rom.spectral.forcing_response import (ForcingResponse,
                                               forcing_frequency_response)
from viv_rom.spectral.sweep import SweepResult, frequency_sweep

__all__ = ["SpectralConfig", "compute_psd", "compute_spanwise_psd",
           "compute_spanwise_phase", "compute_spanwise_amplitude",
           "compute_spectrogram",
           "dominant_peaks",
           "integrate_over_span",
           "SweepResult", "frequency_sweep",
           "ForcingResponse", "forcing_frequency_response",
           "CycleEnergy", "CycleStatistics", "cycle_energy_analysis",
           "cycle_statistics", "per_cycle_integral",
           "SpectralAnalysisResult", "analyse", "run", "write_figures"]

_LAZY = {"SpectralAnalysisResult", "analyse", "run", "write_figures"}


def __getattr__(name):
    if name in _LAZY:
        from viv_rom.spectral import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
