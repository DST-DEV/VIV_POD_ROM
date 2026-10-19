"""Reduced-order modelling of vortex-induced vibration (VIV) CFD data.

Analysis methods live in subpackages (``viv_rom.pod``, ``viv_rom.spectral``).
Array convention throughout the package: sectional fields have shape
``(n_sections, n_steps)``."""
from viv_rom.io.results.export import load_result, save_result
from viv_rom.io.cfd.cfd_reader import HyReader
from viv_rom.pod.config import PODConfig
from viv_rom.spectral.config import SpectralConfig
from viv_rom import pod
from viv_rom import spectral

__all__ = ["HyReader", "PODConfig", "SpectralConfig", "pod", "spectral",
           "load_result", "save_result"]
