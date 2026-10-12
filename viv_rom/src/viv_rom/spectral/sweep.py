"""Wind-speed sweep: dominant response frequency vs. U (lock-in diagram)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from viv_rom.loading import load_case
from viv_rom.spectral.config import SpectralConfig
from viv_rom.spectral.core import (compute_psd,
                                   dominant_peaks,
                                   integrate_over_span)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class SweepResult:
    """Dominant frequency of the span-integrated force per wind speed.

    Attributes
    ----------
    wind_speeds : ndarray, shape (n_found,)
        Wind speeds [m/s] for which a file was loaded.
    f_dom : ndarray, shape (n_found,)
        Dominant frequency [Hz] of the total force at each wind speed.
    """
    wind_speeds: np.ndarray
    f_dom: np.ndarray


def frequency_sweep(config: SpectralConfig) -> SweepResult:
    """Dominant frequency of the span-integrated force for each wind speed.

    Loops over ``config.u_sweep_list`` (same case, trimming, frame and
    ``config.sweep_field``) and skips any wind speed whose file is missing.
    """
    frame = config.frame_instance()
    sweep_field = config.effective_sweep_field
    found: list[int] = []
    f_dom: list[float] = []

    for wind_speed in config.u_sweep_list:
        try:
            ff = load_case(config, wind_speed=wind_speed)
        except FileNotFoundError as err:
            # np.loadtxt raises FileNotFoundError without .filename set; the
            # path is then only available from the message.
            name = Path(err.filename or str(err).removesuffix(" not found.")
                        ).name
            logger.info("  [sweep] skipping U = %s m/s. file not found (%s)",
                        wind_speed, name)
            continue

        data = ff.data
        component = frame.transform(data.y, data.Fx, data.Fz)[sweep_field]
        total = integrate_over_span(data.y, component)
        dt = data.t[1] - data.t[0]
        freqs, psd = compute_psd(total, dt, config.nperseg)
        f_peak, _ = dominant_peaks(freqs, psd, n_peaks=1)

        found.append(wind_speed)
        f_dom.append(float(f_peak[0]))

    return SweepResult(np.array(found, dtype=float), np.array(f_dom))
