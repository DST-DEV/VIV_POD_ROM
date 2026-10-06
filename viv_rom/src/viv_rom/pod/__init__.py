"""Proper orthogonal decomposition (POD) of the sectional-force field.

Only light, numpy-only names are imported eagerly; the pipeline (and with it
matplotlib) is loaded on first access of ``analyse``, ``run`` or
``write_figures``.
"""
from viv_rom.pod.config import PODConfig
from viv_rom.pod.core import (FieldInfo, PODResult, build_field_matrix,
                              compute_pod)
from viv_rom.pod.power import PowerCheck, power_correlation_check

__all__ = ["PODConfig", "FieldInfo", "PODResult", "build_field_matrix",
           "compute_pod", "PowerCheck", "power_correlation_check",
           "PODAnalysisResult", "analyse", "run", "write_figures"]

_LAZY = {"PODAnalysisResult", "analyse", "run", "write_figures"}


def __getattr__(name):
    if name in _LAZY:
        from viv_rom.pod import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
