"""Global proper orthogonal decomposition over a parameter sweep.

The snapshot matrices of several simulations (one per value of a wind speed,
pitch or inclination sweep) are concatenated in time and decomposed together,
giving one spatial basis for all of them. The pipeline (and with it
matplotlib) is loaded lazily on first access of ``analyse``, ``run``,
``leave_one_out`` or ``write_figures``.
"""
from viv_rom.gpod.analysis import (CaseProjection, GlobalPODResult,
                                   LeaveOneOutResult, LocalReference,
                                   SubspaceAngles, analyse_snapshots,
                                   leave_one_out_snapshots, project_case)
from viv_rom.gpod.config import GlobalPODConfig
from viv_rom.gpod.core import (GlobalBasis, fit_global_basis, pooled_svd,
                               principal_angles)
from viv_rom.gpod.snapshots import (CaseSnapshots, load_snapshots,
                                    prepare_case)

__all__ = ["GlobalPODConfig", "GlobalBasis", "fit_global_basis",
           "pooled_svd", "principal_angles", "CaseSnapshots",
           "load_snapshots", "prepare_case", "CaseProjection",
           "GlobalPODResult", "LeaveOneOutResult", "LocalReference",
           "SubspaceAngles", "analyse_snapshots", "leave_one_out_snapshots",
           "project_case", "analyse", "leave_one_out", "run",
           "write_figures"]

_LAZY = {"analyse", "leave_one_out", "run", "write_figures"}


def __getattr__(name):
    if name in _LAZY:
        from viv_rom.gpod import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
