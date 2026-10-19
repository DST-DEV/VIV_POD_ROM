"""Saving and loading analysis results (``SpectralAnalysisResult``,
``PODAnalysisResult``, ``SPODAnalysisResult``) together with their config.

File format
-----------
One compressed NumPy archive (``.npz``) per analysis:

* every array of the result is stored natively (any dtype, including
  complex), so nothing is lost to text conversion;
* the structure around the arrays -- the nested dataclasses, tuples,
  dicts, paths, scalars -- is a JSON document stored in the archive entry
  ``__meta__``. It names the classes, so :func:`load_result` rebuilds the
  *original* dataclasses (``SpectralAnalysisResult`` + ``SpectralConfig``
  etc.), not dictionaries.

No ``pickle`` is involved: the archive is opened with
``allow_pickle=False`` and only classes from the ``viv_rom`` package are
ever instantiated, so loading a file cannot run foreign code. The price is
that results holding objects from outside the package (a user-defined
``Normalizer`` or force frame, a ``lambda`` as ``energy_error_reference``)
cannot be exported; :func:`save_result` says so instead of writing a file
that would not load.

Usage::

    from viv_rom.export import save_result, load_result

    save_result(result, config, "results/pod_results.npz")
    result, config = load_result("results/pod_results.npz")

Raw force fields
----------------
The result keeps the CFD force field it was computed from (``Fx``, ``Fz``
and, for the POD, the snapshot matrix ``field``), which dominates the file
size. ``include_raw=False`` drops these; ``t`` and ``y`` are kept. In the
loaded result ``force_file.data.Fx``/``Fz`` and ``field`` are then ``None``.
Everything that was *computed* (modes, spectra, energies, ...) is kept, but
methods that need the raw field (e.g. ``SpectralAnalysisResult.components``)
cannot be used on such a result.
"""
from __future__ import annotations

import dataclasses
import datetime
import importlib
import inspect
import json
import os
from pathlib import Path, PurePath
from typing import Any, NamedTuple

import numpy as np

FORMAT = "viv_rom.analysis"
FORMAT_VERSION = 1
_META_KEY = "__meta__"
_PACKAGE = "viv_rom"


class LoadedAnalysis(NamedTuple):
    """What :func:`load_result` returns (unpacks as ``result, config``)."""
    result: Any
    config: Any


# =============================================================================
# Omitting the raw force field
# =============================================================================
def omit_raw_data(result):
    """Copy of ``result`` without the raw force field.

    Sets ``force_file.data.Fx``/``Fz`` and, if present, the POD snapshot
    matrix ``field`` to ``None``. The time and span vectors are kept. The
    input is not modified.
    """
    force_file = result.force_file
    data = dataclasses.replace(force_file.data, Fx=None, Fz=None)
    changes = {"force_file": dataclasses.replace(force_file, data=data)}
    if any(f.name == "field" for f in dataclasses.fields(result)):
        changes["field"] = None
    return dataclasses.replace(result, **changes)


# =============================================================================
# Public API
# =============================================================================
def save_result(result, config, path: str | Path, *,
                include_raw: bool = True) -> Path:
    """Write ``result`` and its ``config`` to one ``.npz`` file.

    Parameters
    ----------
    result : SpectralAnalysisResult, PODAnalysisResult or SPODAnalysisResult
    config : SpectralConfig, PODConfig or SPODConfig
        The configuration the result was computed with.
    path : str or Path
        Target file (parent folders are created). Use the ``.npz`` suffix.
    include_raw : bool
        If False, leave out the raw force field (see the module docs); the
        file is then a fraction of the size.

    Returns
    -------
    Path
        The written file.

    Raises
    ------
    TypeError
        If result or config holds something that cannot be exported (the
        message names where).
    """
    path = Path(path)
    if not include_raw:
        result = omit_raw_data(result)

    encoder = _Encoder()
    tree = {"result": encoder.encode(result, "result"),
            "config": encoder.encode(config, "config")}
    meta = {
        "format": FORMAT, "version": FORMAT_VERSION,
        "result_type": type(result).__name__,
        "config_type": type(config).__name__,
        "raw_included": bool(include_raw),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "tree": tree,
    }
    payload = {_META_KEY: np.frombuffer(json.dumps(meta).encode("utf-8"),
                                        dtype=np.uint8),
               **encoder.arrays}

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")     # never leave a half file
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load_result(path: str | Path) -> LoadedAnalysis:
    """Read a file written by :func:`save_result`.

    Returns
    -------
    LoadedAnalysis
        ``(result, config)`` as the original dataclasses. If the file was
        saved with ``include_raw=False`` the raw fields are ``None``.

    Raises
    ------
    ValueError
        If the file is not a viv_rom export, has a newer format version,
        or refers to classes outside the ``viv_rom`` package.
    """
    with np.load(Path(path), allow_pickle=False) as archive:
        if _META_KEY not in archive.files:
            raise ValueError(f"{path} is not a viv_rom export "
                             f"(no {_META_KEY!r} entry).")
        meta = json.loads(archive[_META_KEY].tobytes().decode("utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a viv_rom export.")
        if meta.get("version", 0) > FORMAT_VERSION:
            raise ValueError(
                f"{path} uses export format version {meta['version']}; "
                f"this viv_rom reads up to version {FORMAT_VERSION}.")
        decoder = _Decoder(archive)
        return LoadedAnalysis(
            result=decoder.decode(meta["tree"]["result"]),
            config=decoder.decode(meta["tree"]["config"]))


# =============================================================================
# Encoding
# =============================================================================
def _importable(module: str, qualname: str) -> bool:
    return module == _PACKAGE or module.startswith(_PACKAGE + ".")


def _reference(obj, where: str) -> str:
    """``"module:Qualname"`` of a class or function, checked to be loadable.
    """
    module = getattr(obj, "__module__", None) or ""
    qualname = getattr(obj, "__qualname__", "")
    if not _importable(module, qualname):
        raise TypeError(
            f"Cannot export {where}: {module or '?'}.{qualname} is not part "
            f"of the {_PACKAGE} package (only its own classes and functions "
            "can be stored without pickle).")
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise TypeError(f"Cannot export {where}: {qualname} is a local or "
                        "anonymous function.")
    try:
        found = _resolve(f"{module}:{qualname}")
    except (ImportError, AttributeError):
        found = None
    if found is not obj:
        raise TypeError(f"Cannot export {where}: {module}.{qualname} cannot "
                        "be found again by that name.")
    return f"{module}:{qualname}"


class _Encoder:
    """Turns a nested structure into JSON-able data plus a dict of arrays."""

    def __init__(self):
        self.arrays: dict[str, np.ndarray] = {}
        self._ids: dict[int, str] = {}
        self._keep: list[np.ndarray] = []     # keep ids valid while encoding

    def _array(self, arr: np.ndarray, where: str) -> str:
        if arr.dtype.hasobject:
            raise TypeError(f"Cannot export {where}: object arrays are not "
                            "supported.")
        key = self._ids.get(id(arr))
        if key is None:                       # shared arrays are stored once
            key = f"a{len(self.arrays):05d}"
            self.arrays[key] = arr
            self._ids[id(arr)] = key
            self._keep.append(arr)
        return key

    def encode(self, obj, where: str):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": self._array(obj, where)}
        if isinstance(obj, np.generic):  # before float/int/str: np.float64
            return {"__npscalar__": self._array(np.asarray(obj), where)}
        if obj is None or type(obj) in (bool, int, float, str):
            return obj
        if isinstance(obj, complex):
            return {"__complex__": [obj.real, obj.imag]}
        if isinstance(obj, PurePath):
            return {"__path__": str(obj)}
        if isinstance(obj, tuple):
            return {"__tuple__": [self.encode(v, f"{where}[{i}]")
                                  for i, v in enumerate(obj)]}
        if isinstance(obj, list):
            return [self.encode(v, f"{where}[{i}]")
                    for i, v in enumerate(obj)]
        if isinstance(obj, dict):
            if all(type(k) is str and not k.startswith("__") for k in obj):
                return {k: self.encode(v, f"{where}[{k!r}]")
                        for k, v in obj.items()}
            return {"__dict__": [[self.encode(k, f"{where} key"),
                                  self.encode(v, f"{where}[{k!r}]")]
                                 for k, v in obj.items()]}
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            ref = _reference(type(obj), where)
            return {"__dataclass__": ref,
                    "fields": {f.name: self.encode(getattr(obj, f.name),
                                                   f"{where}.{f.name}")
                               for f in dataclasses.fields(obj)}}
        if inspect.isfunction(obj) or inspect.isclass(obj):
            return {"__callable__": _reference(obj, where)}
        if hasattr(obj, "__dict__"):  # e.g. the ForceFrame classes
            ref = _reference(type(obj), where)
            return {"__object__": ref,
                    "state": {k: self.encode(v, f"{where}.{k}")
                              for k, v in vars(obj).items()}}
        raise TypeError(f"Cannot export {where}: unsupported type "
                        f"{type(obj).__module__}.{type(obj).__qualname__}.")


# =============================================================================
# Decoding
# =============================================================================
def _resolve(reference: str):
    """Import ``"module:Qualname"``; only the viv_rom package is allowed."""
    module, _, qualname = reference.partition(":")
    if not _importable(module, qualname):
        raise ValueError(f"Refusing to load {reference!r}: only classes "
                         f"from the {_PACKAGE} package are allowed.")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


class _Decoder:
    def __init__(self, archive):
        self._archive = archive
        self._cache: dict[str, np.ndarray] = {}

    def _array(self, key: str) -> np.ndarray:
        if key not in self._cache:            # keeps shared arrays shared
            self._cache[key] = self._archive[key]
        return self._cache[key]

    def decode(self, node):
        if isinstance(node, list):
            return [self.decode(v) for v in node]
        if not isinstance(node, dict):
            return node
        if "__ndarray__" in node:
            return self._array(node["__ndarray__"])
        if "__npscalar__" in node:
            return self._array(node["__npscalar__"])[()]
        if "__complex__" in node:
            return complex(*node["__complex__"])
        if "__path__" in node:
            return Path(node["__path__"])
        if "__tuple__" in node:
            return tuple(self.decode(v) for v in node["__tuple__"])
        if "__dict__" in node:
            return {self.decode(k): self.decode(v)
                    for k, v in node["__dict__"]}
        if "__dataclass__" in node:
            return self._dataclass(node)
        if "__callable__" in node:
            return _resolve(node["__callable__"])
        if "__object__" in node:
            cls = _resolve(node["__object__"])
            obj = cls.__new__(cls)
            obj.__dict__.update({k: self.decode(v)
                                 for k, v in node["state"].items()})
            return obj
        return {k: self.decode(v) for k, v in node.items()}

    def _dataclass(self, node):
        cls = _resolve(node["__dataclass__"])
        if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
            raise ValueError(f"{node['__dataclass__']!r} is not a dataclass.")
        values = {k: self.decode(v) for k, v in node["fields"].items()}
        init = {f.name for f in dataclasses.fields(cls) if f.init}
        obj = cls(**{k: v for k, v in values.items() if k in init})
        for name, value in values.items():     # fields excluded from __init__
            if name not in init:
                object.__setattr__(obj, name, value)
        return obj
