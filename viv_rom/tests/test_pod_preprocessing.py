import numpy as np
import pytest

from viv_rom.domain.motion import FIRST_EDGEWISE_MODE
from viv_rom.pod import (DynamicPressureNormalizer, NormalizationContext,
                         PODConfig, PhysicalReconstructor, Preprocessor,
                         WorkWeightNormalizer, build_field_matrix,
                         compute_pod, make_normalizer, register_normalizer,
                         remove_mean)
from viv_rom.pod.normalization import ComponentScales, available_normalizers

N, T = 30, 120


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = np.linspace(3.0, 85.0, N)
    t = np.linspace(0, 8, T)
    Fx = 5 + rng.normal(size=(N, 3)) @ rng.normal(size=(3, T))
    Fz = -2 + rng.normal(size=(N, 3)) @ rng.normal(size=(3, T))
    return y, t, Fx, Fz


@pytest.fixture
def ctx(data):
    return NormalizationContext(y=data[0], wind_speed=10.0,
                                motion=FIRST_EDGEWISE_MODE, amplitude=1.0,
                                omega=2 * np.pi * 0.624)


@pytest.mark.parametrize("mode", ["fx", "fz", "combined"])
@pytest.mark.parametrize("names", [(), ("dynamic_pressure",), ("work",),
                                   ("dynamic_pressure", "work")])
def test_roundtrip_and_full_rank_reconstruction(data, ctx, mode, names):
    _, _, Fx, Fz = data
    field, info = build_field_matrix(Fx, Fz, mode)
    pre = Preprocessor([make_normalizer(n) for n in names])
    fluct, prep = pre.fit_transform(field, info, ctx)
    assert np.allclose(fluct.mean(axis=1), 0, atol=1e-9 * np.abs(fluct).max())
    assert np.allclose(prep.inverse(fluct), field)
    assert np.allclose(prep.forward(field), fluct)
    pod = compute_pod(fluct)
    rec = PhysicalReconstructor(pod, prep)
    assert np.allclose(rec(pod.n_modes), field)


def test_no_normalization_matches_legacy_pod(data):
    _, _, Fx, Fz = data
    field, info = build_field_matrix(Fx, Fz, "combined")
    legacy_fluct = field - field.mean(axis=1, keepdims=True)
    fluct, mean = remove_mean(field)
    assert np.array_equal(fluct, legacy_fluct)
    assert np.allclose(compute_pod(fluct).sigma,
                       np.linalg.svd(legacy_fluct, compute_uv=False))


def test_compute_pod_does_not_remove_mean():
    D = np.ones((4, 10)) * 3.0
    pod = compute_pod(D)
    assert pod.sigma[0] == pytest.approx(3.0 * np.sqrt(40))


def test_dynamic_pressure_factor(data, ctx):
    s = DynamicPressureNormalizer().scales(ctx)
    q = 0.5 * 1.225 * 10.0**2
    assert np.allclose(s.fx, 1 / q) and np.allclose(s.fz, 1 / q)
    with pytest.raises(ValueError):
        DynamicPressureNormalizer().scales(
            NormalizationContext(**{**ctx.__dict__, "wind_speed": 0.0}))


def test_dynamic_pressure_collapses_wind_speed(data, ctx):
    """F ~ U^2 data gives identical normalised fluctuation at all speeds."""
    _, _, Fx, Fz = data
    out = []
    for U in (6.0, 12.0):
        c = NormalizationContext(**{**ctx.__dict__, "wind_speed": U})
        field, info = build_field_matrix(Fx * U**2, Fz * U**2, "combined")
        fluct, _ = Preprocessor([DynamicPressureNormalizer()]).fit_transform(
            field, info, c)
        out.append(fluct)
    assert np.allclose(out[0], out[1])


def test_work_weight_gives_power_amplitude(data, ctx):
    y = data[0]
    s = WorkWeightNormalizer(floor=0).scales(ctx)
    A, w = ctx.amplitude, ctx.omega
    assert np.allclose(s.fx, FIRST_EDGEWISE_MODE.modeshape(
        "Fx", y, normalize=False) * A * w)
    assert np.allclose(s.fz, FIRST_EDGEWISE_MODE.modeshape(
        "Fz", y, normalize=False) * A * w)


def test_work_weight_floor_keeps_sign_and_bounds(ctx):
    s = WorkWeightNormalizer(floor=0.1).scales(ctx)
    for v in (s.fx, s.fz):
        assert np.all(v != 0)
    big = max(np.abs(s.fx).max(), np.abs(s.fz).max())
    assert min(np.abs(s.fx).min(), np.abs(s.fz).min()) >= 0.1 * big - 1e-15


def test_work_weighted_sum_is_power(data, ctx):
    """Sum of weighted Fx+Fz equals sectional power at cos(omega t)=1."""
    y, t, Fx, Fz = data
    field, info = build_field_matrix(Fx, Fz, "combined")
    fluct, prep = Preprocessor([WorkWeightNormalizer(floor=0)]
                               ).fit_transform(field, info, ctx)
    weighted = prep.scaling.apply(field)
    wx, wz = info.split(weighted)
    vx = FIRST_EDGEWISE_MODE.velocity("x", y, t, 1.0, ctx.omega)
    vz = FIRST_EDGEWISE_MODE.velocity("z", y, t, 1.0, ctx.omega)
    p = Fx * vx + Fz * vz
    assert np.allclose((wx + wz) * np.cos(ctx.omega * t), p)


def test_chaining_is_product(ctx):
    a, b = DynamicPressureNormalizer(), WorkWeightNormalizer()
    from viv_rom.pod.normalization import FieldScaling
    from viv_rom.pod.core import FieldInfo
    info = FieldInfo(kind="combined", n=N)
    f_ab = FieldScaling.from_normalizers([a, b], ctx, info).factors
    f_a = FieldScaling.from_normalizers([a], ctx, info).factors
    f_b = FieldScaling.from_normalizers([b], ctx, info).factors
    assert np.allclose(f_ab, f_a * f_b)
    assert FieldScaling.from_normalizers([], ctx, info).is_identity


def test_config_accepts_names_instances_and_rejects_unknown():
    assert PODConfig().normalizers() == ()
    cfg = PODConfig(normalization="work")
    assert cfg.normalization == ("work",)
    inst = WorkWeightNormalizer(floor=1e-2)
    cfg = PODConfig(normalization=("dynamic_pressure", inst))
    assert cfg.normalizers()[1] is inst
    with pytest.raises(ValueError):
        PODConfig(normalization=("nope",))


def test_custom_strategy_is_pluggable(ctx):
    @register_normalizer("test_double")
    class Double:
        name = "test_double"

        def scales(self, c):
            return ComponentScales.uniform(2.0, len(c.y))

    assert "test_double" in available_normalizers()
    from viv_rom.pod.core import FieldInfo
    from viv_rom.pod.normalization import FieldScaling
    sc = FieldScaling.from_normalizers([make_normalizer("test_double")], ctx,
                                       FieldInfo(kind="combined", n=N))
    assert np.allclose(sc.factors, 2.0)
    from viv_rom.pod import normalization
    normalization._REGISTRY.pop("test_double")
