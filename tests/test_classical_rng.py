"""Classical-model random state: the historical protocol (global NumPy stream) is unchanged by default; the isolated-seed
protocol is opt-in, hashed, and independent of the global stream."""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

import faceeval.models  # noqa: F401
from faceeval.core.registry import model_registry
from faceeval.models.traditional.base import CLASSICAL_RNG_GLOBAL, CLASSICAL_RNG_ISOLATED, apply_classical_rng

CFG = lambda mode, seed=42: types.SimpleNamespace(classical_rng=mode, random_seed=seed)
CLASSICAL = ("eigenfaces", "fisherfaces", "pca_svm", "lbph", "hog_svm", "knn")


def test_default_protocol_leaves_models_untouched():
    for name in CLASSICAL:
        m = model_registry.get(name)
        assert apply_classical_rng(m, CFG(CLASSICAL_RNG_GLOBAL)) is False and m._random_state is None
        assert apply_classical_rng(m, types.SimpleNamespace(random_seed=1)) is False     # config without the field = historical


def test_isolated_protocol_sets_the_seed_on_every_classical_model_only():
    for name in CLASSICAL:
        m = model_registry.get(name)
        assert apply_classical_rng(m, CFG(CLASSICAL_RNG_ISOLATED, 7)) is True and m._random_state == 7
    assert apply_classical_rng(types.SimpleNamespace(PARADIGM="deep"), CFG(CLASSICAL_RNG_ISOLATED)) is False


def test_default_extractor_is_built_exactly_as_before():
    from sklearn.decomposition import PCA
    m = model_registry.get("eigenfaces")
    ex = m._build_extractor()
    assert ex.get_params()["random_state"] is None
    assert ex.get_params() == PCA(n_components=m._n_components, whiten=m._whiten).get_params()


def _data():
    return np.random.default_rng(0).normal(size=(700, 1500)).astype(np.float32)


def _fit_features(mode, consume_global):
    m = model_registry.get("eigenfaces")
    m._n_components = 40
    np.random.seed(42)
    if consume_global:
        np.random.random(1000)                      # earlier code used some of the global stream
    apply_classical_rng(m, CFG(mode))
    X = _data()
    m.fit_embedding([x for x in X.reshape(700, 30, 50)], [str(i) for i in range(700)])
    return np.asarray(m._extractor.transform(X)), m._extractor._fit_svd_solver


def test_isolated_result_does_not_depend_on_the_global_stream():
    a, solver = _fit_features(CLASSICAL_RNG_ISOLATED, False)
    b, _ = _fit_features(CLASSICAL_RNG_ISOLATED, True)
    assert np.array_equal(a, b)
    assert solver in ("randomized", "full", "covariance_eigh", "arpack")


def test_default_result_depends_on_the_global_stream_when_the_solver_is_randomized():
    a, solver = _fit_features(CLASSICAL_RNG_GLOBAL, False)
    b, _ = _fit_features(CLASSICAL_RNG_GLOBAL, True)
    if solver != "randomized":
        pytest.skip(f"this scikit-learn resolved the solver to {solver!r}: no randomness to expose")
    assert not np.array_equal(a, b)                 # the historical protocol's hidden dependency, documented not 'fixed'
    c, _ = _fit_features(CLASSICAL_RNG_GLOBAL, False)
    assert np.array_equal(a, c)                     # but it IS repeatable when the stream is in the same state


def test_config_field_default_is_not_serialised_and_isolated_changes_the_hash(tmp_path):
    import yaml
    from faceeval.core.config import load_config
    src = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "lfw_fold0_classical.yaml"))
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(src))
    base = load_config(str(p))
    assert base.classical_rng == "global_seed" and "classical_rng" not in base.to_dict()
    src["classical_rng"] = "isolated_seed"
    p.write_text(yaml.safe_dump(src))
    iso = load_config(str(p))
    assert iso.to_dict()["classical_rng"] == "isolated_seed" and iso.config_hash != base.config_hash
    src["classical_rng"] = "nonsense"
    p.write_text(yaml.safe_dump(src))
    with pytest.raises(Exception):
        load_config(str(p))
