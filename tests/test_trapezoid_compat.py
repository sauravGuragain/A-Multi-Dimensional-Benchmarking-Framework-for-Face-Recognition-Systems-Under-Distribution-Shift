"""NumPy 1.x / 2.x trapezoid compatibility and ROC-AUC correctness (ties, anchoring)."""
import importlib

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import faceeval.evaluation.metrics as M


def _rng_case(seed=0, n=300, decimals=None):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.5
    s = rng.normal(0.3, 0.2, n) + 0.3 * y
    return (np.round(s, decimals) if decimals is not None else s), y


def test_trapezoid_matches_closed_form():
    # signature is trapezoid(y, x)
    assert M.trapezoid([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.5)            # triangle
    assert M.trapezoid([0.0, 1.0, 1.0], [0.0, 0.5, 1.0]) == pytest.approx(0.25 + 0.5)
    xs = np.linspace(0, 1, 11)
    assert M.trapezoid(xs ** 2, xs) == pytest.approx(0.335, abs=1e-12)   # exact trapezoid sum


def _reload_with(monkeypatch, *, has_trapezoid, has_trapz):
    """Reload metrics under a simulated NumPy API surface (1.x: trapz only; late 2.x: trapezoid only)."""
    real = M.trapezoid
    with monkeypatch.context() as mp:
        if has_trapezoid:
            mp.setattr(np, "trapezoid", real, raising=False)
        elif hasattr(np, "trapezoid"):
            mp.delattr(np, "trapezoid")
        if has_trapz:
            mp.setattr(np, "trapz", real, raising=False)
        elif hasattr(np, "trapz"):
            mp.delattr(np, "trapz")
        mod = importlib.reload(M)
        helper_is = mod.trapezoid
        auc = mod.compute_verification_metrics(*_as_lists(_rng_case()))["auc"]
    importlib.reload(M)
    return helper_is, auc, real


def _as_lists(case):
    s, y = case
    return s.tolist(), y.tolist()


def test_numpy1_api_surface_uses_trapz(monkeypatch):
    """NumPy 1.x: no np.trapezoid -> must fall back to np.trapz (this was the reported AttributeError)."""
    helper, auc, real = _reload_with(monkeypatch, has_trapezoid=False, has_trapz=True)
    assert helper is real and 0.5 < auc <= 1.0


def test_numpy2_api_surface_uses_trapezoid_without_touching_trapz(monkeypatch):
    helper, auc, real = _reload_with(monkeypatch, has_trapezoid=True, has_trapz=False)
    assert helper is real and 0.5 < auc <= 1.0


@pytest.mark.parametrize("decimals", [None, 1, 0])
def test_auc_matches_reference_including_ties(decimals):
    s, y = _rng_case(seed=3, decimals=decimals)
    auc = M.compute_verification_metrics(s.tolist(), y.tolist())["auc"]
    assert auc == pytest.approx(roc_auc_score(y, s), abs=2e-3)   # 500-threshold sweep resolution


def test_auc_fully_tied_scores_is_chance():
    y = [True] * 20 + [False] * 20
    assert M.compute_verification_metrics([0.5] * 40, y)["auc"] == pytest.approx(0.5)


def test_auc_perfect_and_inverted_and_order_invariant():
    y = np.r_[np.ones(30), np.zeros(30)].astype(bool)
    s = np.r_[np.linspace(0.6, 1, 30), np.linspace(0, 0.4, 30)]
    assert M.compute_verification_metrics(s.tolist(), y.tolist())["auc"] == pytest.approx(1.0)
    assert M.compute_verification_metrics((1 - s).tolist(), y.tolist())["auc"] == pytest.approx(0.0, abs=1e-12)
    perm = np.random.default_rng(1).permutation(60)
    a = M.compute_verification_metrics(s[perm].tolist(), y[perm].tolist())["auc"]
    assert a == pytest.approx(1.0)


def test_roc_curve_starts_at_origin_and_ends_at_one():
    s, y = _rng_case(seed=5)
    roc = M.compute_verification_metrics(s.tolist(), y.tolist())["roc_curve"]
    assert (roc.fpr[0], roc.tpr[0]) == (0.0, 0.0) and (roc.fpr[-1], roc.tpr[-1]) == (1.0, 1.0)
    assert len(roc.fpr) == len(roc.tpr) == len(roc.thresholds)
    assert all(b >= a for a, b in zip(roc.fpr, roc.fpr[1:]))


def test_sample_efficiency_uses_shared_helper():
    import faceeval.fingerprint.sample_efficiency as SE
    assert SE.trapezoid is M.trapezoid
