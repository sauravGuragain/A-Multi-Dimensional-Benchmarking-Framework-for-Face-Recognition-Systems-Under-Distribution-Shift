"""Exact AUC / EER in compute_verification_metrics: perfect separation, ties, endpoints, repeated scores."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

import faceeval.evaluation.metrics as M


def _metrics(scores, labels, **kw):
    return M.compute_verification_metrics([float(x) for x in scores], [bool(x) for x in labels], **kw)


def _ref_eer(scores, labels):
    """Independent reference: sklearn's full ROC, crossing of FPR and 1 - TPR, linear between vertices."""
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    diff = fpr - (1.0 - tpr)
    k = int(np.argmax(diff >= 0))
    if diff[k] == 0:
        return float(fpr[k])
    t = -diff[k - 1] / (diff[k] - diff[k - 1])
    return float(fpr[k - 1] + t * (fpr[k] - fpr[k - 1]))


# ---------------- the cases requested ----------------

def test_perfect_separation():
    y = np.r_[np.ones(30), np.zeros(30)].astype(bool)
    s = np.r_[np.linspace(0.6, 1.0, 30), np.linspace(0.0, 0.4, 30)]
    m = _metrics(s, y)
    assert m["auc"] == pytest.approx(1.0, abs=1e-12) and m["eer"] == pytest.approx(0.0, abs=1e-12)
    thr = m["eer_threshold"]
    assert (s[y] >= thr).all() and not (s[~y] >= thr).any()      # the reported threshold really separates


def test_perfectly_inverted_scores():
    y = np.r_[np.ones(30), np.zeros(30)].astype(bool)
    s = np.r_[np.linspace(0.0, 0.4, 30), np.linspace(0.6, 1.0, 30)]
    m = _metrics(s, y)
    assert m["auc"] == pytest.approx(0.0, abs=1e-12) and m["eer"] == pytest.approx(1.0, abs=1e-12)


def test_complete_ties():
    y = [True] * 20 + [False] * 20
    m = _metrics([0.5] * 40, y)
    assert m["auc"] == pytest.approx(0.5) and m["eer"] == pytest.approx(0.5)
    assert np.isfinite(m["eer_threshold"])


def test_crossing_between_the_final_two_vertices():
    # vertices strictest -> laxest: (FAR, FRR) = (0, 1), (0.2, 0.5), (1, 0); FAR - FRR = -1, -0.3, +1
    eer, thr = M._compute_eer([0.0, 0.2, 1.0], [1.0, 0.5, 0.0], [1.1, 1.0, 0.0])
    s = 0.3 / 1.3
    assert eer == pytest.approx(0.2 + s * 0.8) and eer == pytest.approx(0.5 - s * 0.5)   # on both lines
    assert thr == pytest.approx(1.0 + s * (0.0 - 1.0))


def test_crossing_between_the_first_two_vertices_when_top_scores_saturate():
    # 5 pos / 5 neg; at the maximum score 1.0 sit 4 positives and 2 negatives, so even the strictest real
    # threshold has FAR 0.4 > FRR 0.2 and the crossing lies against the "accept nothing" vertex.
    # Chord (FAR, FRR): (0, 1) -> (0.4, 0.2) crosses at FAR = FRR = 1/3 (the old endpoint shortcut gave 0.3).
    y = [True] * 4 + [False] * 2 + [True] + [False] * 3
    s = [1.0] * 4 + [1.0] * 2 + [0.5] + [0.1, 0.2, 0.3]
    m = _metrics(s, y)
    assert m["eer"] == pytest.approx(1.0 / 3.0)
    assert m["auc"] == pytest.approx(roc_auc_score(y, s))


def test_endpoint_vertex_exactly_on_the_diagonal():
    # one vertex has FAR == FRR exactly (both 0.5): returned as-is, no interpolation
    y = [True, True, False, False]
    s = [0.9, 0.2, 0.8, 0.1]          # vertices (0,1) (0,.5) (.5,.5) (.5,0) (1,0)
    m = _metrics(s, y)
    assert m["eer"] == pytest.approx(0.5) and m["eer"] == pytest.approx(_ref_eer(s, y))


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_repeated_scores_match_reference(decimals):
    rng = np.random.default_rng(7 + decimals)
    y = rng.random(400) < 0.5
    s = np.round(rng.normal(0.3, 0.2, 400) + 0.3 * y, decimals)
    assert len(np.unique(s)) < 400                                   # genuinely repeated
    m = _metrics(s, y)
    assert m["auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert m["eer"] == pytest.approx(_ref_eer(s, y), abs=1e-12)


# ---------------- properties ----------------

def test_exact_values_are_independent_of_the_threshold_grid():
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(764), np.zeros(764)].astype(bool)
    s = rng.normal(0.0, 0.05, 1528) + 0.06 * y
    s[0] = -1.0                                                      # an outlier that stretches a linspace grid
    a = _metrics(s, y, n_thresholds=500)
    b = _metrics(s, y, n_thresholds=7)
    assert a["auc"] == b["auc"] and a["eer"] == b["eer"]
    assert a["auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert a["eer"] == pytest.approx(_ref_eer(s, y), abs=1e-12)


def test_order_invariance_and_bounds():
    rng = np.random.default_rng(11)
    for _ in range(50):
        n = int(rng.integers(6, 300))
        y = rng.random(n) < 0.5
        if y.all() or not y.any():
            continue
        s = np.round(rng.normal(0, 1, n) + rng.uniform(0, 2) * y, int(rng.integers(0, 3)))
        m = _metrics(s, y)
        p = rng.permutation(n)
        m2 = _metrics(s[p], y[p])
        assert m["auc"] == m2["auc"] and m["eer"] == m2["eer"]
        assert 0.0 <= m["eer"] <= 1.0 and 0.0 <= m["auc"] <= 1.0
        assert m["auc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
        assert m["eer"] == pytest.approx(_ref_eer(s, y), abs=1e-12)


def test_roc_curve_output_is_unchanged_in_shape():
    rng = np.random.default_rng(5)
    y = rng.random(300) < 0.5
    s = rng.normal(0.3, 0.2, 300) + 0.3 * y
    roc = _metrics(s, y)["roc_curve"]
    assert (roc.fpr[0], roc.tpr[0]) == (0.0, 0.0) and (roc.fpr[-1], roc.tpr[-1]) == (1.0, 1.0)
    assert len(roc.fpr) == len(roc.tpr) == len(roc.thresholds) == 501      # 500-point grid + origin anchor
