"""All ten registered models run through the SAME verification runner, and genuine vs placeholder backbones are
reported explicitly.

SCOPE: on fake random-noise LFW with no pretrained weights installed. This proves the shared code path, the
failure semantics and the stub labelling for every model. It does NOT validate real insightface/dlib weights or any accuracy.
"""
import csv
import glob
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_verification_stage as T  # noqa: E402
from faceeval.core.config import ConfigLoader  # noqa: E402
from faceeval.orchestration.runner import ExperimentRunner  # noqa: E402

TRAD = ["eigenfaces", "fisherfaces", "pca_svm", "lbph", "hog_svm", "knn"]
DEEP = ["facenet", "arcface", "insightface", "dlib_fr"]
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def force_stub_backbones(mp):
    """Make the test independent of what is installed: no deep package is importable, so every deep model takes its
    documented stub path on any machine (including one that has real FaceNet/InsightFace/dlib weights)."""
    import faceeval.models.deep.arcface as af
    import faceeval.models.deep.facenet as fn
    mp.setattr(fn, "_FACENET_AVAILABLE", False)
    mp.setattr(af, "_INSIGHTFACE_AVAILABLE", False)
    for name in ("insightface", "dlib", "facenet_pytorch"):
        mp.setitem(sys.modules, name, None)


def _text(root, require_real, trad, deep):
    return f"""
experiment_name: ten_models
random_seed: 42
dataset_names: [lfw]
data_root: {root}
split_strategy: stratified
evaluation_protocol: lfw_fold0_verification
require_real_models: {str(require_real).lower()}
traditional_models: [{", ".join(trad)}]
deep_models: [{", ".join(deep)}]
perturbation_schedule: {{perturbation_types: [gaussian_blur], severity_levels: [0.0, 1.0], compose_perturbations: false}}
compute_calibration: false
compute_fairness: false
compute_fingerprints: false
compute_failure_analysis: false
compute_deployment_scores: false
deployment_scenarios: [research_benchmark]
profile_resources: false
use_feature_cache: false
feature_cache_dir: {root}/cache/
output_dir: {root}/out/
report_dir: {root}/rep/
export_png: false
mlflow_tracking_uri: sqlite:///{root}/mlflow.db
mlflow_experiment_name: ten
device: cpu
"""


@pytest.fixture(scope="module")
def ten(tmp_path_factory):
    root = tmp_path_factory.mktemp("ten")
    T._make_fake_lfw(root)
    old = os.getcwd()
    os.chdir(root)
    try:
        with pytest.MonkeyPatch.context() as mp:
            force_stub_backbones(mp)
            ExperimentRunner(ConfigLoader().loads(_text(root, False, TRAD, DEEP))).run()
    finally:
        os.chdir(old)
    p = glob.glob(str(root / "out" / "verification" / "*" / "verification_results.json"))
    assert len(p) == 1
    j = json.load(open(p[0]))
    rows = list(csv.DictReader(open(p[0].replace(".json", ".csv"))))
    return j, rows


def test_all_ten_models_complete_in_one_run(ten):
    j, rows = ten
    assert j["status"] == "COMPLETE" and j["failures"] == [] and j["n_rows"] == 20
    assert {r["model"] for r in rows} == set(TRAD + DEEP)
    assert all(r["scores_file"] for r in rows if r["model"] in DEEP)      # deep models save score files


def test_stub_backbones_are_named_not_hidden(ten):
    j, rows = ten
    assert sorted(j["stub_backbones"]) == sorted(DEEP)                   # nothing installed here -> all four are placeholders
    bb = {r["model"]: r["backbone"] for r in rows}
    assert all(bb[m] == "STUB" for m in DEEP) and all(bb[m] == "n/a" for m in TRAD)


def test_require_real_models_refuses_placeholder_backbones(tmp_path, monkeypatch):
    force_stub_backbones(monkeypatch)
    T._make_fake_lfw(tmp_path)
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        r = ExperimentRunner(ConfigLoader().loads(_text(tmp_path, True, [], ["insightface"])))
        with pytest.raises(Exception) as e:
            r.run()
    finally:
        os.chdir(old)
    assert "require_real_models" in str(e.value) or "unavailable" in str(e.value)


@pytest.mark.parametrize("path", sorted(glob.glob(str(CONFIGS / "lfw_fold0_*.yaml"))), ids=lambda p: Path(p).stem)
def test_every_shipped_fold0_config_loads_and_keeps_the_frozen_protocol(path):
    cfg = ConfigLoader().load(path) if hasattr(ConfigLoader(), "load") else ConfigLoader().loads(open(path).read())
    assert str(cfg.evaluation_protocol) == "lfw_fold0_verification"
    assert cfg.random_seed == 42 and cfg.verification_fold == 0
    assert cfg.compute_calibration is False                               # shipped configs never silently add calibration


def test_extra_four_model_configs_cover_the_four_missing_models():
    got = {}
    for name in ("pca_svm", "knn"):
        got[name] = ConfigLoader().loads(open(CONFIGS / f"lfw_fold0_{name}.yaml").read()).traditional_models
    for name in ("insightface", "dlib_fr"):
        got[name] = ConfigLoader().loads(open(CONFIGS / f"lfw_fold0_deep_{name}.yaml").read()).deep_models
    assert {k: [str(x).split(".")[-1].lower() for x in v] for k, v in got.items()} == {k: [k] for k in got}
