"""The config hash (and with it ``run_id`` and the result store's duplicate-run detection) of every PRE-EXISTING
shipped configuration must be identical to its value at base commit aa8e83a. The values below were computed on an
unmodified checkout of aa8e83a; they must never be updated to make this test pass.

The new ``verification_fold`` field is serialised only when it is not the reference fold 0.
"""
from pathlib import Path

import pytest

from faceeval.core.config import ConfigLoader

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
HASHES_AT_AA8E83A = {
    "lfw_fold0_classical.yaml": "c93ed5eeba248f92d6bb14a265adbbc7db02c7378f6b98947fd6d1db1800075f",
    "lfw_fold0_deep_arcface.yaml": "8c5d5f6094c32980d859590bd30e9a9f1872e7da933165e9117b59e3c368e760",
    "lfw_fold0_deep_facenet.yaml": "142949aadd6b04792e16ec54679983cca0b5ce810479b6c689143e458d831d7b",
    "lfw_fold0_eigenfaces_final.yaml": "1000e6ff01a6ccfc388f12d17e6349f791f1644e3529925f0ceaca7a82a9795e",
    "lfw_fold0_eigenfaces_only.yaml": "8e136ab641e40f17ac9a85e01019727732daf914bac08a770142433984345230",
    "lfw_fold0_eigenfaces_per_image_noise.yaml": "9179f2e0c86e8e8d00cb6b7a985c12791398b7f59477cbce3eb77cd2b1f9682e",
    "lfw_fold0_verification_full.yaml": "7fea31686bae2e50e34d09bb8e6a0cb113d11a1a14cc80be09dbba1051695903",
    "lfw_full_benchmark.yaml": "81540f226e48f094053819a776a33743b3764dff4d504fd2bd9a5c819f64499d",
}


@pytest.mark.parametrize("name", sorted(HASHES_AT_AA8E83A))
def test_existing_config_hash_is_unchanged(name):
    cfg = ConfigLoader().loads((CONFIGS / name).read_text())
    assert cfg.config_hash == HASHES_AT_AA8E83A[name]


def test_non_default_fold_or_seed_changes_the_hash_and_default_fold_is_not_serialised():
    text = (CONFIGS / "lfw_fold0_deep_facenet.yaml").read_text()
    base = ConfigLoader().loads(text)
    assert "verification_fold" not in base.to_dict()
    fold3 = ConfigLoader().loads(text + "\nverification_fold: 3\n")
    assert fold3.to_dict()["verification_fold"] == 3 and fold3.config_hash != base.config_hash
    seed7 = ConfigLoader().loads(text.replace("random_seed: 42", "random_seed: 7"))
    assert seed7.config_hash != base.config_hash


def test_yaml_comments_are_documentation_only_but_the_description_field_is_part_of_the_identity():
    """Documented rule (docs/USER_GUIDE.md, 'Configuration identity'): '#' comments never enter the hash; the YAML
    ``description:`` value does. A stale sentence inside ``description:`` therefore cannot be corrected without issuing a
    new run identity, and is documented in an errata note instead."""
    text = (CONFIGS / "lfw_fold0_classical.yaml").read_text()
    base = ConfigLoader().loads(text)
    assert base.config_hash == HASHES_AT_AA8E83A["lfw_fold0_classical.yaml"]
    commented = ConfigLoader().loads(text + "\n# any new comment, however long, is documentation only\n")
    assert commented.config_hash == base.config_hash
    assert ConfigLoader().loads(text.replace("# ── Evaluation flags", "# ── Evaluation flags (reworded comment)")).config_hash == base.config_hash
    assert base.description.startswith("LFW Fold 0 pair-verification benchmark")
    changed = ConfigLoader().loads(text.replace("  LFW Fold 0 pair-verification benchmark", "  (edited) LFW Fold 0 pair-verification benchmark", 1))
    assert changed.config_hash != base.config_hash


def test_new_default_valued_fields_are_not_part_of_existing_identities():
    for name in HASHES_AT_AA8E83A:
        d = ConfigLoader().loads((CONFIGS / name).read_text()).to_dict()
        assert not {"verification_fold", "dlib_alignment", "classical_rng"} & set(d), name
