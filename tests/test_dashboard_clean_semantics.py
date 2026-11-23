"""
Guards the dashboard's notion of the "clean" condition (dashboard/src/lib/clean.js)
against the real perturbation definitions.  It only applies perturbations to a
small synthetic array; it runs no model and touches no result or database.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from faceeval.core.types import PerturbationType
from faceeval.perturbation.composer import PerturbationComposer

CLEAN_JS = Path(__file__).resolve().parents[1] / "dashboard" / "src" / "lib" / "clean.js"


def _js_neutral_table() -> dict[str, float]:
    body = re.search(r"NEUTRAL_SEVERITY\s*=\s*\{(.*?)\}", CLEAN_JS.read_text(), re.S).group(1)
    return {k: float(v) for k, v in re.findall(r"(\w+)\s*:\s*([0-9.]+)", body)}


@pytest.fixture(scope="module")
def composer():
    return PerturbationComposer()


@pytest.fixture(scope="module")
def image():
    return np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)


def _apply(composer, ptype, image, severity):
    return composer._get_perturbation(ptype).apply(image, severity, image_id="img-0")


@pytest.mark.parametrize("ptype", list(PerturbationType))
def test_severity_zero_is_an_exact_copy_for_every_type(composer, image, ptype):
    out = _apply(composer, ptype, image, 0.0)
    assert np.array_equal(out, image)


def test_js_neutral_table_matches_the_perturbation_code(composer, image):
    # Every type whose severity-0.5 output is pixel-identical to the input
    # must be listed in the dashboard table, and nothing else may be.
    identity_at_half = {
        p.value for p in PerturbationType
        if np.array_equal(_apply(composer, p, image, 0.5), image)
    }
    table = _js_neutral_table()
    assert set(table) == identity_at_half == {"brightness", "gamma"}
    assert all(v == 0.5 for v in table.values())


def test_contrast_has_no_exact_neutral_at_half(composer, image):
    spec = composer._get_perturbation(PerturbationType.CONTRAST)._severity_to_params(0.5)
    assert spec["scale_factor"] != 1.0                       # 0.94868
    assert not np.array_equal(_apply(composer, PerturbationType.CONTRAST, image, 0.5), image)
    assert "contrast" not in _js_neutral_table()


def test_brightness_and_gamma_parameters_at_half_are_identity(composer):
    b = composer._get_perturbation(PerturbationType.BRIGHTNESS)._severity_to_params(0.5)
    g = composer._get_perturbation(PerturbationType.GAMMA)._severity_to_params(0.5)
    assert b["additive_shift"] == 0.0 and g["gamma"] == 1.0


# --- the real JavaScript classifier vs. the real perturbation outputs -----------

import json
import shutil
import subprocess

_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_classifier_agrees_with_pixel_identity_for_every_type_and_severity(composer, image):
    """isCleanCondition (clean.js) must say "clean" exactly when the perturbation
    leaves the synthetic image pixel-identical, for all 16 types x the severity grid."""
    types = [p.value for p in PerturbationType]
    expected = {
        f"{p}@{s:g}": bool(np.array_equal(_apply(composer, PerturbationType(p), image, s), image))
        for p in types for s in _GRID
    }
    script = (
        f'import {{ isCleanCondition }} from {json.dumps(CLEAN_JS.as_uri())};'
        f'const types = {json.dumps(types)}; const grid = {json.dumps(_GRID)};'
        'const out = {}; for (const t of types) for (const s of grid)'
        ' out[`${t}@${s}`] = isCleanCondition({perturbation_type: t, severity: s});'
        'console.log(JSON.stringify(out));'
    )
    run = subprocess.run(["node", "--input-type=module", "-e", script],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == expected
    # sanity: the table really covers 16 types and contains both outcomes
    assert len(types) == 16 and any(expected.values()) and not all(expected.values())
