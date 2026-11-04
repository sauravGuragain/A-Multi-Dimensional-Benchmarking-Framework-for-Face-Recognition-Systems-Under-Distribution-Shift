"""Apply perturbations in the pixel domain the perturbation classes are written for.

Perturbations (``faceeval.perturbation``) expect uint8 images or float images in [0, 1].
Models whose normalisation leaves that range (ArcFace/InsightFace [-1, 1], FaceNet
per-image standardisation, ImageNet mean/std) must therefore be perturbed on the aligned
uint8 chip and normalised afterwards. Models already normalised to [0, 1] (grayscale
traditional models, Dlib, unit range) keep the original behaviour: perturb the normalised
array directly.
"""
from __future__ import annotations

from faceeval.preprocessing.normalizer import NormalizationMode

CHIP_MODES = frozenset({
    NormalizationMode.ARCFACE,
    NormalizationMode.FACENET,
    NormalizationMode.IMAGENET,
})


def needs_chips(normalizer) -> bool:
    """True when perturbations must be applied to uint8 chips before normalisation."""
    return normalizer.mode in CHIP_MODES


def perturb_batch(composer, spec, normalizer, clean_arrays, chips=None) -> list:
    """Return the perturbed, model-ready arrays for one perturbation spec."""
    perturbation = composer._get_perturbation(spec.perturbation_type)
    if chips is None:
        return [perturbation.apply(arr, spec.severity) for arr in clean_arrays]
    return [normalizer.normalize(perturbation.apply(chip, spec.severity)) for chip in chips]
