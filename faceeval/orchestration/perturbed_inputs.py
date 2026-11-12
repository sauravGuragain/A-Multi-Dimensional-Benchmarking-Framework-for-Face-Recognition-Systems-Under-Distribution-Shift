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


def perturb_batch(composer, spec, normalizer, clean_arrays, chips=None, image_ids=None) -> list:
    """Return the perturbed, model-ready arrays for one perturbation spec.

    ``image_ids`` (parallel to the images) seeds the per-image noise; see ``perturbation.noise``.
    """
    perturbation = composer._get_perturbation(spec.perturbation_type)
    source = clean_arrays if chips is None else chips
    ids = [None] * len(source) if image_ids is None else image_ids
    if len(ids) != len(source):
        raise ValueError(f"image_ids has {len(ids)} entries for {len(source)} images")
    out = [perturbation.apply(img, spec.severity, image_id=i) for img, i in zip(source, ids)]
    return out if chips is None else [normalizer.normalize(o) for o in out]
