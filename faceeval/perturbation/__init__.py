"""
faceeval.perturbation
======================
Perturbation engine for FaceEval-X.
Importing this package registers all built-in perturbations.
"""

# Auto-register all perturbations by importing their modules
import faceeval.perturbation.blur
import faceeval.perturbation.noise
import faceeval.perturbation.photometric
import faceeval.perturbation.geometric
import faceeval.perturbation.occlusion
import faceeval.perturbation.compression

from faceeval.perturbation.base import BasePerturbation, SeverityScheduler
from faceeval.perturbation.composer import PerturbationComposer

from faceeval.perturbation.blur import GaussianBlur, MotionBlur
from faceeval.perturbation.noise import GaussianNoise, SaltPepperNoise, SpeckleNoise
from faceeval.perturbation.photometric import (
    BrightnessPerturbation, ContrastPerturbation, GammaPerturbation,
)
from faceeval.perturbation.geometric import (
    RotationPerturbation, ScalingPerturbation, CroppingPerturbation,
)
from faceeval.perturbation.occlusion import RandomOcclusion, FaceMask, SunglassesPerturbation
from faceeval.perturbation.compression import JPEGCompression, ResolutionDegradation

__all__ = [
    "BasePerturbation", "SeverityScheduler", "PerturbationComposer",
    "GaussianBlur", "MotionBlur",
    "GaussianNoise", "SaltPepperNoise", "SpeckleNoise",
    "BrightnessPerturbation", "ContrastPerturbation", "GammaPerturbation",
    "RotationPerturbation", "ScalingPerturbation", "CroppingPerturbation",
    "RandomOcclusion", "FaceMask", "SunglassesPerturbation",
    "JPEGCompression", "ResolutionDegradation",
]
