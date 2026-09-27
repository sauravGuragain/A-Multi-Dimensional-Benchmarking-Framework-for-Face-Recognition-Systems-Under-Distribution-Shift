from __future__ import annotations

import numpy as np

from faceeval.core.types import ImageRecord


def generate_synthetic_dataset(
    n_subjects: int = 8,
    images_per_subject: int = 10,
    image_size: int = 112,
    seed: int = 42,
) -> tuple[list[ImageRecord], dict[str, np.ndarray]]:
    """
    Generate a synthetic in-memory face dataset.

    Parameters
    ----------
    n_subjects:
        Number of distinct synthetic identities.
    images_per_subject:
        Number of image variants per identity.
    image_size:
        Square image side length in pixels.
    seed:
        Random seed for full reproducibility.

    Returns
    -------
    (records, images)
        ``records`` is a list of ``ImageRecord`` (the framework's standard
        dataset unit). ``images`` maps ``image_id -> np.ndarray`` (uint8
        BGR, shape ``(image_size, image_size, 3)``) so the runner can look
        up pixel data without touching disk.
    """
    rng = np.random.default_rng(seed)
    records: list[ImageRecord] = []
    images: dict[str, np.ndarray] = {}

    # Smooth coordinate grids reused for every subject's base pattern
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    yy /= image_size
    xx /= image_size

    for s in range(n_subjects):
        subject_id = f"synth_{s:03d}"

        # Distinctive per-subject pattern: a smooth gradient with a
        # subject-specific phase/frequency, so subjects are genuinely
        # separable by both pixel-space (Eigenfaces) and texture (LBP/HOG)
        # methods.
        freq_x = rng.uniform(1.5, 4.0)
        freq_y = rng.uniform(1.5, 4.0)
        phase = rng.uniform(0, 2 * np.pi)
        base = 0.5 + 0.35 * np.sin(2 * np.pi * freq_x * xx + phase) \
                   + 0.35 * np.cos(2 * np.pi * freq_y * yy + phase / 2)
        base = np.clip(base, 0.05, 0.95)

        # Per-subject colour tint so colour-aware models also see structure
        tint = rng.uniform(0.7, 1.3, size=3).astype(np.float32)

        for i in range(images_per_subject):
            noise = rng.normal(0.0, 0.04, (image_size, image_size)).astype(np.float32)
            channel = np.clip(base + noise, 0.0, 1.0)
            img_f = np.stack([channel * tint[c] for c in range(3)], axis=-1)
            img_u8 = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)

            image_id = f"{subject_id}_{i:03d}"
            images[image_id] = img_u8
            records.append(ImageRecord(
                image_id=image_id,
                image_path=f"<synthetic>/{image_id}.png",
                subject_id=subject_id,
                dataset_name="synthetic",
                split="unassigned",
                metadata={"source": "synthetic_generator"},
            ))

    return records, images