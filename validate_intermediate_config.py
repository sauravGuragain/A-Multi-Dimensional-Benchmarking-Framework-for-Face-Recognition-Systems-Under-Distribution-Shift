#!/usr/bin/env python3
"""
Configuration validation for lfw_intermediate.yaml
VALIDATION ONLY - does NOT run the benchmark
"""

import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

print("=" * 80)
print("LFW INTERMEDIATE CONFIGURATION VALIDATION")
print("=" * 80)

print("\n[1/3] Loading configuration...")
try:
    from faceeval.core.config import ConfigLoader

    config = ConfigLoader().load("configs/lfw_intermediate.yaml")
    print("✓ Configuration loaded successfully")
except Exception as e:
    print(f"✗ Configuration load failed: {e}")
    sys.exit(1)

print("\n[2/3] Resolved configuration values:")
print("-" * 80)

config_values = {
    "experiment_name": config.experiment_name,
    "dataset_names": config.dataset_names,
    "min_images_per_subject": getattr(config, "min_images_per_subject", "NOT SET"),
    "max_subjects": getattr(config, "max_subjects", "NOT SET"),
    "split_strategy": config.split_strategy,
    "test_fraction": config.test_fraction,
    "val_fraction": config.val_fraction,
    "traditional_models": config.traditional_models,
    "deep_models": config.deep_models,
    "perturbation_types": config.perturbation_schedule.perturbation_types,
    "severity_levels": config.perturbation_schedule.severity_levels,
    "compose_perturbations": config.perturbation_schedule.compose_perturbations,
    "compute_calibration": config.compute_calibration,
    "compute_fairness": config.compute_fairness,
    "compute_failure_analysis": config.compute_failure_analysis,
    "compute_deployment_scores": config.compute_deployment_scores,
    "compute_fingerprints": config.compute_fingerprints,
    "device": config.device,
}

for key, value in config_values.items():
    print(f"  {key:.<40} {value}")

print("\n[3/3] Expected evaluation condition count:")
print("-" * 80)

num_models = len(config.traditional_models) + len(config.deep_models)
num_perturbations = len(config.perturbation_schedule.perturbation_types)
num_severity_levels = len(config.perturbation_schedule.severity_levels)

expected_conditions = (
    num_models * num_perturbations * num_severity_levels
)

print(f"  Traditional models: {len(config.traditional_models)} {config.traditional_models}")
print(f"  Deep models:        {len(config.deep_models)} {config.deep_models}")
print(f"  Total models:       {num_models}")
print()
print(
    f"  Perturbation types: {num_perturbations} "
    f"{config.perturbation_schedule.perturbation_types}"
)
print(
    f"  Severity levels:    {num_severity_levels} "
    f"{config.perturbation_schedule.severity_levels}"
)
print()
print(
    f"  Expected conditions = {num_models} models × "
    f"{num_perturbations} perturbations × "
    f"{num_severity_levels} severity levels"
)
print(f"  Expected conditions = {expected_conditions}")

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

checks = {
    "Configuration loads": True,
    "Experiment name set": bool(config.experiment_name),
    "Dataset is 'lfw'": config.dataset_names == ["lfw"],
    "Min images per subject set": (
        getattr(config, "min_images_per_subject", None) == 8
    ),
    "Max subjects set to 200": (
        getattr(config, "max_subjects", None) == 200
    ),
    "Split strategy is 'stratified'": (
        config.split_strategy == "stratified"
    ),
    "4 models defined": num_models == 4,
    "3 perturbations defined": num_perturbations == 3,
    "3 severity levels defined": num_severity_levels == 3,
    "Expected condition count is 36": expected_conditions == 36,
    "All compute flags enabled": all([
        config.compute_calibration,
        config.compute_fairness,
        config.compute_failure_analysis,
        config.compute_deployment_scores,
        config.compute_fingerprints,
    ]),
    "Device is 'cpu'": config.device == "cpu",
}

all_passed = True

for check_name, passed in checks.items():
    status = "✓" if passed else "✗"
    print(f"  {status} {check_name}")
    if not passed:
        all_passed = False

print("\n" + "=" * 80)

if all_passed and expected_conditions == 36:
    print("RESULT: PASS ✓")
    print(
        f"\nConfiguration is valid. "
        f"Expected {expected_conditions} evaluation conditions."
    )
    print("\nVALIDATION ONLY — benchmark was NOT run.")
    sys.exit(0)
else:
    print("RESULT: FAIL ✗")
    print("\nConfiguration validation failed.")
    sys.exit(1)
