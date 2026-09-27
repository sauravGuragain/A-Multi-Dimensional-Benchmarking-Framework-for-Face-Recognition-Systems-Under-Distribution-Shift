from __future__ import annotations

from typing import Any

from faceeval.core.types import DeploymentScenario, DeploymentWeights


# ---------------------------------------------------------------------------
# Built-in scenario definitions
# ---------------------------------------------------------------------------

# All weight vectors are validated at construction (must sum to 1.0).
# Weights are: accuracy, robustness, calibration, fairness, latency, memory,
#              computational_cost

SCENARIO_WEIGHTS: dict[DeploymentScenario, DeploymentWeights] = {

    DeploymentScenario.BORDER_CONTROL: DeploymentWeights(
        accuracy=0.30,          # Core requirement
        robustness=0.25,        # Must handle travel conditions
        calibration=0.10,       # Confidence matters for borderline cases
        fairness=0.15,          # Equal treatment is legally mandated
        latency=0.08,           # Throughput matters but not bottleneck
        memory=0.06,            # Server-side deployment
        computational_cost=0.06,
    ),

    DeploymentScenario.ACCESS_CONTROL: DeploymentWeights(
        accuracy=0.28,
        robustness=0.18,        # Controlled environment → lower
        calibration=0.08,
        fairness=0.12,
        latency=0.18,           # Turnstile speed matters
        memory=0.08,
        computational_cost=0.08,
    ),

    DeploymentScenario.MOBILE_AUTHENTICATION: DeploymentWeights(
        accuracy=0.22,
        robustness=0.15,        # Controlled (self-facing camera)
        calibration=0.08,
        fairness=0.10,
        latency=0.22,           # Must be sub-second
        memory=0.13,            # Limited phone RAM
        computational_cost=0.10,
    ),

    DeploymentScenario.SURVEILLANCE: DeploymentWeights(
        accuracy=0.20,
        robustness=0.35,        # Extreme variation in conditions
        calibration=0.08,
        fairness=0.15,          # Critical to avoid discriminatory targeting
        latency=0.08,
        memory=0.07,
        computational_cost=0.07,
    ),

    DeploymentScenario.RESEARCH_BENCHMARK: DeploymentWeights(
        accuracy=0.20,
        robustness=0.20,
        calibration=0.15,       # Research cares about uncertainty
        fairness=0.15,
        latency=0.10,
        memory=0.10,
        computational_cost=0.10,
    ),

    DeploymentScenario.CUSTOM: DeploymentWeights(
        # Equal weights as neutral default; override via config
        accuracy=1/7,
        robustness=1/7,
        calibration=1/7,
        fairness=1/7,
        latency=1/7,
        memory=1/7,
        computational_cost=1/7,
    ),
}


# ---------------------------------------------------------------------------
# Scenario metadata (descriptions and key constraints)
# ---------------------------------------------------------------------------

SCENARIO_METADATA: dict[DeploymentScenario, dict[str, Any]] = {
    DeploymentScenario.BORDER_CONTROL: {
        "display_name": "Border Control",
        "description": (
            "High-security identity verification at national borders. "
            "Near-zero false accept rate mandatory. Must handle varied "
            "lighting, pose, and partial occlusion (e.g. masks, hats)."
        ),
        "critical_metrics": ["accuracy", "robustness", "fairness"],
        "acceptable_far": 0.001,      # <0.1% FAR
        "minimum_accuracy": 0.95,
        "environment": "semi-controlled",
    },
    DeploymentScenario.ACCESS_CONTROL: {
        "display_name": "Physical Access Control",
        "description": (
            "Workplace entry or building security. Balance throughput "
            "with security. Indoor controlled lighting. Speed is key."
        ),
        "critical_metrics": ["accuracy", "latency"],
        "acceptable_far": 0.01,
        "minimum_accuracy": 0.90,
        "environment": "controlled",
    },
    DeploymentScenario.MOBILE_AUTHENTICATION: {
        "display_name": "Mobile Device Authentication",
        "description": (
            "Smartphone unlock or app authentication. Model must run "
            "on-device within tight latency and memory budgets."
        ),
        "critical_metrics": ["latency", "memory", "accuracy"],
        "acceptable_far": 0.01,
        "minimum_accuracy": 0.92,
        "environment": "user-controlled",
    },
    DeploymentScenario.SURVEILLANCE: {
        "display_name": "Video Surveillance",
        "description": (
            "Continuous identification in public spaces. Extreme robustness "
            "required. Fairness critical to prevent discriminatory surveillance."
        ),
        "critical_metrics": ["robustness", "fairness"],
        "acceptable_far": 0.05,
        "minimum_accuracy": 0.75,
        "environment": "uncontrolled",
    },
    DeploymentScenario.RESEARCH_BENCHMARK: {
        "display_name": "Research Benchmark",
        "description": (
            "Academic benchmarking with no specific deployment context. "
            "Uniform weights provide an unbiased model comparison baseline."
        ),
        "critical_metrics": ["accuracy", "calibration", "fairness"],
        "acceptable_far": None,
        "minimum_accuracy": None,
        "environment": "laboratory",
    },
    DeploymentScenario.CUSTOM: {
        "display_name": "Custom Scenario",
        "description": "User-defined weights from configuration file.",
        "critical_metrics": [],
        "acceptable_far": None,
        "minimum_accuracy": None,
        "environment": "custom",
    },
}


def get_weights(scenario: DeploymentScenario) -> DeploymentWeights:
    """Return the default ``DeploymentWeights`` for a scenario."""
    return SCENARIO_WEIGHTS[scenario]


def get_metadata(scenario: DeploymentScenario) -> dict[str, Any]:
    """Return descriptive metadata for a scenario."""
    return SCENARIO_METADATA[scenario]


def weights_from_dict(weights_dict: dict[str, float]) -> DeploymentWeights:
    """
    Build a ``DeploymentWeights`` from a plain dict.

    Used to load custom weights from YAML configuration.  Raises
    ``ValueError`` if keys are missing or weights don't sum to 1.0.
    """
    required = {
        "accuracy", "robustness", "calibration",
        "fairness", "latency", "memory", "computational_cost",
    }
    missing = required - set(weights_dict.keys())
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")
    return DeploymentWeights(**{k: weights_dict[k] for k in required})


def list_scenarios() -> list[dict[str, Any]]:
    """
    Return a summary list of all built-in scenarios for the dashboard.
    """
    return [
        {
            "scenario": scenario.value,
            "display_name": meta["display_name"],
            "description": meta["description"],
            "weights": {
                "accuracy": w.accuracy,
                "robustness": w.robustness,
                "calibration": w.calibration,
                "fairness": w.fairness,
                "latency": w.latency,
                "memory": w.memory,
                "computational_cost": w.computational_cost,
            },
        }
        for scenario, (w, meta) in zip(
            SCENARIO_WEIGHTS.keys(),
            zip(SCENARIO_WEIGHTS.values(), SCENARIO_METADATA.values()),
        )
    ]
