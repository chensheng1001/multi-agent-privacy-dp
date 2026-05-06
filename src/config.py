"""Configuration management for the multi-agent privacy experiment."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class APIConfig:
    """LLM API configuration."""
    base_url: str = "https://token-plan-sgp.xiaomimimo.com/v1"
    api_key: str = ""
    model: str = "mimo-v2.5-pro"
    temperature: float = 0.7
    max_tokens: int = 1024
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class ExperimentConfig:
    """Experiment parameters."""
    num_adversarial_scenarios: int = 60
    num_benign_scenarios: int = 59
    num_users_range: List[int] = field(default_factory=lambda: [8, 15])
    num_agents_range: List[int] = field(default_factory=lambda: [3, 5])
    random_seed: int = 42
    results_dir: str = "results"


@dataclass
class FactDPConfig:
    """Fact-Level Differential Privacy parameters."""
    epsilon: float = 2.0
    max_total_epsilon: float = 10.0
    lambda_tradeoff: float = 1.0
    composition: str = "advanced"
    delta: float = 1e-5


@dataclass
class Config:
    """Main configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    factdp: FactDPConfig = field(default_factory=FactDPConfig)
    defenses: List[str] = field(default_factory=lambda: ["none", "cot", "tom", "factdp"])
    evaluator_model: Optional[str] = None


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"[WARN] Config file {config_path} not found, using defaults.")
        return Config()

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    config = Config()
    if "api" in raw:
        config.api = APIConfig(**raw["api"])
    if "experiment" in raw:
        config.experiment = ExperimentConfig(**raw["experiment"])
    if "factdp" in raw:
        factdp_raw = raw["factdp"]
        # YAML may parse scientific notation (1e-5) as string, force float
        for key in ["epsilon", "max_total_epsilon", "lambda_tradeoff", "delta"]:
            if key in factdp_raw:
                factdp_raw[key] = float(factdp_raw[key])
        config.factdp = FactDPConfig(**factdp_raw)
    if "defenses" in raw:
        config.defenses = raw["defenses"]
    if "evaluator" in raw and raw["evaluator"].get("model"):
        config.evaluator_model = raw["evaluator"]["model"]

    return config
