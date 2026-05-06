"""Main experiment runner: orchestrates scenarios, defenses, and evaluation."""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from tabulate import tabulate

from src.config import load_config, Config
from src.llm_client import LLMClient
from src.scenario_gen import ScenarioGenerator
from src.defenders import create_defender, BaseDefender
from src.attacker import Attacker
from src.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiment.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def run_single_scenario(scenario: Dict, defense_type: str,
                         llm: LLMClient, config: Config) -> Dict[str, Any]:
    """
    Run a single scenario with the specified defense type.

    Returns:
        evaluation result dict
    """
    evaluator = Evaluator(llm)

    # Create defenders for each agent in the scenario
    defenders: Dict[str, BaseDefender] = {}
    for agent_info in scenario["agents"]:
        defender = create_defender(
            defense_type=defense_type,
            agent_id=agent_info["id"],
            role=agent_info["role"],
            knowledge_base=agent_info["knowledge_base"],
            dp_config=config.factdp if defense_type == "factdp" else None,
            sensitive_target=scenario["sensitive_target"] if defense_type == "factdp" else None,
        )
        defenders[agent_info["id"]] = defender

    attacker = Attacker(llm)

    if scenario["scenario_type"] == "adversarial":
        # Execute adversarial plan
        attack_result = attacker.execute_plan(
            plan=scenario["adversarial_plan"],
            defenders=defenders,
            context=scenario["context"],
        )

        # Infer sensitive attribute
        target = scenario["sensitive_target"]
        inference = attacker.infer_sensitive(
            target_attribute=target["attribute"],
            target_value=target["value"],
            target_user=target["user_name"],
        )

        attack_result["inference"] = inference
        return evaluator.evaluate_adversarial(scenario, attack_result, defense_type)

    else:  # benign
        # Execute benign queries
        benign_results = attacker.execute_benign_queries(
            benign_queries=scenario["benign_queries"],
            defenders=defenders,
        )
        return evaluator.evaluate_benign(scenario, benign_results, defense_type)


def run_experiment(config: Config) -> Dict[str, Any]:
    """
    Run the full experiment: all scenarios × all defense types.

    Returns:
        dict with all results and aggregate metrics
    """
    logger.info("=" * 70)
    logger.info("Multi-Agent Privacy Experiment: Fact-Level Differential Privacy")
    logger.info("=" * 70)

    # Initialize LLM client
    llm = LLMClient(config.api)
    logger.info(f"Using model: {config.api.model} at {config.api.base_url}")

    # Generate scenarios
    logger.info("Generating scenarios...")
    generator = ScenarioGenerator(config.experiment)
    scenarios = generator.generate_all_scenarios()

    adv_count = sum(1 for s in scenarios if s["scenario_type"] == "adversarial")
    ben_count = sum(1 for s in scenarios if s["scenario_type"] == "benign")
    logger.info(f"Generated {len(scenarios)} scenarios ({adv_count} adversarial, {ben_count} benign)")

    # Run each defense type
    all_results = {}
    all_metrics = {}

    for defense_type in config.defenses:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Running defense: {defense_type}")
        logger.info(f"{'─' * 50}")

        results = []
        start_time = time.time()

        for idx, scenario in enumerate(scenarios):
            if (idx + 1) % 10 == 0 or idx == 0:
                logger.info(f"  Scenario {idx+1}/{len(scenarios)}: {scenario['scenario_id']}")

            try:
                result = run_single_scenario(scenario, defense_type, llm, config)
                results.append(result)
            except Exception as e:
                logger.error(f"  ERROR on {scenario['scenario_id']} ({defense_type}): {type(e).__name__}: {e}",
                             exc_info=True)
                results.append({
                    "scenario_id": scenario["scenario_id"],
                    "defense_type": defense_type,
                    "error": str(e),
                    "scenario_type": scenario["scenario_type"],
                })

        elapsed = time.time() - start_time
        logger.info(f"  Completed {defense_type} in {elapsed:.1f}s")

        # Compute aggregate metrics
        metrics = Evaluator.compute_aggregate_metrics(results)
        metrics["defense_type"] = defense_type
        metrics["elapsed_seconds"] = elapsed

        all_results[defense_type] = results
        all_metrics[defense_type] = metrics

    return {
        "results": all_results,
        "metrics": all_metrics,
        "config": {
            "model": config.api.model,
            "num_scenarios": len(scenarios),
            "defenses": config.defenses,
            "factdp_epsilon": config.factdp.epsilon,
            "factdp_lambda": config.factdp.lambda_tradeoff,
        },
        "llm_usage": llm.get_usage_stats(),
    }


def print_results(metrics: Dict[str, Dict]):
    """Print a formatted summary table."""
    headers = ["Defense", "Leakage Acc ↓", "Blocking Rate ↑",
               "Benign Success ↑", "Balanced Outcome ↑"]
    rows = []

    for defense, m in sorted(metrics.items()):
        rows.append([
            defense,
            f"{m.get('leakage_accuracy', 0):.1%}",
            f"{m.get('avg_blocking_rate', 0):.1%}",
            f"{m.get('benign_success_rate', 0):.1%}",
            f"{m.get('balanced_outcome', 0):.1%}",
        ])

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()


def save_results(experiment_output: Dict, results_dir: str):
    """Save results to JSON files."""
    os.makedirs(results_dir, exist_ok=True)

    # Save full results
    results_path = os.path.join(results_dir, "full_results.json")
    with open(results_path, "w") as f:
        json.dump(experiment_output, f, indent=2, default=str)
    logger.info(f"Full results saved to {results_path}")

    # Save metrics summary
    metrics_path = os.path.join(results_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(experiment_output["metrics"], f, indent=2, default=str)
    logger.info(f"Metrics summary saved to {metrics_path}")


def main():
    """Main entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    logger.info(f"Config loaded from {config_path}")
    logger.info(f"Defenses to run: {config.defenses}")

    # Run experiment
    experiment_output = run_experiment(config)

    # Print results
    print_results(experiment_output["metrics"])

    # Save results
    save_results(experiment_output, config.experiment.results_dir)

    # Print LLM usage
    usage = experiment_output["llm_usage"]
    logger.info(f"LLM Usage: {usage['total_calls']} calls, "
                f"{usage['total_tokens']} tokens")

    logger.info("Experiment complete!")


if __name__ == "__main__":
    main()
