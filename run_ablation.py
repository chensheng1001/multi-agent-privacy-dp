"""Ablation study: run FactDP with different epsilon values."""

import os
import sys
import json
import copy
import logging
from tabulate import tabulate

from src.config import load_config, Config
from src.llm_client import LLMClient
from src.scenario_gen import ScenarioGenerator
from src.defenders import create_defender
from src.attacker import Attacker
from src.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_epsilon_ablation(epsilon_values, config_path="config.yaml"):
    """Run FactDP defense with different epsilon values."""
    config = load_config(config_path)
    llm = LLMClient(config.api)

    # Generate scenarios once (shared across ablation)
    generator = ScenarioGenerator(config.experiment)
    scenarios = generator.generate_all_scenarios()

    # Only run adversarial scenarios for ablation
    adv_scenarios = [s for s in scenarios if s["scenario_type"] == "adversarial"]

    results_table = []

    for epsilon in epsilon_values:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running ablation with ε = {epsilon}")
        logger.info(f"{'='*50}")

        # Override epsilon
        ablation_config = copy.deepcopy(config)
        ablation_config.factdp.epsilon = epsilon

        leakage_results = []
        for idx, scenario in enumerate(adv_scenarios):
            if (idx + 1) % 10 == 0:
                logger.info(f"  Scenario {idx+1}/{len(adv_scenarios)}")

            # Create FactDP defenders
            defenders = {}
            for agent_info in scenario["agents"]:
                defender = create_defender(
                    defense_type="factdp",
                    agent_id=agent_info["id"],
                    role=agent_info["role"],
                    knowledge_base=agent_info["knowledge_base"],
                    dp_config=ablation_config.factdp,
                    sensitive_target=scenario["sensitive_target"],
                )
                defenders[agent_info["id"]] = defender

            attacker = Attacker(llm)
            evaluator = Evaluator(llm)

            try:
                attack_result = attacker.execute_plan(
                    plan=scenario["adversarial_plan"],
                    defenders=defenders,
                    context=scenario["context"],
                )
                target = scenario["sensitive_target"]
                inference = attacker.infer_sensitive(
                    target_attribute=target["attribute"],
                    target_value=target["value"],
                    target_user=target["user_name"],
                )
                attack_result["inference"] = inference
                eval_result = evaluator.evaluate_adversarial(
                    scenario, attack_result, "factdp"
                )
                leakage_results.append(eval_result)
            except Exception as e:
                logger.error(f"  ERROR: {e}")

        # Compute metrics
        if leakage_results:
            leak_acc = sum(1 for r in leakage_results if r.get("leakage_correct", False)) / len(leakage_results)
            avg_blocking = sum(r.get("blocking_rate", 0) for r in leakage_results) / len(leakage_results)
            results_table.append([epsilon, f"{leak_acc:.1%}", f"{avg_blocking:.1%}", len(leakage_results)])

    # Print ablation results
    headers = ["ε", "Leakage Accuracy ↓", "Blocking Rate ↑", "Scenarios"]
    print("\n" + "=" * 60)
    print("ABLATION: ε vs Privacy-Utility Trade-off")
    print("=" * 60)
    print(tabulate(results_table, headers=headers, tablefmt="grid"))

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_epsilon.json", "w") as f:
        json.dump(results_table, f, indent=2)
    logger.info("\nAblation results saved to results/ablation_epsilon.json")


if __name__ == "__main__":
    epsilons = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_epsilon_ablation(epsilons, config_path)
