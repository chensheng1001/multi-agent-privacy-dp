"""Ablation study: run FactDP with different (epsilon, lambda) combinations."""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ablation.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def run_single_scenario(scenario, config):
    """Run a single scenario with FactDP defense, return evaluation result."""
    llm = LLMClient(config.api)
    evaluator = Evaluator(llm)

    defenders = {}
    for agent_info in scenario["agents"]:
        defender = create_defender(
            defense_type="factdp",
            agent_id=agent_info["id"],
            role=agent_info["role"],
            knowledge_base=agent_info["knowledge_base"],
            dp_config=config.factdp,
            sensitive_target=scenario["sensitive_target"],
        )
        defenders[agent_info["id"]] = defender

    if scenario["scenario_type"] == "adversarial":
        attacker = Attacker(llm)
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
        return evaluator.evaluate_adversarial(scenario, attack_result, "factdp")
    else:
        benign_queries = scenario.get("benign_queries", [])
        responses = []
        for query in benign_queries[:3]:
            for agent_id, defender in defenders.items():
                try:
                    response = defender.respond(query, llm)
                    responses.append(response)
                except Exception:
                    pass
        return evaluator.evaluate_benign(scenario, responses, "factdp")


def run_ablation(config_path="config.yaml"):
    """Run FactDP ablation over epsilon × lambda combinations."""
    base_config = load_config(config_path)

    # Generate scenarios once
    generator = ScenarioGenerator(base_config.experiment)
    scenarios = generator.generate_all_scenarios()

    # Parameter grid
    epsilon_values = [2.0, 5.0, 10.0]
    lambda_values = [0.3, 0.5, 1.0]

    all_results = []
    total = len(epsilon_values) * len(lambda_values)
    combo_idx = 0

    for epsilon in epsilon_values:
        for lambda_val in lambda_values:
            combo_idx += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Ablation {combo_idx}/{total}: ε={epsilon}, λ={lambda_val}")
            logger.info(f"{'='*60}")

            ablation_config = copy.deepcopy(base_config)
            ablation_config.factdp.epsilon = epsilon
            ablation_config.factdp.lambda_tradeoff = lambda_val
            # Increase budget so it doesn't run out mid-scenario
            ablation_config.factdp.max_total_epsilon = max(50.0, epsilon * 10)

            leakage_correct = 0
            total_adv = 0
            total_blocking = 0.0
            benign_success = 0
            total_benign = 0

            for idx, scenario in enumerate(scenarios):
                scenario_label = scenario["scenario_id"]
                if (idx + 1) % 20 == 0:
                    logger.info(f"  Progress: {idx+1}/{len(scenarios)}")

                try:
                    result = run_single_scenario(scenario, ablation_config)

                    if scenario["scenario_type"] == "adversarial":
                        total_adv += 1
                        if result.get("leakage_correct", False):
                            leakage_correct += 1
                        total_blocking += result.get("blocking_rate", 0)
                    else:
                        total_benign += 1
                        if result.get("benign_success", False):
                            benign_success += 1
                except Exception as e:
                    logger.error(f"  ERROR on {scenario_label}: {type(e).__name__}: {e}")

            leak_acc = leakage_correct / total_adv if total_adv > 0 else 0
            avg_blocking = total_blocking / total_adv if total_adv > 0 else 0
            benign_rate = benign_success / total_benign if total_benign > 0 else 0
            balanced = ((1 - leak_acc) + benign_rate) / 2

            row = {
                "epsilon": epsilon,
                "lambda": lambda_val,
                "leakage_accuracy": round(leak_acc, 4),
                "blocking_rate": round(avg_blocking, 4),
                "benign_success_rate": round(benign_rate, 4),
                "balanced_outcome": round(balanced, 4),
                "adv_scenarios": total_adv,
                "benign_scenarios": total_benign,
            }
            all_results.append(row)
            logger.info(f"  Result: leak={leak_acc:.1%} block={avg_blocking:.1%} benign={benign_rate:.1%} balanced={balanced:.3f}")

    # Print summary table
    headers = ["ε", "λ", "Leak↓", "Block↑", "Benign↑", "Balanced↑"]
    table = []
    for r in all_results:
        table.append([
            r["epsilon"], r["lambda"],
            f"{r['leakage_accuracy']:.1%}",
            f"{r['blocking_rate']:.1%}",
            f"{r['benign_success_rate']:.1%}",
            f"{r['balanced_outcome']:.3f}",
        ])

    print("\n" + "=" * 70)
    print("ABLATION: ε × λ  (Fact-Level DP)")
    print("=" * 70)
    print(tabulate(table, headers=headers, tablefmt="grid"))

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_factdp.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nAblation results saved to results/ablation_factdp.json")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_ablation(config_path)
