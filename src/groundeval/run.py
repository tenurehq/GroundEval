"""
groundeval/run.py
======================
CLI entrypoint. One command:

    python -m groundeval task --config config.yaml

Runs task-contract evaluation. No event log required.
One config format: task_contracts with actors and roles.
One question type: task contract.
Three scoring tracks applied to every run.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import logging
from datetime import datetime
from pathlib import Path
import random as _random

import yaml

from .core import (
    AgentTrajectory,
    TaskContract,
)
from .adapters import (
    FileCorpusAdapter,
    YamlAccessPolicy,
)
from .task_eval import (
    run_all_tasks,
)

logger = logging.getLogger("groundeval")


def _merge_with_defaults(main_cfg: dict) -> dict:
    """Load evaluation defaults from config/evaluation.yaml and merge main_cfg on top."""
    defaults_path = Path("config/evaluation.yaml")
    if not defaults_path.exists():
        logger.warning(
            f"  Defaults file not found at {defaults_path}; using main config as-is."
        )
        return dict(main_cfg)

    with open(defaults_path) as f:
        defaults = yaml.safe_load(f)
        if not isinstance(defaults, dict):
            raise ValueError("evaluation.yaml must be a YAML mapping")

    merged = dict(defaults)
    merged.update(main_cfg)
    return merged


def _validate_config(cfg: dict) -> None:
    logger.info("  Config schema: OK")

    task_contracts_raw = cfg.get("task_contracts", [])
    if not task_contracts_raw:
        logger.warning("  No task_contracts defined in config.")

    for tc in task_contracts_raw:
        name = tc.get("name", "unnamed")
        preconditions = tc.get("preconditions", [])
        logger.info(f"  Task '{name}': {len(preconditions)} precondition(s)")

    any_fixture = any(tc.get("allowed_tools") for tc in task_contracts_raw)

    if not any_fixture:
        artifacts_dir = cfg.get("artifacts_dir", "./task_artifacts")
        art_path = Path(artifacts_dir)
        if not art_path.exists():
            raise FileNotFoundError(
                f"Artifacts directory '{artifacts_dir}' does not exist. "
                f"Set artifacts_dir in your config or create the directory."
            )
        artifacts = list(art_path.rglob("*.json"))
        if not artifacts:
            raise FileNotFoundError(
                f"Artifacts directory '{artifacts_dir}' contains no JSON files. "
                f"Add seed artifact files before running."
            )
        logger.info(f"  Artifacts: {len(artifacts)} JSON files in {artifacts_dir}")
    else:
        logger.info("  Artifacts: fixture mode (no corpus needed)")

    actors_declared = set(cfg.get("actors", {}).keys())
    roles_declared = set(cfg.get("roles", {}).keys())
    if actors_declared:
        logger.info(f"  Actors: {len(actors_declared)} declared")
    if roles_declared:
        logger.info(f"  Roles: {len(roles_declared)} declared")
        for role_name, role_cfg in cfg.get("roles", {}).items():
            subsystems = role_cfg.get("subsystems", [])
            logger.info(f"    {role_name}: {subsystems}")

    logger.info("Validation complete. No errors.")


def cmd_task(args) -> None:
    """
    Run task-contract evaluation.

    Loads task_contracts from config, seeds artifacts as ground truth,
    runs the agent against each task contract, and scores all three tracks.
    No event log required.
    """
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
        if not isinstance(raw_cfg, dict):
            raise ValueError(f"Expected YAML mapping, got {type(raw_cfg).__name__}")
        cfg: dict = raw_cfg

    cfg = _merge_with_defaults(cfg)

    from .config_schema import validate_config

    validate_config(cfg, command="task")
    _validate_config(cfg)

    seed = cfg.get("seed")
    if seed is not None:
        _random.seed(seed)
        logger.info(f"  Random seed set to {seed}")

    task_contracts_raw = cfg.get("task_contracts", [])
    if not task_contracts_raw:
        raise SystemExit("ERROR: No task_contracts defined in config.")

    contracts = [TaskContract.from_dict(tc) for tc in task_contracts_raw]
    logger.info(f"Loaded {len(contracts)} task contract(s)")

    artifacts_dir = cfg.get("artifacts_dir") or (
        contracts[0].artifacts_dir if contracts else "./task_artifacts"
    )
    logger.info(f"Artifacts directory: {artifacts_dir}")

    agent_fn = _build_agent_fn(cfg, contracts)

    actors = cfg.get("actors", {})
    roles = cfg.get("roles", {})
    if contracts and contracts[0].actors:
        actors = contracts[0].actors
    if contracts and contracts[0].roles:
        roles = contracts[0].roles
    policy = YamlAccessPolicy({"actors": actors, "roles": roles})
    logger.info("  Access policy: YamlAccessPolicy")

    results = run_all_tasks(
        contracts=contracts,
        agent_fn=agent_fn,
        artifacts_dir=artifacts_dir,
        policy=policy,
        max_steps=args.max_steps,
    )

    from .scorers import aggregate_task_results

    summary = aggregate_task_results(results)

    out_dir = Path(cfg.get("output_dir", "./eval_output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_model = args.model or cfg.get("model", "claude-sonnet-4-6")
    model_safe = resolved_model.replace("/", "_").replace(":", "_")
    out_path = out_dir / f"task_results_{model_safe}.json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "meta": {
                    "model": resolved_model,
                    "n_tasks": len(results),
                    "evaluation_mode": "task_contract",
                },
                "summary": summary,
            },
            f,
            indent=2,
        )

    logger.info(f"Task results written to {out_path}")
    logger.info(
        f"Overall -- "
        f"cf={summary['counterfactual_score']:.3f}  "
        f"sl={summary['silence_score']:.3f}  "
        f"ps={summary['perspective_score']:.3f}  "
        f"overall={summary['overall_score']:.3f}  "
        f"accuracy={summary['accuracy']:.3f}"
    )
    logger.info(f"Total violations: {summary['total_violations']}")

    for tr in results:
        logger.info(
            f"  {tr.task_name}: "
            f"counterfactual={tr.counterfactual_score:.3f} "
            f"silence={tr.silence_score:.3f} "
            f"perspective={tr.perspective_score:.3f} "
            f"overall={tr.overall_score:.3f}"
        )


def _build_agent_fn(cfg: dict, contracts: list | None = None) -> Any:
    agent_cfg = cfg.get("agent", {})

    if agent_cfg.get("framework") == "crewai":
        try:
            from .adapters.crewai_adapter import build_crewai_agent_fn
        except ImportError:
            raise ImportError(
                "CrewAI is required for the CrewAI adapter. "
                "Install it with: pip install groundeval[crewai]"
            )

        contract = contracts[0] if contracts else None

        return build_crewai_agent_fn(
            crew_class_path=agent_cfg["crew_class"],
            tool_map=agent_cfg.get("tool_map"),
            answer_key=agent_cfg.get("answer_key"),
            output_mode=agent_cfg.get("output_mode", "auto"),
            contract=contract,
        )

    from .providers import ModelProvider, build_agent_fn

    provider = ModelProvider.from_config(cfg)
    return build_agent_fn(provider)


def main():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(str(log_file), mode="a"),
            logging.StreamHandler(),
        ],
    )

    logger.info(f"Logging to {log_file}")

    parser = argparse.ArgumentParser(
        prog="groundeval",
        description="Deterministic agentic evaluation framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    task_parser = sub.add_parser(
        "task", help="Run task-contract evaluation (no event log needed)"
    )
    task_parser.add_argument(
        "--config", required=True, help="Path to config.yaml with task_contracts"
    )
    task_parser.add_argument("--model", default="claude-sonnet-4-6")
    task_parser.add_argument("--max-steps", type=int, default=10)

    val = sub.add_parser(
        "validate", help="Validate config + artifacts without running tasks"
    )
    val.add_argument("--config", required=True, help="Path to config.yaml")

    args = parser.parse_args()

    if args.command == "task":
        cmd_task(args)
    elif args.command == "validate":
        from .config_schema import validate_config

        with open(args.config) as f:
            raw_cfg = yaml.safe_load(f)
            if not isinstance(raw_cfg, dict):
                raise ValueError(f"Expected YAML mapping, got {type(raw_cfg).__name__}")
        validate_config(raw_cfg, command="validate")
        _validate_config(raw_cfg)


if __name__ == "__main__":
    main()
