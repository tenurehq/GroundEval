"""
groundeval
===============
A deterministic evaluation framework for AI agents.

No LLM-as-judge. All scoring is structural comparison against
seeded artifacts and task contracts.

Three tracks applied to every run:
  - COUNTERFACTUAL : did cited evidence support the conclusion?
  - SILENCE        : did the agent verify all preconditions?
  - PERSPECTIVE    : did the agent stay within permission boundaries?

Usage:
    python -m groundeval task --config config.yaml
"""

from .core import (
    AgentTrajectory,
    AccessPolicy,
    CorpusAdapter,
    GatedRuntime,
    TaskContract,
    TaskPrecondition,
    TaskEvalResult,
    ToolCall,
    ANSWER_SCHEMA_TASK,
)
from .adapters import (
    FileCorpusAdapter,
    InMemoryCorpusAdapter,
    NullCorpusAdapter,
    YamlAccessPolicy,
)
from .scorers import (
    score_task_run,
    aggregate_task_results,
)
from .task_eval import (
    run_task,
    run_all_tasks,
    build_task_question_text,
)


__all__ = [
    "AgentTrajectory",
    "AccessPolicy",
    "CorpusAdapter",
    "GatedRuntime",
    "TaskContract",
    "TaskPrecondition",
    "TaskEvalResult",
    "ToolCall",
    "ANSWER_SCHEMA_TASK",
    "FileCorpusAdapter",
    "InMemoryCorpusAdapter",
    "NullCorpusAdapter",
    "YamlAccessPolicy",
    "score_task_run",
    "aggregate_task_results",
    "run_task",
    "run_all_tasks",
    "build_task_question_text",
    "DistractorGenerator",
    "load_seed_artifacts",
]
