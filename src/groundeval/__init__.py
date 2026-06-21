"""
groundeval
===============
A generalizable, deterministic evaluation framework for AI agents.

No LLM-as-judge. All scoring is gate-based and verifiable against a
user-supplied event log + access policy.

Three tracks:
  - PERSPECTIVE   : epistemic discipline (actor visibility cone)
  - COUNTERFACTUAL: causal mechanism identification
  - SILENCE       : absence verification with search-space coverage

Usage:
    python -m groundeval generate --config config.yaml --events events.jsonl
    python -m groundeval eval --config config.yaml --questions eval_output/eval_questions.json
"""

from .core import (
    AbsenceRecord,
    AgentTrajectory,
    CausalJoinSpec,
    CausalLink,
    CausalLinkSpec,
    CorpusAdapter,
    AccessPolicy,
    EvalQuestion,
    EvalResult,
    GatedRuntime,
    LogEvent,
    PerspectiveConfig,
    SearchSpaceSelector,
    SilencePairSpec,
    ToolCall,
    ANSWER_SCHEMAS,
    load_events,
)
from .adapters import (
    FileCorpusAdapter,
    NullCorpusAdapter,
    YamlAccessPolicy,
    EventLogPolicy,
)
from .question_gen import (
    CausalLinkIndexer,
    AbsenceCatalogBuilder,
    QuestionGenerator,
)
from .scorers import (
    PerspectiveScorer,
    CounterfactualScorer,
    SilenceScorer,
    combine_scores,
    aggregate,
)

__all__ = [
    "AbsenceRecord",
    "AgentTrajectory",
    "CausalJoinSpec",
    "CausalLink",
    "CausalLinkSpec",
    "CorpusAdapter",
    "AccessPolicy",
    "EvalQuestion",
    "EvalResult",
    "GatedRuntime",
    "LogEvent",
    "PerspectiveConfig",
    "SearchSpaceSelector",
    "SilencePairSpec",
    "ToolCall",
    "ANSWER_SCHEMAS",
    "load_events",
    "FileCorpusAdapter",
    "NullCorpusAdapter",
    "YamlAccessPolicy",
    "EventLogPolicy",
    "CausalLinkIndexer",
    "AbsenceCatalogBuilder",
    "QuestionGenerator",
    "PerspectiveScorer",
    "CounterfactualScorer",
    "SilenceScorer",
    "combine_scores",
    "aggregate",
]
