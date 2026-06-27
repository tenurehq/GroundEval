# Changelog

All notable changes to GroundEval will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.01]

### Added

- **Observe Mode** (`observe`): Introduced a new CLI command that records an existing agent's tool calls, evidence, and final answer, then generates a draft evaluation config for review. Supports `--framework`, `--agent-class`, `--draft-mode` (conservative/standard/aggressive), and `--no-draft` flags.
- **`draft` CLI command**: Added a standalone command to regenerate a draft config from a previously saved `observed_run.json` file, decoupling observation from config generation.
- **`CrewAIObserver` class** (`crewai_adapter.py`): Added an observer implementation for the CrewAI framework that deep-copies the crew, instruments tools with recording hooks, and captures tool calls, arguments, return values, and latency during observation runs.
- **Draft config validation gate**: Added a warning when running `task` against a config generated from observation that has not been marked as reviewed, requiring either `--allow-draft-config` or the `validate --mark-reviewed` flow before scoring.
- **`validate --mark-reviewed` flag**: Added a CLI flag that updates a draft config's `groundeval` block from `config_status: draft` to `config_status: reviewed`, enabling the review confirmation workflow.
- **`groundeval` top-level config key**: Added a new top-level key (`config_status`, `generated_from_observation`, `reviewed`, `draft_mode`) to config schema to track whether a config was generated from observation and its review state.

### Changed

- **CrewAI adapter field rename** (`run.py`): Changed the agent config key from `crew_class` to `agent_class` in the CrewAI agent builder, aligning with the generalized observer interface.
- **CrewAI adapter import path** (`run.py`): Updated the CrewAI adapter import from `groundeval.adapters.crewai_adapter` to `groundeval.framework_adapters.crewai_adapter`.
- **README overhaul** (`README.md`): Rewrote the project description, problem statement, and track explanations with clearer, more direct language. Added a full Observe Mode quickstart section, reorganized the demo section, and removed the design principles section.
- **CrewAI docs restructured** (`docs/crewai.md`): Reorganized the guide into a numbered step-by-step flow, added an Observe Mode section as step 2, clarified corpus vs. fixture mode descriptions, renamed `crew_class` references to `agent_class`, and added a known constraints section.
- **Output score labels** (`run.py`): Changed score display labels from abbreviated (`cf`, `sl`, `ps`) to full names (`counterfactual`, `silence`, `perspective`).
- **Default config path** (`run.py`): Changed the default validation config path from `config/config.yaml` to `config/evaluation.yaml`.
- **Config validation**: Added `groundeval` to the set of known top-level config keys (`config_schema.py`).

### Removed

- **Docstrings from `run.py`**: Removed module-level and function-level docstrings from the CLI entrypoint.
- **Unused `_merge_with_defaults` docstring**: Removed the docstring from the internal merge helper.
- **Test section comments**: Removed comment separators from `tests/test_run.py`.