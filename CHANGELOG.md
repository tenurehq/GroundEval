# Changelog

All notable changes to GroundEval will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v0.04]

### Added

- **OpenAI Agents SDK support**: Added OpenAI Agents SDK to the framework support matrix, observer auto-registration, framework-specific reporting, CLI guidance, and optional dependency groups.
- **Framework registry and normalization**: Added centralized framework adapter detection and name normalization for configuration validation and CLI routing.
- **Deterministic multi-agent scoring**: Added scoring for required agents, runtime handoffs, and agent-specific tool expectations, including argument and return-value matching.
- **Expanded observed trajectory metadata**: Added observed agents, observed handoffs, per-call agent IDs, and observed tool return values to framework trajectories.
- **LangGraph runtime handoff capture**: Added branch-scoped detection of runtime node transitions while preserving static graph edges separately.
- **LangGraph stream shape support**: Added support for dictionary, two-item tuple, and namespace-aware three-item tuple stream events.
- **LangGraph recursion limits**: Added `max_steps` enforcement through LangGraph's `recursion_limit` configuration for synchronous and asynchronous streams.
- **MAF step-limit configuration**: Added best-effort `max_steps`, `max_iterations`, and `max_turns` configuration across common MAF entry-object settings.
- **Expanded adapter tests**: Added coverage for LangGraph stream formats, runtime handoffs, branch behavior, recursion limits, agent serialization, and invalid step limits.

### Changed

- **Framework detection behavior**: Replaced hard-coded framework lists with the shared adapter registry across config validation and task routing.
- **Framework config validation**: Updated agent-class requirements, observe scoring behavior, and multi-agent compatibility checks to recognize all registered framework adapters.
- **Framework-native overall scoring**: Changed framework scoring so the base Counterfactual, Silence, and Perspective score is multiplied by the multi-agent requirement score.
- **Framework correctness criteria**: Changed framework runs to count as correct only when answer checks pass and every configured multi-agent requirement is satisfied.
- **LangGraph handoff capabilities**: Changed the `handoffs` capability to represent observed runtime transitions, with static graph edges exposed separately.
- **LangGraph agent serialization**: Changed normalized LangGraph agents to use `ObservedAgent` objects for consistent serialization and deserialization.
- **Observed final-answer parsing**: Changed framework observation handling to parse structured JSON final outputs before scoring when possible.
- **Tool-return scoring**: Changed framework scoring to use observed normalized return values carried into each trajectory tool call.
- **CLI framework guidance**: Updated the `observe --framework` help text to include CrewAI, MAF, LangGraph, and OpenAI Agents SDK.
- **MAF task-path behavior**: Consolidated framework routing through the generic observe-and-score path rather than maintaining a dedicated MAF evaluation stub.
- **Lint configuration**: Added Ruff `E501` suppression while retaining the configured maximum line length.
- **Code formatting**: Applied consistent formatting and quoting across configuration, runtime, scoring, CLI, and adapter modules.

### Removed

- **Legacy MAF evaluation stub**: Removed `_build_maf_eval_agent_fn`, which previously raised an error directing MAF users to `observe --score`.
- **Obsolete adapter tests and imports**: Removed tests and unused imports tied to the deleted MAF stub, older LangGraph workflow behavior, and unused test dependencies.

---

## [v0.03]

### Added

- **Framework-native observe scoring**: Added `groundeval observe --score` so reviewed configs can score fresh observed runs directly, writing timestamped scored outputs alongside observation artifacts.
- **Observe diagram PDF**: Added `observe_diagram.pdf` as a standard observe artifact, providing a behavior-only visual summary of tool calls, arguments, returns, evidence tags, swimlanes, handoffs, and final output when available.
- **Framework support expansion**: Added implemented observe mode and deterministic scoring support for **LangGraph** and **Microsoft Agent Framework**, alongside new framework docs and framework-specific observe artifacts.
- **`compare` CLI command**: Added a JSON comparison workflow for reviewing changes between GroundEval outputs, including observed score outputs and task result files.
- **MAF dependency group**: Added a new `maf` dependency group in `pyproject.toml` for Microsoft Agent Framework integration, and added `reportlab` for PDF diagram generation.
- **Framework-native scoring path**: Added `score_framework_observed_run` and observe-mode scoring helpers to score framework-observed trajectories without requiring the task runner path.
- **Rich framework observation capture for CrewAI**: Reworked CrewAI observation to capture event-bus data, normalized events, agents, workflow nodes, handoffs, model events, errors, and framework-specific output via `framework_extra` and `observed_run_crewai.json`.
- **Tool expectations and multi-agent contract support**: Added `tool_expectations`, `required_agents`, `required_handoffs`, and `required_agent_tool_expectations` to task contracts and validation logic for framework-native evaluation.
- **Expanded trajectory metadata**: Added per-tool-call metadata such as `agent_name`, `node_name`, `workflow_run_id`, `branch_id`, `call_id`, and `parent_event_id` to support richer observed traces and multi-agent analysis.
- **Duplicate artifact ID detection in file corpus loading**: Added warnings when duplicate artifact IDs are found while indexing file-based artifacts.
- **Test coverage tooling**: Added `pytest-cov` to the development dependency group.

### Changed

- **README overhaul**: Rewrote the README to emphasize deterministic, framework-native evaluation, expanded observe/scoring workflows, added artifact filename behavior, compare workflow, observe diagram docs, LangGraph and MAF support, and updated quickstart examples to use `--agent-class` and observe-based scoring.
- **Observe output structure**: Changed observe draft output to generate `observed_tools.yaml` and `task_contracts/inferred_task.yaml`, and removed older draft artifact/output references like `tool_map.yaml` and observed artifact dumps.
- **Artifact naming behavior**: Changed observe flows so `--no-draft` and `--score` write timestamped filenames, while draft-generating observe runs keep stable filenames.
- **CrewAI evaluation flow**: Changed CrewAI docs and implementation to a framework-native observe, review, and score workflow instead of relying on the task runner or artifact-mode-first framing.
- **Config validation behavior**: Expanded config validation to support framework configs for `crewai`, `maf`, and `langgraph`, tool expectations, multi-agent requirements, and framework-native observe scoring behavior.
- **Decision field validation**: Broadened accepted `decision_field` values to include `should_escalate` in addition to prior defaults.
- **CLI behavior for framework configs**: Changed `task` command behavior to block framework agents and direct users to `observe --score` instead.
- **Logging and output formatting**: Updated score logging in the CLI to use `Overall - ...` formatting and simplified log message formatting.
- **Observed run schema**: Changed observed runs so `final_answer` can be non-dict data and added a `framework_extra` field for framework-specific normalized observation payloads.
- **Docs refresh**: Updated `docs/configuration.md`, `docs/reports.md`, and `docs/crewai.md` to reflect new observe artifacts, observe diagrams, framework-specific outputs, and framework-native scoring workflows.
- **Package exports**: Removed `DistractorGenerator` and `load_seed_artifacts` from `src/groundeval/__init__.py` exports.
- **Prompt construction**: Removed the interpolated f-string for the static `"Question type: TASK"` system prompt line in `providers.py`.
- **Formatting and lint config**: Updated Ruff configuration to include a `max-line-length` setting and adopted broader code formatting changes across several modules.

### Removed

- **Legacy CrewAI optional-dependency section layout**: Removed the old `[project.optional-dependencies]` block and moved framework dependency groups under the dependency-group layout in `pyproject.toml`.
- **Legacy framework adapter package exports**: Removed the contents of `src/groundeval/framework_adapters/__init__.py`, including the old `build_crewai_agent_fn` export.
- **Obsolete README and docs references**: Removed references to gated enforcement from the framework support table, removed older observe artifact layouts like `tool_map.yaml` and draft artifact snapshots, and removed outdated task-based CrewAI scoring instructions.
- **Redundant and older comments/docstrings in core modules**: Removed many module-level and class/function docstrings and explanatory comments across `config_schema.py`, `core.py`, `task_eval.py`, and related modules as part of the refactor.

---

## [v0.02]

### Changed

- **`crew_class` renamed to `agent_class`**: Renamed the configuration field, function parameters, and internal variables from `crew_class` to `agent_class` across the entire codebase for consistency with the generalized agent interface. This affects:
  - `docs/crewai.md`: Updated field name in config example and doc table 
  - `config_schema.py`: Updated validation logic and error messages to reference `agent_class` instead of `crew_class` 
  - `crewai_adapter.py`: Renamed parameters in `_load_crew`, `build_crewai_agent_fn`, and `observe_crew` functions 
  - `run.py`: Updated keyword argument passed to `build_crewai_agent_fn` from `crew_class_path` to `agent_class_path` 
  - `tests/test_config_schema.py`: Updated test function names and assertions to reflect the renamed field 

---

## [v0.01]

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