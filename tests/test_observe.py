from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from groundeval.observe import (
    ObservedRun,
    ObservedToolCall,
    RecordingRuntime,
    DraftGenerator,
    observe_crew,
    write_draft_output,
    _tool_name_to_verb,
    _extract_return_schema_from_tool,
    _parse_observed_answer,
)


def test_tool_name_to_verb_search_keywords():
    assert _tool_name_to_verb("search_tickets") == "search"
    assert _tool_name_to_verb("query_logs") == "search"
    assert _tool_name_to_verb("find_customer") == "search"
    assert _tool_name_to_verb("list_projects") == "search"
    assert _tool_name_to_verb("discover_endpoints") == "search"


def test_tool_name_to_verb_fetch_keywords():
    assert _tool_name_to_verb("fetch_customer") == "fetch"
    assert _tool_name_to_verb("get_ticket") == "fetch"
    assert _tool_name_to_verb("retrieve_logs") == "fetch"
    assert _tool_name_to_verb("read_document") == "fetch"
    assert _tool_name_to_verb("lookup_user") == "fetch"


def test_tool_name_to_verb_defaults_to_fetch():
    assert _tool_name_to_verb("process_payment") == "fetch"
    assert _tool_name_to_verb("update_record") == "fetch"
    assert _tool_name_to_verb("delete_orphans") == "fetch"


def test_tool_name_to_verb_search_before_fetch():
    assert _tool_name_to_verb("search_and_fetch") == "search"


def test_tool_name_to_verb_case_insensitive():
    assert _tool_name_to_verb("SEARCH_LOGS") == "search"
    assert _tool_name_to_verb("FetchCustomer") == "fetch"


def test_observed_tool_call_dataclass():
    tc = ObservedToolCall(
        tool_name="fetch_customer",
        arguments={"customer_id": "42"},
        return_value={"name": "Acme", "plan": "enterprise"},
        latency_ms=150.0,
    )
    d = tc.__dict__
    assert d["tool_name"] == "fetch_customer"
    assert d["arguments"]["customer_id"] == "42"
    assert d["return_value"]["plan"] == "enterprise"
    assert d["latency_ms"] == 150.0


def test_observed_run_to_dict_and_from_dict_roundtrip():
    run = ObservedRun(
        run_id="run_001",
        framework="crewai",
        agent_class="my.crew.Class",
        tool_calls=[
            ObservedToolCall("fetch_customer", {"id": "1"}, {"name": "Acme"}, 100.0),
            ObservedToolCall(
                "search_tickets", {"query": "bug"}, {"results": []}, 200.0
            ),
        ],
        final_answer={"should_act": True, "reasoning": "ok"},
        total_latency_ms=3000.0,
    )
    serialized = run.to_dict()
    restored = ObservedRun.from_dict(serialized)
    assert restored.run_id == run.run_id
    assert restored.framework == run.framework
    assert len(restored.tool_calls) == 2
    assert restored.tool_calls[0].tool_name == "fetch_customer"
    assert restored.final_answer["should_act"] is True


def test_observed_run_from_dict_empty_calls():
    run = ObservedRun.from_dict({
        "run_id": "r",
        "framework": "crewai",
        "agent_class": "x.y",
        "tool_calls": [],
        "final_answer": {},
        "total_latency_ms": 0,
    })
    assert run.tool_calls == []


def test_observed_run_from_dict_missing_final_answer():
    run = ObservedRun.from_dict({
        "run_id": "r",
        "framework": "crewai",
        "agent_class": "x.y",
        "tool_calls": [],
        "total_latency_ms": 0,
    })
    assert run.final_answer == {}


def test_recording_runtime_records_calls():
    rr = RecordingRuntime()
    rr.record("fetch_ticket", {"id": 1}, {"status": "open"}, 12.0)
    rr.record("search_logs", {"query": "x"}, ["a", "b"], 45.0)
    log = rr.call_log
    assert len(log) == 2
    assert log[0].tool_name == "fetch_ticket"
    assert log[1].latency_ms == 45.0


def test_recording_runtime_call_log_is_copy():
    rr = RecordingRuntime()
    rr.record("t", {}, {}, 1.0)
    log1 = rr.call_log
    log2 = rr.call_log
    assert log1 is not log2
    assert log1 == log2


def test_parse_observed_answer_dict():
    class FakeResult:
        raw = {"key": "value"}

    result = _parse_observed_answer(FakeResult())
    assert result == {"key": "value"}


def test_parse_observed_answer_json_string():
    class FakeResult:
        raw = '{"key": "value"}'

    result = _parse_observed_answer(FakeResult())
    assert result == {"key": "value"}


def test_parse_observed_answer_malformed_json_with_repair():
    class FakeResult:
        raw = '{"key": "value"'

    with patch("groundeval.observe._json_repair") as mock_repair:
        mock_repair.repair_json.return_value = '{"key": "repaired"}'
        result = _parse_observed_answer(FakeResult())
        assert result == {"key": "repaired"}


def test_parse_observed_answer_repair_string():
    class FakeResult:
        raw = "not json at all"

    with patch("groundeval.observe._json_repair") as mock_repair:
        mock_repair.repair_json.return_value = '{"repaired": true}'
        result = _parse_observed_answer(FakeResult())
        assert result == {"repaired": True}


def test_parse_observed_answer_pydantic():
    mock_pydantic = MagicMock()
    mock_pydantic.model_dump.return_value = {"decisions": ["a", "b"]}

    class FakeResult:
        raw = ""
        pydantic = mock_pydantic

    result = _parse_observed_answer(FakeResult())
    assert result == {"decisions": ["a", "b"]}


def test_parse_observed_answer_pydantic_uses_dict():
    mock_pydantic = MagicMock()
    del mock_pydantic.model_dump
    mock_pydantic.dict.return_value = {"old_style": True}

    class FakeResult:
        raw = ""
        pydantic = mock_pydantic

    result = _parse_observed_answer(FakeResult())
    assert result == {"old_style": True}


def test_parse_observed_answer_pydantic_fallback():
    class RealPydantic:
        def __iter__(self):
            return iter([("k", "v")])

    class FakeResult:
        raw = ""
        pydantic = RealPydantic()

    result = _parse_observed_answer(FakeResult())
    assert "k" in result or "raw_output" in result


def test_parse_observed_answer_not_pydantic_raw_none():
    class FakeResult:
        raw = None

    result = _parse_observed_answer(FakeResult())
    assert "raw_output" in result


def test_parse_observed_answer_truncates_long_raw():
    class FakeResult:
        raw = "x" * 2000

    result = _parse_observed_answer(FakeResult())
    assert len(result["raw_output"]) <= 1000


def test_extract_return_schema_from_pydantic_tool():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    mock_model = MagicMock()
    mock_model.model_json_schema.return_value = schema

    def fake_run(self, **kwargs):
        return mock_model

    fake_run.__annotations__ = {"return": mock_model}

    tool = MagicMock()
    tool._run = fake_run

    result = _extract_return_schema_from_tool(tool)
    assert result == schema


def test_extract_return_schema_from_args_schema():
    schema = {"type": "object", "properties": {"id": {"type": "string"}}}

    mock_schema = MagicMock()
    mock_schema.model_json_schema.return_value = schema

    tool = MagicMock()
    tool.args_schema = mock_schema
    del tool._run

    result = _extract_return_schema_from_tool(tool)
    assert result == schema


def test_extract_return_schema_no_schema():
    tool = MagicMock()
    del tool._run
    del tool.args_schema

    result = _extract_return_schema_from_tool(tool)
    assert result is None


def test_extract_return_schema_from_signature_return():
    schema = {"type": "object"}

    mock_model = MagicMock()
    mock_model.model_json_schema.return_value = schema

    import inspect

    tool = MagicMock()
    del tool.args_schema

    def fake_run(**kwargs):
        return "x"

    sig = inspect.signature(fake_run)
    fake_run.__signature__ = sig.replace(return_annotation=mock_model)
    tool._run = fake_run

    result = _extract_return_schema_from_tool(tool)
    assert result == schema


def test_observe_crew_with_mock():
    class RealTool:
        name = "fetch_customer"

        def _run(self, **kwargs):
            return {"customer": "Acme", "plan": "enterprise"}

    class RealAgent:
        tools = [RealTool()]

    class RealCrew:
        agents = [RealAgent()]

        def kickoff(self):
            for agent in self.agents:
                for tool in agent.tools:
                    tool._run()
            result = MagicMock()
            result.raw = '{"preconditions_verified":[], "should_act":true}'
            return result

    crew = RealCrew()

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=crew,
    ):
        observed = observe_crew("my.crew.Class", max_steps=5)

    assert observed.framework == "crewai"
    assert observed.agent_class == "my.crew.Class"
    assert len(observed.tool_calls) == 1
    assert observed.tool_calls[0].tool_name == "fetch_customer"
    assert observed.tool_calls[0].return_value == {
        "customer": "Acme",
        "plan": "enterprise",
    }
    assert observed.final_answer["should_act"] is True


def test_observe_crew_with_custom_tool_map():
    class RealTool:
        name = "my_custom_tool"

        def _run(self, **kwargs):
            return {"result": "ok"}

    class RealAgent:
        tools = [RealTool()]

    class RealCrew:
        agents = [RealAgent()]

        def kickoff(self):
            for agent in self.agents:
                for tool in agent.tools:
                    tool._run()
            result = MagicMock()
            result.raw = "{}"
            return result

    crew = RealCrew()

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=crew,
    ):
        observed = observe_crew("c", tool_map={"my_custom_tool": "search"})

    assert len(observed.tool_calls) == 1


def test_observe_crew_undeepcopyable_tool_is_passed_through():
    mock_crew = MagicMock()
    mock_agent = MagicMock()

    class UnDeepCopyable:
        name = "module_level_thing"

        def _run(self, **kwargs):
            return {"ok": True}

        def __deepcopy__(self, memo):
            raise TypeError("cannot deepcopy")

    tool = UnDeepCopyable()
    mock_agent.tools = [tool]
    mock_crew.agents = [mock_agent]

    result_obj = MagicMock()
    result_obj.raw = "{}"
    mock_crew.kickoff.return_value = result_obj

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=mock_crew,
    ):
        with patch(
            "copy.deepcopy",
            return_value=mock_crew,
        ):
            observed = observe_crew("x")

    assert observed.tool_calls == []


def test_observe_crew_no_tool_run_method():
    mock_crew = MagicMock()
    mock_agent = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "weird_tool"
    mock_agent.tools = [mock_tool]
    mock_crew.agents = [mock_agent]

    result_obj = MagicMock()
    result_obj.raw = "{}"
    mock_crew.kickoff.return_value = result_obj

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=mock_crew,
    ):
        observed = observe_crew("x")

    assert observed.tool_calls == []


def test_observe_crew_respects_max_iter():
    mock_crew = MagicMock()
    mock_crew.max_iter = None
    mock_agent = MagicMock()
    mock_agent.tools = []
    mock_crew.agents = [mock_agent]

    result_obj = MagicMock()
    result_obj.raw = "{}"
    mock_crew.kickoff.return_value = result_obj

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=mock_crew,
    ):
        with patch(
            "copy.deepcopy",
            return_value=mock_crew,
        ):
            observe_crew("x", max_steps=3)

    assert mock_crew.max_iter == 3


def test_observe_crew_max_iter_set_fails_silently():
    class ReadOnlyCrew:
        agents = []

        @property
        def max_iter(self):
            return 10

        def kickoff(self):
            result = MagicMock()
            result.raw = "{}"
            return result

    crew = ReadOnlyCrew()

    with (
        patch(
            "groundeval.framework_adapters.crewai_adapter._load_crew",
            return_value=crew,
        ),
        patch("copy.deepcopy", return_value=crew),
    ):
        observed = observe_crew("x.y", max_steps=5)

    assert observed.total_latency_ms >= 0


def test_observe_crew_uses_pydantic_output():
    mock_crew = MagicMock()
    mock_agent = MagicMock()
    mock_agent.tools = []
    mock_crew.agents = [mock_agent]

    mock_model = MagicMock()
    mock_model.model_dump.return_value = {
        "preconditions_verified": [],
        "should_act": False,
    }

    result_obj = MagicMock()
    result_obj.raw = ""
    result_obj.pydantic = mock_model
    mock_crew.kickoff.return_value = result_obj

    with patch(
        "groundeval.framework_adapters.crewai_adapter._load_crew",
        return_value=mock_crew,
    ):
        observed = observe_crew("x")

    assert observed.final_answer == {"preconditions_verified": [], "should_act": False}


def test_draft_generator_standard_mode_structured_answer():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="my.crew.Class",
        tool_calls=[
            ObservedToolCall(
                "fetch_customer",
                {"customer_id": "42"},
                {"plan_tier": "enterprise"},
                10.0,
            ),
            ObservedToolCall("search_tickets", {"query": "bug"}, {"results": []}, 15.0),
        ],
        final_answer={
            "preconditions_verified": [
                {
                    "check": "customer_is_enterprise",
                    "passed": True,
                    "facts_found": {"plan_tier": "enterprise"},
                    "evidence_artifacts": ["crm_account"],
                }
            ],
            "should_act": True,
            "reasoning": "Customer is enterprise, escalating.",
        },
        total_latency_ms=100,
    )

    gen = DraftGenerator(run, mode="standard")
    config = gen.generate()

    assert config["groundeval"]["config_status"] == "draft"
    assert config["groundeval"]["generated_from_observation"] is True
    assert config["groundeval"]["reviewed"] is False
    assert config["agent"]["framework"] == "crewai"
    assert config["agent"]["agent_class"] == "my.crew.Class"
    assert "fetch_customer" in config["agent"]["tool_map"]
    assert "search_tickets" in config["agent"]["tool_map"]

    tc = config["task_contracts"][0]
    assert tc["decision_field"] == "should_act"
    assert len(tc["preconditions"]) == 1
    assert tc["preconditions"][0]["check"] == "customer_is_enterprise"
    assert tc["preconditions"][0]["required_facts"] == ["plan_tier"]
    assert tc["preconditions"][0]["ground_truth_field"] == "crm_account.plan_tier"
    assert tc["preconditions"][0]["review_required"] is True
    assert "inferred_from" in tc["preconditions"][0]


def test_draft_generator_conservative_no_answer():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run, mode="conservative")
    config = gen.generate()
    assert config["task_contracts"][0]["preconditions"] == []


def test_draft_generator_standard_infers_from_tool_patterns():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall(
                "fetch_customer",
                {"id": "1"},
                {"plan_tier": "enterprise", "status": "active"},
                10.0,
            ),
            ObservedToolCall(
                "get_ticket", {"ticket_id": "2"}, {"title": "Bug report"}, 8.0
            ),
        ],
        final_answer={"should_act": True},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run, mode="standard")
    config = gen.generate()
    preconditions = config["task_contracts"][0]["preconditions"]
    assert len(preconditions) == 2
    assert any(pc["check"] == "customer_verified" for pc in preconditions)
    assert any(pc["check"] == "ticket_verified" for pc in preconditions)


def test_draft_generator_aggressive_includes_search_calls():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall("fetch_customer", {}, {"plan": "x"}, 1.0),
            ObservedToolCall("search_duplicates", {"query": "dup"}, {"count": 0}, 1.0),
        ],
        final_answer={"should_act": False},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run, mode="aggressive")
    config = gen.generate()
    preconditions = config["task_contracts"][0]["preconditions"]
    assert len(preconditions) == 2
    assert any("duplicates_performed" in pc["check"] for pc in preconditions)


def test_draft_generator_allowed_tools():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall(
                "fetch_customer",
                {"customer_id": "42"},
                {
                    "plan_tier": "enterprise",
                    "id": "x",
                    "_id": "y",
                    "subsystem": "crm",
                    "timestamp": "t",
                },
                10.0,
            ),
        ],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run, mode="standard")
    config = gen.generate()
    allowed = config["task_contracts"][0].get("allowed_tools", {})
    assert "fetch_customer" in allowed
    assert allowed["fetch_customer"]["entity_arg"] == "customer_id"
    assert allowed["fetch_customer"]["artifact_id"] == "customer_id"
    assert allowed["fetch_customer"]["returns"] == {"plan_tier": "enterprise"}


def test_draft_generator_allowed_tools_no_entity_arg():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall("process_report", {"data": "{}"}, {"result": "ok"}, 1.0),
        ],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run, mode="standard")
    config = gen.generate()
    allowed = config["task_contracts"][0].get("allowed_tools", {})
    assert "process_report" in allowed
    assert allowed["process_report"]["artifact_id"] == "process_report"


def test_draft_generator_allowed_tools_non_dict_return():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall("list_items", {}, ["a", "b", "c"], 1.0),
        ],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run, mode="standard")
    config = gen.generate()
    allowed = config["task_contracts"][0].get("allowed_tools", {})
    assert "returns" not in allowed["list_items"]


def test_draft_generator_deduplicates_tool_calls():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall("fetch_customer", {"id": "1"}, {"x": 1}, 1.0),
            ObservedToolCall("fetch_customer", {"id": "2"}, {"x": 2}, 2.0),
            ObservedToolCall("fetch_customer", {"id": "3"}, {"x": 3}, 3.0),
        ],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run)
    config = gen.generate()
    allowed = config["task_contracts"][0].get("allowed_tools", {})
    assert len(allowed) == 1


def test_draft_generator_empty_roles():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[ObservedToolCall("unknown_tool", {}, {}, 1.0)],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run)
    config = gen.generate()
    assert config["roles"] == {}


def test_draft_generator_roles_from_subsystem_field():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall("fetch_ticket", {}, {"subsystem": "jira", "id": "1"}, 1.0)
        ],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run)
    config = gen.generate()
    assert "agent" in config["roles"]
    assert "jira" in config["roles"]["agent"]["subsystems"]


def test_draft_generator_roles_from_tool_name():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall("zendesk_fetcher", {}, {"subsystem": "zendesk"}, 1.0)
        ],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run)
    config = gen.generate()
    assert "zendesk" in config["roles"]["agent"]["subsystems"]


def test_draft_generator_roles_multiple_subsystems():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall("fetch_ticket", {}, {"subsystem": "jira"}, 1.0),
            ObservedToolCall("fetch_customer", {}, {"subsystem": "crm"}, 1.0),
            ObservedToolCall("confluence_reader", {}, {"subsystem": "confluence"}, 1.0),
        ],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run)
    config = gen.generate()
    subsystems = config["roles"]["agent"]["subsystems"]
    assert "jira" in subsystems
    assert "crm" in subsystems
    assert "confluence" in subsystems


def test_draft_generator_decision_field_candidates():
    for candidate in ("should_act", "all_preconditions_pass", "should_escalate"):
        run = ObservedRun(
            run_id="r1",
            framework="x",
            agent_class="x",
            tool_calls=[],
            final_answer={candidate: True},
            total_latency_ms=0,
        )
        gen = DraftGenerator(run)
        config = gen.generate()
        assert config["task_contracts"][0]["decision_field"] == candidate


def test_draft_generator_decision_field_default():
    run = ObservedRun(
        run_id="r1",
        framework="x",
        agent_class="x",
        tool_calls=[],
        final_answer={"something_else": 42},
        total_latency_ms=0,
    )
    gen = DraftGenerator(run)
    config = gen.generate()
    assert config["task_contracts"][0]["decision_field"] == "should_act"


def test_generate_review_checklist_has_sections():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall(
                "fetch_customer", {"id": "1"}, {"plan": "enterprise"}, 1.0
            ),
        ],
        final_answer={
            "preconditions_verified": [
                {
                    "check": "customer_check",
                    "facts_found": {"plan": "enterprise"},
                    "evidence_artifacts": ["crm"],
                }
            ],
            "should_act": True,
        },
        total_latency_ms=0,
    )

    gen = DraftGenerator(run, mode="standard")
    text = gen.generate_review_checklist()

    assert "Before you can use" in text
    assert "Preconditions" in text
    assert "Allowed Tools" in text
    assert "After review" in text


def test_generate_review_checklist_empty_run():
    run = ObservedRun(
        run_id="r1",
        framework="x",
        agent_class="x",
        tool_calls=[],
        final_answer={},
        total_latency_ms=0,
    )
    gen = DraftGenerator(run, mode="conservative")
    text = gen.generate_review_checklist()
    assert "REVIEW.md" in text or text.strip() != ""


def test_generate_observe_report():
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[ObservedToolCall("t1", {"a": 1}, {"b": 2}, 10.0)],
        final_answer={"k": "v"},
        total_latency_ms=50,
    )

    gen = DraftGenerator(run)
    text = gen.generate_observe_report()
    assert "r1" in text
    assert "50ms" in text
    assert "t1" in text
    assert "k" in text


def test_generate_observe_report_truncates_long_return():
    run = ObservedRun(
        run_id="r1",
        framework="x",
        agent_class="x",
        tool_calls=[ObservedToolCall("t", {}, {"data": "x" * 2000}, 1.0)],
        final_answer={},
        total_latency_ms=0,
    )

    gen = DraftGenerator(run)
    text = gen.generate_observe_report()
    assert "..." in text


def test_write_draft_output_full(isolate_filesystem):
    tmp = str(isolate_filesystem)
    run = ObservedRun(
        run_id="r1",
        framework="crewai",
        agent_class="x",
        tool_calls=[
            ObservedToolCall(
                "fetch_customer",
                {"id": "1"},
                {"plan": "enterprise", "status": "active"},
                10.0,
            ),
        ],
        final_answer={"preconditions_verified": [], "should_act": True},
        total_latency_ms=100,
    )

    gen = DraftGenerator(run, mode="standard")
    result_dir = write_draft_output(tmp, run, gen)

    assert (result_dir / "observed_run.json").exists()
    assert (result_dir / "observe_report.md").exists()
    assert (result_dir / "draft_config" / "config.yaml").exists()
    assert (result_dir / "draft_config" / "tool_map.yaml").exists()
    artifacts_dir = result_dir / "draft_config" / "artifacts" / "observed"
    artifact_files = list(artifacts_dir.glob("*.json"))
    assert len(artifact_files) == 1
    assert artifact_files[0].name.startswith("001_fetch_customer")
    assert (result_dir / "draft_config" / "REVIEW.md").exists()

    with open(result_dir / "draft_config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["groundeval"]["config_status"] == "draft"

    with open(result_dir / "observed_run.json") as f:
        saved = json.load(f)
        assert saved["run_id"] == "r1"


def test_write_draft_output_no_return_value_skips_artifacts(isolate_filesystem):
    tmp = str(isolate_filesystem)
    run = ObservedRun(
        run_id="r1",
        framework="x",
        agent_class="x",
        tool_calls=[ObservedToolCall("search", {}, "just a string", 1.0)],
        final_answer={},
        total_latency_ms=0,
    )
    gen = DraftGenerator(run)
    result_dir = write_draft_output(tmp, run, gen)
    artifacts_dir = result_dir / "draft_config" / "artifacts" / "observed"
    assert not list(artifacts_dir.glob("*.json"))


def test_write_draft_output_creates_output_dir(isolate_filesystem):
    out = Path(isolate_filesystem) / "nested" / "output"
    run = ObservedRun("r", "x", "x", [], {}, 0)
    gen = DraftGenerator(run)
    result = write_draft_output(out, run, gen)
    assert result.exists()


def test_write_draft_output_overwrites_existing(isolate_filesystem):
    out = Path(isolate_filesystem)
    (out / "observed_run.json").write_text("stale")
    run = ObservedRun("r2", "crewai", "x", [], {}, 0)
    gen = DraftGenerator(run)
    write_draft_output(out, run, gen)
    with open(out / "observed_run.json") as f:
        data = json.load(f)
    assert data["run_id"] == "r2"


def test_draft_mode_keyword_preserved_in_config():
    run = ObservedRun("r1", "crewai", "x", [], {}, 0)
    gen = DraftGenerator(run, mode="aggressive")
    config = gen.generate()
    assert config["groundeval"]["draft_mode"] == "aggressive"


def test_inferred_from_present_on_all_preconditions():
    run = ObservedRun(
        "r1",
        "x",
        "x",
        [ObservedToolCall("fetch_customer", {}, {"x": 1}, 1.0)],
        {"should_act": True},
        0,
    )
    gen = DraftGenerator(run, mode="standard")
    config = gen.generate()
    for pc in config["task_contracts"][0]["preconditions"]:
        assert "inferred_from" in pc
        assert pc["inferred_from"]["run_id"] == "r1"


def test_inferred_from_on_allowed_tools():
    run = ObservedRun(
        "r1",
        "x",
        "x",
        [ObservedToolCall("fetch_ticket", {"ticket_id": "1"}, {"status": "open"}, 1.0)],
        {},
        0,
    )
    gen = DraftGenerator(run)
    config = gen.generate()
    for _, cfg in config["task_contracts"][0].get("allowed_tools", {}).items():
        assert "inferred_from" in cfg
        assert cfg["inferred_from"]["run_id"] == "r1"


def test_inferred_from_on_roles():
    run = ObservedRun(
        "r1", "x", "x", [ObservedToolCall("jira_fetcher", {}, {}, 1.0)], {}, 0
    )
    gen = DraftGenerator(run)
    config = gen.generate()
    for _, role_cfg in config["roles"].items():
        assert "inferred_from" in role_cfg
        assert role_cfg["inferred_from"]["run_id"] == "r1"


def test_fixture_flag_on_allowed_tools():
    run = ObservedRun(
        "r1",
        "x",
        "x",
        [ObservedToolCall("fetch_customer", {"id": "1"}, {"plan": "gold"}, 1.0)],
        {},
        0,
    )
    gen = DraftGenerator(run, mode="standard")
    config = gen.generate()
    gen2 = DraftGenerator(run, mode="conservative")
    config2 = gen2.generate()

    assert "allowed_tools" in config["task_contracts"][0]

    at_entry = config["task_contracts"][0]["allowed_tools"]["fetch_customer"]
    assert "returns" in at_entry
    at_entry2 = config2["task_contracts"][0]["allowed_tools"]["fetch_customer"]
    assert "returns" in at_entry2


def test_empty_tool_calls_empty_draft():
    run = ObservedRun("r1", "crewai", "x", [], {}, 0)
    gen = DraftGenerator(run)
    config = gen.generate()
    assert config["agent"]["tool_map"] == {}
    assert config["task_contracts"][0]["preconditions"] == []


def test_non_serializable_return_value_in_artifact(isolate_filesystem):
    run = ObservedRun(
        "r1",
        "x",
        "x",
        [ObservedToolCall("fetch_data", {}, {"safe": "string"}, 1.0)],
        {},
        0,
    )
    import math

    run.tool_calls.append(ObservedToolCall("complex_math", {}, {"fn": math.sqrt}, 1.0))

    gen = DraftGenerator(run)
    result_dir = write_draft_output(str(isolate_filesystem), run, gen)
    artifacts_dir = result_dir / "draft_config" / "artifacts" / "observed"
    assert len(list(artifacts_dir.glob("*.json"))) > 0


def test_very_large_observed_run_serialization():
    run = ObservedRun(
        "r_big",
        "crewai",
        "x",
        [
            ObservedToolCall(f"tool_{i}", {"arg": i}, {"val": i}, float(i))
            for i in range(100)
        ],
        {"should_act": True, "reasoning": "a" * 500},
        10000.0,
    )
    d = run.to_dict()
    restored = ObservedRun.from_dict(d)
    assert len(restored.tool_calls) == 100
    assert restored.tool_calls[99].tool_name == "tool_99"


def test_unicode_in_tool_names_and_values(isolate_filesystem):
    run = ObservedRun(
        "r_unicode",
        "crewai",
        "class_é",
        [
            ObservedToolCall(
                "récupérer_client", {"id": "1"}, {"nom": "Élise", "état": "actif"}, 10.0
            )
        ],
        {"décision": "approuvé"},
        0,
    )
    gen = DraftGenerator(run)
    result_dir = write_draft_output(str(isolate_filesystem), run, gen)

    config_path = result_dir / "draft_config" / "config.yaml"
    with open(config_path) as f:
        loaded = yaml.safe_load(f)
    assert "récupérer_client" in loaded["agent"]["tool_map"]
