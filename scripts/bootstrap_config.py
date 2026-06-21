import json
from pathlib import Path

EVENTS = Path("events.jsonl")
ARTIFACTS = Path("artifacts")
OUT = Path("config2.yaml")


def parse_actors(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return []


actors = set()
with open(EVENTS) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        actors.update(parse_actors(d.get("actors", [])))

# Infer subsystems from artifact directory names (e.g., artifacts/slack/)
subsystems = sorted({p.name for p in ARTIFACTS.iterdir() if p.is_dir()})
if not subsystems:
    subsystems = ["misc"]

actors_lines = "\n".join(f"  {a}: staff" for a in sorted(actors))

subsystems_lines = "\n".join(f"    - {s}" for s in subsystems)

yaml = f"""output_dir: ./eval_output

actors:
{actors_lines}

roles:
  staff:
    subsystems:
{subsystems_lines}

artifacts_dir: {ARTIFACTS}

# Derive visibility from event participation when possible
use_event_log_policy: true
"""

OUT.write_text(yaml)
print(f"Wrote {OUT}")
print(f"  Actors: {len(actors)}")
print(f"  Subsystems: {subsystems}")
