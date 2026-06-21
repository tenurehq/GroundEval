import json, os
from pathlib import Path

os.makedirs("artifacts/slack", exist_ok=True)
events_out = []
with open("output.jsonl") as f:
    for line in f:
        d = json.loads(line)

        # normalize string-wrapped fields
        for key in ("actors", "artifact_ids", "facts"):
            val = d.get(key, {})
            if isinstance(val, str) and val:
                try:
                    d[key] = json.loads(val)
                except json.JSONDecodeError:
                    d[key] = {} if key == "facts" or key == "artifact_ids" else []
            elif not val:
                d[key] = {} if key in ("facts", "artifact_ids") else []

        if d.pop("category", None) == "artifact":
            # write artifact file for FileCorpusAdapter
            aid = d.get("doc_id")
            subsystem = d.get("doc_type", "misc")  # e.g. 'slack', 'jira'
            d["id"] = aid
            d["timestamp"] = d.get("timestamp", d.get("date"))
            d["subsystem"] = subsystem
            out = Path("artifacts") / subsystem / f"{aid}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(d, indent=2))
        else:
            # keep as event
            d["id"] = d.get("doc_id")
            d["type"] = d.get("doc_type")
            events_out.append(d)

Path("events.jsonl").write_text("\n".join(json.dumps(e) for e in events_out))
