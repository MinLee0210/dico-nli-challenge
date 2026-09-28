# research/

Structured approach survey behind [`../RESEARCH.md`](../RESEARCH.md), produced
with the `/research` → `/research-deep` workflow on 2026-09-28.

| File | Content |
|---|---|
| `outline.yaml` | topic, the 20 surveyed items (11 models + 9 methods), execution config |
| `fields.yaml` | the 25 fields each item is described by (applicability, evidence, consistency impact, risks, priority, ...) |
| `results/*.json` | one file per item; values marked `[uncertain]` are listed in its `uncertain` array |

Validate a result file:

```bash
python3 ~/.claude/skills/research/validate_json.py -f docs/research/fields.yaml -j docs/research/results/<item>.json
```

Add items or fields with `/research-add-items` or `/research-add-fields`, and
re-run `/research-deep` (completed items are skipped).
