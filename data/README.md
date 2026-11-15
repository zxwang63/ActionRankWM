# Released evaluation records

All files are gzip-compressed JSON Lines (one object per line). Read them with
`actionrankwm.io.iter_jsonl(path)`, which accepts `.jsonl` and `.jsonl.gz`, or with
`gzip.open(path, "rt")`. `MANIFEST.json` lists the row count and the SHA-256
of each uncompressed file.

These are the records from which `scripts/reproduce_paper_tables.py`
recomputes evaluation metrics from the included panels. The records were produced with package
protocol version 0.9.2 (the `schema_version` in `configs/`). Running
`scripts/preflight_toolsandbox.py --split e4a|e4b` with ToolSandbox at the pinned
commit regenerates `tasks.jsonl` byte-for-byte and the same real branches.

## Layout

| Path | Rows | Content |
|---|---:|---|
| `e4b/tasks.jsonl.gz` | 120 | Matched development panel: 6 mechanisms x 20 latent worlds (indices 10-29), 5 fixed candidates per task |
| `e4b/real_branches.jsonl.gz` | 600 | Real execution of every candidate from the cloned initial state, with continuation outcome and Eq. 1 value |
| `e4b/<model>/scored_candidates.jsonl.gz` | 600 | Per-candidate real vs. predicted value and fidelity scores; models: `qwen3.5-4b`, `qwen3.5-9b`, `llama3.1-8b`, `gemma4-12b` |
| `e4b/qwen3.5-{4b,9b}/predictions.jsonl.gz` | 600 | Raw world-model outputs (greedy decoding) and their parsed observations |
| `e4a/tasks.jsonl.gz`, `e4a/<model>/scored_candidates.jsonl.gz` | 12 / 60 | Development pilot (worlds 0-1); models: `qwen3.5-4b`, `qwen3.5-9b`, `gpt-5.6-sol` |
| `e4c_mr_sft/tasks.jsonl.gz` | 300 | Held-out panel (worlds 100-149) |
| `e4c_mr_sft/<model>/<condition>/scored_candidates.jsonl.gz` | 1,500 | `base` and MR-SFT adapters `seed20260720`, `seed20260721`, `seed20260722` for `qwen3.5-4b`, `qwen3.5-9b`, `llama3.1-8b` |
| `mr_sft_training/training_records.jsonl.gz` | 360 | MR-SFT training records (training worlds 30-49): prompt messages, positive real-transition response, sample weight; the two `*_negative_response` fields are used only by the DCCP baseline |

## Main fields

`tasks`: `task_id`, `domain`, `instruction`, `initial_state`, `target`,
`constraints`, `oracle_plan`, `candidates` (each with `action_id` c0-c4, `tool`, `arguments`,
`description`, `role`, `risk`), and `metadata` (`mechanism`, `template_id`,
`world_id`, `critical_fields`, `tool_docs`). Roles and risks are used only
for analysis and are never shown to the world model.

`real_branches`: `task_id`, `action`, `real_observation`, `real_state_patch`,
`candidate_ok`, `executed_steps`, `terminal_success`, `violations`, `real_value`.

`scored_candidates`: `task_id`, `action_id`, `action_role`, `action_risk`,
`real_value`, `simulated_value` (the same continuation and utility applied to
the predicted successor; `-1.01` marks a fail-closed parse/serialization/replay
failure), `token_f1`, `structured_f1`, `critical_field_accuracy`, `parse_ok`,
`patch_repair`, `prediction_error`, `predicted_success`, `real_terminal_success`,
`real_violations`.

`predictions`: `task_id`, `action_id`, `model` (served name), `raw_text`,
`parsed_observation`, `parse_ok`, `error`, token `usage`.

Task-level decisions (top sets, selected action, regret, HO-HR flags) are not
stored; `actionrankwm.metrics.decision.score_task_decisions` recomputes them from
`scored_candidates` and `tasks` with regret threshold 0.20 and token-F1 threshold 0.75.

## Provenance and terms

All task content is authored by us; names, phone numbers, coordinates and
identifiers are synthetic. Real branches were executed with Apple ToolSandbox
(commit `165848b9a78cead7ca7fe7c89c688b58e6501219`); predictions come from the
models listed in `THIRD_PARTY_NOTICES.md`. The records are released under
CC BY 4.0 (`data/LICENSE`); upstream ToolSandbox and model terms also apply.
