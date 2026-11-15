from __future__ import annotations

from actionrankwm.io import canonical_json
from actionrankwm.types import Action, JSONDict, TaskSpec


def _tool_semantics(domain: str) -> str:
    if domain == "sqlite_mcp":
        return """
Tools are deterministic SQLite-style operations:
- db.select returns matching rows and does not mutate state.
- db.update changes exactly the columns in `set` for rows matching `where`.
- db.delete removes rows matching `where`.
For db.update, the top-level observation MUST contain `affected_rows` and
`rows`, and state_patch MUST be exactly shaped like
{"domain":"sqlite_mcp","operation":"update","table":TABLE,"where":WHERE,
"rows":[COMPLETE_RESULTING_ROW,...]}. Every resulting row includes every
column, including unchanged columns. Never put `set` inside state_patch.
For db.select, include top-level `rows`; its patch operation is `select` and
contains the returned rows. For db.delete, include top-level `affected_rows`
and `deleted_rows`; its patch contains table, where, and deleted_rows.
""".strip()
    if domain == "terminal_fs":
        return """
Tools are deterministic workspace filesystem operations:
- fs.read returns the exact UTF-8 file content without mutation.
- fs.write writes the supplied complete content.
- fs.append appends the supplied text to the existing file.
- fs.replace replaces every exact occurrence of `old` with `new`.
- fs.delete removes the file.
- tests.run reports deterministic test results.
The state_patch domain is `terminal_fs`; file patches contain operation, path,
exists, and complete resulting content for writes/appends/replacements. For
fs.write, fs.append, or fs.replace, include `path` and the tool result field
(`bytes`, `bytes_appended`, or `replacements`) at top level, and include path,
exists=true, and the complete post-action file content inside state_patch. Use
patch operation `append` for fs.append. For fs.delete, state_patch has
exists=false and deleted_content. Do not put a full `files` mapping inside
state_patch; report only the one action transition. The strings `domain` and
`operation` are mandatory inside every state_patch. A write patch is shaped
exactly like {"domain":"terminal_fs","operation":"write","path":PATH,
"exists":true,"content":COMPLETE_RESULTING_CONTENT}.
""".strip()
    if domain == "toolsandbox":
        return """
The action calls one deterministic function from Apple's ToolSandbox. The user
message supplies the exact signature and semantics for every available tool.
Return top-level `result` (JSON value or null), `error_type` (exception class or
null), and `error` (message or null). On success `ok` is true; on any raised
tool error it is false.

The state_patch domain is `toolsandbox`. If no namespace changes, use exactly
{"domain":"toolsandbox","operation":"none","namespaces":{}}. Otherwise use
operation `replace_namespaces` and include only changed namespaces. Every
included namespace value is the complete post-action list of rows, including
all unchanged fields in those rows. Namespace names are setting, contact,
messaging, and reminder. Never invent a partial row delta.

For a successful action that creates an ID during this one transition, emit the
literal string `<generated_id_0>` both as the result and in the new row. The
benchmark clock is supplied in `benchmark_context`; use that exact timestamp
for generated creation_timestamp fields.
""".strip()
    raise ValueError(f"Unsupported domain: {domain}")


def build_messages(
    task: TaskSpec, action: Action, prompt_style: str = "direct"
) -> list[JSONDict]:
    system = f"""You are a language world model simulating one environment transition.
Predict the exact environment observation after the supplied action. Do not
choose a better action, solve the user's goal, or use knowledge of desired
outcomes. Simulate only the stated action from the complete current state.
If `current_state.environment_rules` is present, apply every matching rule
exactly. These explicit world rules override the default tool behavior and may
reject, normalize, or add a derived transition.

{_tool_semantics(task.domain)}

Return exactly one JSON object and no markdown. It must include boolean `ok`,
the domain-specific top-level tool-result fields described above, and a
canonical `state_patch`. Do not return an action request in place of the
resulting observation. Do not omit unchanged fields from a resulting row or
file.
"""
    if prompt_style == "rule_trace":
        direct_output_instruction = """Return exactly one JSON object and no markdown. It must include boolean `ok`,
the domain-specific top-level tool-result fields described above, and a
canonical `state_patch`. Do not return an action request in place of the
resulting observation. Do not omit unchanged fields from a resulting row or
file.
"""
        trace_output_instruction = """Return exactly one JSON object and no markdown, using this structured
rule-execution wrapper:
{"direct_effect":{},"rule_trace":[{"rule_index":0,"matched":true,
"derived_effect":{}}],"observation":{}}.

`direct_effect` records the default tool transition before environment rules.
`rule_trace` contains exactly one entry for every visible environment rule in
its original order, using zero-based `rule_index`. Set `matched` to whether the
rule applies to this action and put only that rule's rejection, normalization,
or derived transition in `derived_effect`; use an empty object when unmatched.
`observation` is the final exact environment observation after applying every
matching rule. The nested observation must include boolean `ok`, the required
domain-specific top-level tool-result fields, and a canonical `state_patch`.
Do not emit free-form reasoning or any keys outside this wrapper.
"""
        if direct_output_instruction not in system:
            raise AssertionError("Direct prompt output contract changed unexpectedly")
        system = system.replace(direct_output_instruction, trace_output_instruction)
    elif prompt_style != "direct":
        raise ValueError(f"Unsupported prompt style: {prompt_style}")
    user = {
        "domain": task.domain,
        "current_state": task.initial_state,
        "action": {"tool": action.tool, "arguments": action.arguments},
    }
    if task.domain == "toolsandbox":
        user["tool_docs"] = task.metadata.get("tool_docs", {})
        user["benchmark_context"] = {
            "current_timestamp": task.metadata.get("frozen_timestamp")
        }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": canonical_json(user)},
    ]
