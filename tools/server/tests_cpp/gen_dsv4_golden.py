# PROGRESS-OK: 15 fixed in-memory conversations, finishes in well under a second
"""Generate golden prompt renderings for DeepSeek-V4-Flash-0731 from the OFFICIAL encoder.

Reference: xinference vendors the official DeepSeek tokenizer assets unmodified
(router/tokenizer_assets/deepseek_v4_flash_0731/encoding/encoding_dsv4.py, MIT, pinned by its SOURCE.json
upstream_revision f981a343464c25f82b901e5882716b3b2fa514de, sha256 abc0d261...0dde6). The encoder is pure Python,
no dependencies; this script only CALLS it and records its output, no code is copied.

Writes tools/server/tests_cpp/data/dsv4-golden.json: [{name, messages, tools, opts, official}] where `official` is the
rendering or "OFFICIAL-ERR ...".
Row: S69-XINFERENCE-PICK-DSV4-OFFICIAL-ENCODER-GOLDEN-PARITY-2026-09-29-01
"""
import importlib.util
import json
import os

ENC = os.environ.get("DSV4_ENCODER", "upstream/xinference/xinference/router/tokenizer_assets/"
                                     "deepseek_v4_flash_0731/encoding/encoding_dsv4.py")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dsv4-golden.json")

spec = importlib.util.spec_from_file_location("enc", ENC)
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)


def official(messages, tools=None, thinking=False, effort=None, drop_thinking=True):
    msgs = json.loads(json.dumps(messages))
    if tools:
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["tools"] = tools
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": tools})
    return enc.encode_messages(msgs, "thinking" if thinking else "chat", drop_thinking=drop_thinking, reasoning_effort=effort)


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]
SYS = {"role": "system", "content": "You are helpful."}
U1 = {"role": "user", "content": "Weather in Paris?"}
A_TC = {"role": "assistant", "content": "Let me check.", "reasoning_content": "need tool", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris", "days": 2})}}]}
A_TC2 = {"role": "assistant", "content": "", "reasoning_content": "two calls", "tool_calls": [
    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}},
    {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Rome"})}}]}
T1 = {"role": "tool", "tool_call_id": "c1", "content": "Sunny 20C"}
T2 = {"role": "tool", "tool_call_id": "c2", "content": "Rain 15C"}
A_END = {"role": "assistant", "content": "It is sunny.", "reasoning_content": "done"}


def tc(cid, name, args):
    return {"role": "assistant", "content": "", "reasoning_content": "r",
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]}


# (name, messages, tools, opts). The invalid "developer directly before assistant" conversation of the first
# differential is left out on purpose: the official encoder and the template disagree there only because the
# conversation is not valid.
CASES = [
    ("plain chat", [SYS, U1], None, {}),
    ("plain thinking", [SYS, U1], None, dict(thinking=True)),
    ("no system, thinking high", [U1], None, dict(thinking=True, effort="high")),
    ("system, thinking max", [SYS, U1], None, dict(thinking=True, effort="max")),
    ("chat effort high (ignored)", [SYS, U1], None, dict(effort="high")),
    ("tools chat", [SYS, U1], TOOLS, {}),
    ("tools thinking", [SYS, U1], TOOLS, dict(thinking=True)),
    ("tool round trip chat", [SYS, U1, A_TC, T1], TOOLS, {}),
    ("tool round trip thinking", [SYS, U1, A_TC, T1], TOOLS, dict(thinking=True)),
    ("parallel calls in order", [SYS, U1, A_TC2, T1, T2], TOOLS, dict(thinking=True)),
    ("parallel calls OUT OF ORDER results", [SYS, U1, A_TC2, T2, T1], TOOLS, dict(thinking=True)),
    ("multi-turn drop_thinking (no tools)", [SYS, U1, A_END, {"role": "user", "content": "thanks"}], None, dict(thinking=True)),
    ("assistant no-arg tool call, no tools in request", [SYS, U1, tc("c9", "ping", "{}"), {"role": "tool", "tool_call_id": "c9", "content": "pong"}], None, dict(thinking=True)),
    ("unicode + nonstring args, no tools in request", [SYS, U1, tc("c8", "f", json.dumps({"s": "caf\u00e9", "n": 3, "l": [1, "a"], "b": True})), {"role": "tool", "tool_call_id": "c8", "content": "ok"}], None, dict(thinking=True)),
    ("malformed args string", [SYS, U1, tc("c7", "f", "{\"a\": "), {"role": "tool", "tool_call_id": "c7", "content": "ok"}], None, dict(thinking=True)),
]

rows = []
for name, messages, tools, opts in CASES:
    try:
        o = official(messages, tools, **opts)
    except Exception as exc:  # recorded, not hidden: an official error is itself a golden fact
        o = "OFFICIAL-ERR %r" % exc
    rows.append({"name": name, "messages": messages, "tools": tools, "opts": opts, "official": o})

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, ensure_ascii=False, indent=1)
print("wrote", os.path.abspath(OUT), len(rows), "cases;",
      sum(1 for r in rows if r["official"].startswith("OFFICIAL-ERR")), "official errors")
