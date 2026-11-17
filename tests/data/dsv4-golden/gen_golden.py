# Generates golden vectors for the DeepSeek-V4-Flash-0731 chat template from the OFFICIAL encoder.
# Reference encoder: encoding/encoding_dsv4.py from the DeepSeek-V4-Flash-0731 release (MIT, (c) DeepSeek),
#   obtained via xinference tokenizer_assets (sha256 abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6,
#   upstream revision f981a343464c25f82b901e5882716b3b2fa514de). Only its OUTPUT is stored here.
# Usage: python gen_golden.py [path/to/encoding_dsv4.py]   -> writes golden.json next to this file.
# Row: S69-XINFERENCE-PICK-DSV4-OFFICIAL-ENCODER-GOLDEN-PARITY-2026-09-29-01
import sys, os, json, hashlib, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ENC = sys.argv[1] if len(sys.argv) > 1 else "upstream/xinference/xinference/router/tokenizer_assets/deepseek_v4_flash_0731/encoding/encoding_dsv4.py"
spec = importlib.util.spec_from_file_location("enc", ENC)
enc = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)

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
CASES = [
 ("plain chat", dict(messages=[SYS, U1]), {}),
 ("plain thinking", dict(messages=[SYS, U1]), dict(thinking=True)),
 ("no system, thinking high", dict(messages=[U1]), dict(thinking=True, effort="high")),
 ("system, thinking max", dict(messages=[SYS, U1]), dict(thinking=True, effort="max")),
 ("chat effort high (ignored)", dict(messages=[SYS, U1]), dict(effort="high")),
 ("tools chat", dict(messages=[SYS, U1], tools=TOOLS), {}),
 ("tools thinking", dict(messages=[SYS, U1], tools=TOOLS), dict(thinking=True)),
 ("tool round trip chat", dict(messages=[SYS, U1, A_TC, T1], tools=TOOLS), {}),
 ("tool round trip thinking", dict(messages=[SYS, U1, A_TC, T1], tools=TOOLS), dict(thinking=True)),
 ("parallel calls in order", dict(messages=[SYS, U1, A_TC2, T1, T2], tools=TOOLS), dict(thinking=True)),
 ("parallel calls OUT OF ORDER results", dict(messages=[SYS, U1, A_TC2, T2, T1], tools=TOOLS), dict(thinking=True)),
 ("multi-turn drop_thinking (no tools)", dict(messages=[SYS, U1, A_END, {"role": "user", "content": "thanks"}]), dict(thinking=True)),
 ("assistant no-arg tool call", dict(messages=[SYS, U1, {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [{"id": "c9", "type": "function", "function": {"name": "ping", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "c9", "content": "pong"}]), dict(thinking=True)),
 ("unicode + nonstring args", dict(messages=[SYS, U1, {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [{"id": "c8", "type": "function", "function": {"name": "f", "arguments": json.dumps({"s": "caf\u00e9", "n": 3, "l": [1, "a"], "b": True})}}]}, {"role": "tool", "tool_call_id": "c8", "content": "ok"}]), dict(thinking=True)),
 ("malformed args string", dict(messages=[SYS, U1, {"role": "assistant", "content": "", "reasoning_content": "r", "tool_calls": [{"id": "c7", "type": "function", "function": {"name": "f", "arguments": "{\"a\": "}}]}, {"role": "tool", "tool_call_id": "c7", "content": "ok"}]), dict(thinking=True)),
]
out = []
for name, spec_, opts in CASES:
    msgs = spec_["messages"]; tools = spec_.get("tools")
    try:
        exp = official(msgs, tools, **opts); err = None
    except Exception as e:
        exp = None; err = repr(e)
    out.append({"name": name, "messages": msgs, "tools": tools, "thinking": bool(opts.get("thinking")),
                "reasoning_effort": opts.get("effort"), "expected": exp, "official_error": err})
doc = {"encoder_sha256": hashlib.sha256(open(ENC, "rb").read()).hexdigest(), "cases": out}
with open(os.path.join(HERE, "golden.json"), "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
print("cases", len(out), "errors", sum(1 for c in out if c["official_error"]), "sha", doc["encoder_sha256"][:12])
