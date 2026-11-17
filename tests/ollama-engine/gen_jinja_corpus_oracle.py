#!/usr/bin/env python3
"""Renders the chat templates ollama ships as renderer test data (model/renderers/testdata, MIT repo) with REAL Python Jinja2 (the HF
chat-template environment: trim_blocks, lstrip_blocks, loopcontrols, tojson without ASCII escaping) over a grid of conversations, and
stores {template, vars, expected}. tests/test-ollama-jinja-corpus.cpp renders the same with the fork's engine and compares.
usage: gen_jinja_corpus_oracle.py <ollama-checkout> <out.json>   (run with the interpreter that has jinja2: /usr/bin/python3)"""
import json, sys, os, datetime
import jinja2, jinja2.ext as jext
from jinja2.sandbox import SandboxedEnvironment

root, out = sys.argv[1], sys.argv[2]
TPL = {
    "gemma4_e2b": "gemma4_e2b_chat_template.jinja2", "gemma4_31b": "gemma4_31b_chat_template.jinja2",
    "glimmer": "glimmer_chat_template.jinja",   # laguna_* use the HF-only {% generation %} tag: not plain Jinja, left out
    "nemotron35": "nemotron35_chat_template.jinja2", "nemotron3nano": "nemotron3nano_chat_template.jinja2", "qwen38": "qwen38_chat_template.jinja",
}
env = SandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jext.loopcontrols])
def raise_exception(m): raise jinja2.exceptions.TemplateError(m)
env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
env.globals["raise_exception"] = raise_exception
env.globals["strftime_now"] = lambda f: "2026-01-01"

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
         "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "add", "description": "Add numbers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]
U = lambda c: {"role": "user", "content": c}
A = lambda c, **k: dict({"role": "assistant", "content": c}, **k)
S = lambda c: {"role": "system", "content": c}
call = lambda n, a: {"type": "function", "function": {"name": n, "arguments": a}}
grid = {
    "user": dict(messages=[U("Hello")]),
    "system_user": dict(messages=[S("Be brief."), U("Hello")]),
    "multi_turn": dict(messages=[U("Hi"), A("Hello!"), U("How are you?")]),
    "system_multi": dict(messages=[S("Be brief."), U("Hi"), A("Hello!"), U("And now?")]),
    "tools_declared": dict(messages=[U("Weather in Paris?")], tools=tools),
    "tool_call_roundtrip": dict(messages=[U("Weather in Paris?"), A("", tool_calls=[call("get_weather", {"city": "Paris", "unit": "c"})]),
                                          {"role": "tool", "name": "get_weather", "content": "18C sunny"}], tools=tools),
    "tool_call_then_answer": dict(messages=[U("Weather in Paris?"), A("", tool_calls=[call("get_weather", {"city": "Paris"})]),
                                            {"role": "tool", "name": "get_weather", "content": "18C"}, A("It is 18C."), U("Thanks")], tools=tools),
    "parallel_calls": dict(messages=[U("2+3 and weather?"), A("", tool_calls=[call("add", {"a": 2, "b": 3}), call("get_weather", {"city": "Rome"})]),
                                     {"role": "tool", "name": "add", "content": "5"}, {"role": "tool", "name": "get_weather", "content": "20C"}], tools=tools),
    "reasoning_history": dict(messages=[U("Hi"), A("Hello!", reasoning_content="greeting back"), U("Again")]),
    "unicode": dict(messages=[U("héllo wörld ✓ 日本語"), A("ok")]),
    "multiline": dict(messages=[S("Line1\nLine2"), U("a\n\nb")]),
    # thinking / developer rows of ollama model/renderers/gemma4_reference_test.go's case table (row
    # S69-OLLAMA-PICK-GEMMA4-RENDERER-GOLDEN-VECTORS-2026-09-29-01): enable_thinking on and explicitly off, a developer turn,
    # and the combinations with tools and a tool round trip. gemma4, nemotron35 and qwen38 read enable_thinking; the
    # gemma4 templates also read the developer role.
    "thinking_user": dict(messages=[U("Hello")], enable_thinking=True),
    "thinking_system": dict(messages=[S("Be brief."), U("Hello")], enable_thinking=True),
    "thinking_tools": dict(messages=[U("Weather in Paris?")], tools=tools, enable_thinking=True),
    "thinking_system_tools": dict(messages=[S("Be brief."), U("Weather in Paris?")], tools=tools, enable_thinking=True),
    "thinking_explicitly_disabled": dict(messages=[U("Hello")], enable_thinking=False),
    "thinking_tool_roundtrip": dict(messages=[U("Weather in Paris?"), A("", reasoning_content="need the tool", tool_calls=[call("get_weather", {"city": "Paris"})]),
                                              {"role": "tool", "name": "get_weather", "content": "18C"}], tools=tools, enable_thinking=True),
    "developer_user": dict(messages=[{"role": "developer", "content": "Be brief."}, U("Hello")]),
    "developer_tools_thinking": dict(messages=[{"role": "developer", "content": "Be brief."}, U("Weather in Paris?")], tools=tools, enable_thinking=True),
}
cases, templates = [], {}
for tname, fn in TPL.items():
    src = open(os.path.join(root, "model/renderers/testdata", fn), encoding="utf-8").read()
    templates[tname] = src
    for gname, g in grid.items():
        for gen in (True, False):
            vars_ = dict(g, bos_token="<bos>", eos_token="<eos>", add_generation_prompt=gen)
            c = {"name": f"{tname}/{gname}/gen={int(gen)}", "template": tname, "vars": vars_}
            try:
                c["expected"] = env.from_string(src).render(**json.loads(json.dumps(vars_)))
            except Exception as e:  # the oracle itself refuses: the fork must refuse too (recorded, not compared by text)
                c["oracle_error"] = f"{type(e).__name__}: {e}"[:200]
            cases.append(c)
with open(out, "w", encoding="utf-8") as f:   # explicit: the corpus carries non-ASCII cases; Windows' default codec is cp1252
    json.dump({"templates": templates, "cases": cases}, f, ensure_ascii=False)
print(len(cases), "cases,", sum(1 for c in cases if "oracle_error" in c), "oracle errors,", os.path.getsize(out), "bytes")
