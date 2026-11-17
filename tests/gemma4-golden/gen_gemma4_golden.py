# REWRITE-SAFE: writes only cases.json next to itself; reads the three template files named below and no data store.
"""Golden render vectors for the Gemma 4 chat templates, produced by REAL Jinja2 configured the way Hugging Face's apply_chat_template configures it.

Row S69-OLLAMA-PICK-GEMMA4-RENDERER-GOLDEN-VECTORS-2026-09-29-01.

Why Jinja2 and not Ollama's expected strings: Ollama's own tests treat the Jinja2 render of the shipped template as the oracle and its Go renderer as the
thing under test; they are only trustworthy where its VERIFY_JINJA2 path passes. The fork has no Go renderer, it runs the template itself in its own C++ Jinja
engine, so the question here is "does our engine render the template exactly like Jinja2 does", and the oracle is Jinja2.

HF settings reproduced (transformers tokenization_utils_base / utils/chat_template_utils.py, behaviour only): ImmutableSandboxedEnvironment with
trim_blocks=True and lstrip_blocks=True, the loopcontrols extension, tojson = json.dumps(ensure_ascii=False), and raise_exception / strftime_now globals.

Templates:
  e2b        Ollama testdata/gemma4_e2b_chat_template.jinja2        (small models; MIT repo copy of Google's template)
  31b_new    Ollama testdata/gemma4_31b_chat_template.jinja2        (published 2026-07-09: "fixed tool-calling loops, turn closures, thinking content-ordering")
  31b_fork   models/templates/google-gemma-4-31B-it.jinja           (the older copy this fork ships)

Usage: pythonw gen_gemma4_golden.py   (writes cases.json)
"""
import json
import os
import sys

sys.path.insert(0, "build")
import jinja2  # noqa: E402
import jinja2.ext  # noqa: E402 (a submodule: "import jinja2" alone does not bind jinja2.ext)
import jinja2.exceptions  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = {
    "e2b": os.path.join(HERE, "gemma4_e2b_chat_template.jinja"),
    "31b_new": os.path.join(HERE, "gemma4_31b_chat_template_2026-07-09.jinja"),
    "31b_fork": os.path.join(HERE, "..", "..", "models", "templates", "google-gemma-4-31B-it.jinja"),
}


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def make_env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: "2026-01-01"
    return env


def tool(name, desc, props, required=None):
    return {"type": "function", "function": {"name": name, "description": desc,
                                              "parameters": {"type": "object", "properties": props, "required": required or []}}}


BASH = tool("bash", "Run a command", {"command": {"type": "string", "description": "The command"}}, ["command"])
READ = tool("read", "Read a file", {"path": {"type": "string", "description": "File path"}}, ["path"])
WEATHER = tool("get_weather", "Get weather", {"city": {"type": "string", "description": "City"}})
NESTED = tool("create_task", "Create a task", {
    "title": {"type": "string", "description": "Title", "enum": ["a", "b"]},
    "tags": {"type": "array", "description": "Tags", "items": {"type": "string"}},
    "opts": {"type": "object", "description": "Options", "properties": {"urgent": {"type": "boolean", "description": "Flag"}, "n": {"type": "number"}},
             "required": ["urgent"]},
}, ["title"])


def call(i, name, args):
    return {"id": "call_%d" % i, "type": "function", "function": {"name": name, "arguments": args}}


U = lambda t: {"role": "user", "content": t}          # noqa: E731
A = lambda t: {"role": "assistant", "content": t}     # noqa: E731
S = lambda t: {"role": "system", "content": t}        # noqa: E731

CASES = [
    # name, messages, tools, enable_thinking, extra kwargs
    ("user_only", [U("Hello")], None, False, {}),
    ("user_only_thinking", [U("Hello")], None, True, {}),
    ("system_user", [S("You are terse."), U("Hello")], None, False, {}),
    ("developer_user", [{"role": "developer", "content": "Follow the rules."}, U("Hello")], None, False, {}),
    ("system_content_parts", [{"role": "system", "content": [{"type": "text", "text": "A "}, {"type": "text", "text": "B"}]}, U("Hi")], None, False, {}),
    ("tools_no_system", [U("List files")], [BASH], False, {}),
    ("system_tools", [S("Be careful."), U("Read it")], [BASH, READ], False, {}),
    ("thinking_no_system", [U("Think")], None, True, {}),
    ("thinking_system", [S("Sys"), U("Think")], None, True, {}),
    ("thinking_tools", [U("Think")], [WEATHER], True, {}),
    ("thinking_system_tools", [S("Sys"), U("Think")], [WEATHER], True, {}),
    ("nested_tool_declaration", [U("Make one")], [NESTED], False, {}),
    ("multi_turn", [U("Hi"), A("Hello!"), U("Again")], None, False, {}),
    ("multi_turn_thinking", [U("Hi"), A("Hello!"), U("Again")], None, True, {}),
    ("assistant_reasoning_after_last_user", [U("Q"), {"role": "assistant", "content": "A", "reasoning_content": "because"}], None, False, {"add_generation_prompt": False}),
    ("assistant_reasoning_before_last_user", [U("Q1"), {"role": "assistant", "content": "A1", "reasoning_content": "because"}, U("Q2")], None, False, {}),
    ("assistant_content_with_thought_markup", [U("Q"), A("<|channel>thought\nsecret<channel|>The answer"), U("Next")], None, False, {}),
    ("tool_call_then_response", [U("Weather?"), {"role": "assistant", "content": "", "tool_calls": [call(1, "get_weather", {"city": "Paris"})]},
                                 {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}], [WEATHER], False, {}),
    ("tool_call_then_response_thinking", [U("Weather?"), {"role": "assistant", "content": "", "tool_calls": [call(1, "get_weather", {"city": "Paris"})]},
                                          {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}], [WEATHER], True, {}),
    ("tool_call_final_answer_next_user", [U("Weather?"), {"role": "assistant", "content": "", "tool_calls": [call(1, "get_weather", {"city": "Paris"})]},
                                          {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}, A("It is sunny."), U("Thanks")], [WEATHER], False, {}),
    ("parallel_tool_calls", [U("Two cities"), {"role": "assistant", "content": "", "tool_calls": [call(1, "get_weather", {"city": "Paris"}), call(2, "get_weather", {"city": "Rome"})]},
                             {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}, {"role": "tool", "tool_call_id": "call_2", "content": "Rainy"}], [WEATHER], False, {}),
    ("tool_call_no_arguments", [U("Go"), {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "ping", "arguments": None}}]}], [WEATHER], False, {}),
    ("tool_call_nested_arguments", [U("Go"), {"role": "assistant", "content": "", "tool_calls": [call(1, "create_task", {"title": "a", "tags": ["x", "y"], "opts": {"urgent": True, "n": 3}})]}], [NESTED], False, {}),
    ("tool_response_by_name_only", [U("Go"), {"role": "assistant", "content": "", "tool_calls": [call(1, "get_weather", {"city": "Paris"})]},
                                    {"role": "tool", "name": "get_weather", "content": "Warm"}], [WEATHER], False, {}),
    ("tool_response_content_parts", [U("Go"), {"role": "assistant", "content": "", "tool_calls": [call(1, "get_weather", {"city": "Paris"})]},
                                     {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "Hot"}, {"type": "text", "text": " and dry"}]}], [WEATHER], False, {}),
    ("preserve_thinking_with_tool_calls", [U("Go"), {"role": "assistant", "content": "", "reasoning_content": "plan", "tool_calls": [call(1, "get_weather", {"city": "Paris"})]},
                                           {"role": "tool", "tool_call_id": "call_1", "content": "ok"}, A("done"), U("More")], [WEATHER], True, {"preserve_thinking": True}),
    ("user_parts_image_text", [{"role": "user", "content": [{"type": "text", "text": "Look"}, {"type": "image_url", "image_url": {"url": "x"}}]}], None, False, {}),
    ("consecutive_assistant_messages", [U("Hi"), A("one"), A("two"), U("Go")], None, False, {}),
    ("unicode_and_quotes", [U("Say \"hi\" \u2014 \u4f60\u597d <b>&</b> 'x'")], None, False, {}),
    ("whitespace_trim", [U("  padded  \n"), A("\n answer \n"), U("next")], None, False, {}),
    ("no_generation_prompt", [U("Q"), A("A")], None, False, {"add_generation_prompt": False}),
]


def main():
    env = make_env()
    compiled = {}
    for key, path in TEMPLATES.items():
        with open(path, encoding="utf-8", newline="") as fh:
            compiled[key] = env.from_string(fh.read())
    cases = []
    for name, msgs, tools, think, extra in CASES:
        ctx = {"messages": msgs, "bos_token": "<bos>", "eos_token": "<eos>", "add_generation_prompt": extra.get("add_generation_prompt", True),
               "enable_thinking": think}
        if tools:
            ctx["tools"] = tools
        else:
            ctx["tools"] = None
        if "preserve_thinking" in extra:
            ctx["preserve_thinking"] = extra["preserve_thinking"]
        expected = {}
        for key, tpl in compiled.items():
            try:
                expected[key] = {"text": tpl.render(**ctx)}
            except Exception as e:  # a template may legitimately raise (raise_exception): the engine must raise too
                expected[key] = {"error": "%s: %s" % (type(e).__name__, e)}
        cases.append({"name": name, "messages": msgs, "tools": tools or [], "enable_thinking": think,
                      "add_generation_prompt": ctx["add_generation_prompt"], "preserve_thinking": extra.get("preserve_thinking"), "expected": expected})
    out = {"jinja2": jinja2.__version__, "templates": {k: os.path.relpath(os.path.abspath(v), HERE).replace("\\", "/") for k, v in TEMPLATES.items()}, "cases": cases}
    with open(os.path.join(HERE, "cases.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("wrote %d cases x %d templates, jinja2 %s" % (len(cases), len(compiled), jinja2.__version__))


if __name__ == "__main__":
    main()
