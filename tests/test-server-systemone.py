#!/usr/bin/env python3
"""End-to-end check of POST /v1/systemone (typed decision models, ik_llama.cpp c069e89e1 #2592) on a tiny random-weight model.

    python test-server-systemone.py <tree-root> <llama-server> <workdir>

Builds tests/make_tiny_llama_gguf.py's "bytes" model (every ASCII letter is one token), then two copies carrying decision
metadata (llama.decision.type = openjev / lev, a "systemone" chat template, an openjev choice temperature of 2.0), and checks:
  - a model without decision metadata answers NOT_SUPPORTED (501), a bad request 400;
  - openjev choice / score / noul answers have the right shape, probabilities sum to 1, choice = argmax;
  - the served numbers ARE the label logits at the last prompt token: the choice probabilities equal a softmax (at the
    model's temperature) over the logprobs /completion reports for the same prompt tokens (raw, post_sampling_probs=false);
  - lev asks a choice in two orders (2 tasks) and still answers; a repeat request gives the same choices and numbers
    within 1e-3 (prompt-cache reuse changes the batch shape, so bit equality is not the contract).
Exit 0 = every check passed. Run against a binary without the route and it fails (404). Merge-owed item 1 of
S69-ENGINE-MERGE-IK-LLAMA-MAIN-TRIAL-2026-09-29-01.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT, BIN, WORK = sys.argv[1], sys.argv[2], sys.argv[3]
NOWIN = 0x08000000 if os.name == "nt" else 0
PORT = 18990
fails = []

sys.path.insert(0, os.path.join(ROOT, "gguf-py"))
import gguf  # noqa: E402

TEMPLATE = "S: {{ state }}|Q: {{ instructions }}|{% for o in options %}{{ o.key }};{% endfor %}A:"


def render(state, instructions, keys):
    return f"S: {state}|Q: {instructions}|" + "".join(k + ";" for k in keys) + "A:"


def check(cond, msg):
    print(("  ok:   " if cond else "  FAIL: ") + msg, flush=True)
    if not cond:
        fails.append(msg)


def make_models():
    os.makedirs(WORK, exist_ok=True)
    base = os.path.join(WORK, "tiny-bytes.gguf")
    subprocess.run([sys.executable, os.path.join(ROOT, "tests", "make_tiny_llama_gguf.py"), ROOT, base, "2048", "bytes"],
                   check=True, creationflags=NOWIN)
    out = {"base": base}
    for kind in ("openjev", "lev"):
        dst = os.path.join(WORK, f"tiny-{kind}.gguf")
        r = gguf.GGUFReader(base, "r")
        w = gguf.GGUFWriter(dst, "llama")
        for f in r.fields.values():
            if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
                continue
            vt = f.types[0]
            st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(f.name, f.contents(), vt, sub_type=st)
        w.add_key_value("llama.decision.type", kind, gguf.GGUFValueType.STRING)
        w.add_key_value("tokenizer.chat_template.systemone", TEMPLATE, gguf.GGUFValueType.STRING)
        if kind == "openjev":
            w.add_key_value("llama.decision.temperature.choice", 2.0, gguf.GGUFValueType.FLOAT32)
        for t in r.tensors:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        for t in r.tensors:
            w.write_tensor_data(t.data, tensor_endianess=r.endianess)
        w.close()
        out[kind] = dst
    return out


class Server:
    def __init__(self, model):
        global PORT
        PORT += 1
        self.port = PORT
        self.log = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False, dir=WORK)
        cmd = [BIN, "-m", model, "--port", str(self.port), "--host", "127.0.0.1", "-c", "2048", "--threads", "2", "-np", "2", "--no-webui"]
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, creationflags=NOWIN)
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1).read()
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError("server died:\n" + self.text())
                time.sleep(0.2)
        raise RuntimeError("server did not start:\n" + self.text())

    def text(self):
        self.log.flush()
        return open(self.log.name, encoding="utf-8", errors="replace").read()

    def post(self, path, body, timeout=120):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            raw = e.read() or b"{}"
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"raw": raw.decode(errors="replace")}

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except Exception:
            self.proc.kill()


REQ = {
    "state": "printer on fire",
    "questions": {
        "route": {"type": "choice", "instructions": "who handles it", "criteria": {"it": "IT desk", "facilities": None, "security": "guards"}},
        "urgency": {"type": "score", "instructions": "how urgent", "criteria": ["none", "low", "high", "now"]},
        "fire": {"type": "noul", "instructions": "is there a fire"},
    },
}


def main():
    models = make_models()

    print("[base model: no decision metadata]")
    s = Server(models["base"])
    try:
        st, js = s.post("/v1/systemone", REQ)
        check(st == 501, f"non-decision model -> 501 NOT_SUPPORTED (got {st} {json.dumps(js)[:160]})")
    finally:
        s.stop()

    print("[openjev]")
    s = Server(models["openjev"])
    try:
        st, js = s.post("/v1/systemone", {"questions": REQ["questions"]})
        check(st == 400, f"missing state -> 400 (got {st})")

        st, js = s.post("/v1/systemone", REQ)
        check(st == 200, f"decision request -> 200 (got {st} {json.dumps(js)[:200]})")
        if st == 200:
            ans = js.get("answers", {})
            check(set(ans) == {"route", "urgency", "fire"}, f"one answer per question: {sorted(ans)}")
            r = ans.get("route", {})
            probs = r.get("probabilities", {})
            check(set(probs) == {"it", "facilities", "security"}, "choice probabilities keyed by option")
            check(abs(sum(probs.values()) - 1) < 1e-9, f"choice probabilities sum to 1 ({sum(probs.values())})")
            check(r.get("choice") == max(probs, key=probs.get), "choice = argmax")
            check(0 <= r.get("confidence", -1) <= 1, "choice confidence in [0, 1]")
            u = ans.get("urgency", {})
            check(0 <= u.get("score", -1) <= 3 and set(u.get("legend", {})) == {"0", "1", "2", "3"}, "score in [0, 3] with legend")
            n = ans.get("fire", {})
            check(0 <= n.get("noul", -1) <= 1, "noul in [0, 1]")
            check(js.get("usage", {}).get("input_tokens", 0) > 0 and js["usage"].get("output_tokens") == 0, f"usage {js.get('usage')}")

            # A repeat may reuse the slot's cached prompt prefix, so the last token is decoded in a different batch
            # shape and the logits move at float level (measured first run: exact equality failed). Same choices and
            # numbers within 1e-3 is the contract; a wrong label or a stale result moves them by far more.
            st2, js2 = s.post("/v1/systemone", REQ)
            ans2 = js2.get("answers", {}) if st2 == 200 else {}
            def nums(a):
                out = []
                for q in sorted(a):
                    v = a[q]
                    out += [v.get("choice"), v.get("type")]
                    out += [("p", k, round(p, 9)) for k, p in sorted(v.get("probabilities", {}).items())]
                    for f in ("score", "noul", "confidence"):
                        if f in v:
                            out.append((f, v[f]))
                return out
            same_keys = sorted(ans2) == sorted(ans) and all(ans2[q].get("choice") == ans[q].get("choice") for q in ans)
            max_d = 0.0
            if same_keys:
                for q in ans:
                    for k, p in ans[q].get("probabilities", {}).items():
                        max_d = max(max_d, abs(p - ans2[q].get("probabilities", {}).get(k, 1e9)))
                    for f in ("score", "noul", "confidence"):
                        if f in ans[q]:
                            max_d = max(max_d, abs(ans[q][f] - ans2[q].get(f, 1e9)))
            check(st2 == 200 and same_keys and max_d < 1e-3,
                  f"a repeated request gives the same answers (choices equal, max numeric diff {max_d:.2e} < 1e-3; "
                  f"bit-identical: {nums(ans2) == nums(ans)})")

            # the numbers must be the label logits at the last prompt token: compare with /completion's raw logprobs
            prompt = render(REQ["state"], "who handles it", ["it", "facilities", "security"])
            stt, tk = s.post("/tokenize", {"content": prompt, "add_special": False, "parse_special": True})
            ids = tk.get("tokens", [])
            lab = []
            for c in "ABC":
                _, t = s.post("/tokenize", {"content": c, "add_special": False})
                lab.append(t.get("tokens", [None])[0])
            stc, cj = s.post("/completion", {"prompt": ids, "n_predict": 1, "n_probs": 300, "post_sampling_probs": False,
                                              "temperature": 0.0, "cache_prompt": False})
            top = (cj.get("completion_probabilities") or [{}])[0].get("top_logprobs", [])
            lp = {e.get("id"): e.get("logprob") for e in top}
            if stt == 200 and stc == 200 and all(i in lp for i in lab):
                z = [lp[i] / 2.0 for i in lab]  # the model's choice temperature is 2.0
                m = max(z)
                e = [math.exp(x - m) for x in z]
                want = [x / sum(e) for x in e]
                got = [probs["it"], probs["facilities"], probs["security"]]
                err = max(abs(a - b) for a, b in zip(got, want))
                check(err < 1e-4, f"choice probabilities = softmax(label logit / 2) from /completion (max err {err:.2e}; got {got}, want {want})")
                # control: the same comparison at temperature 1 must NOT match (proves the temperature metadata was read)
                e1 = [math.exp(lp[i] - max(lp[j] for j in lab)) for i in lab]
                w1 = [x / sum(e1) for x in e1]
                check(max(abs(a - b) for a, b in zip(got, w1)) > 1e-4, "control: temperature 1 does not match")
            else:
                check(False, f"logit cross-check could not run (tokenize {stt}, completion {stc}, labels {lab}, n_top {len(top)})")
    finally:
        s.stop()

    print("[lev]")
    s = Server(models["lev"])
    try:
        st, js = s.post("/v1/systemone", REQ)
        check(st == 200, f"lev request -> 200 (got {st} {json.dumps(js)[:200]})")
        if st == 200:
            r = js["answers"].get("route", {})
            check(abs(sum(r.get("probabilities", {}).values()) - 1) < 1e-9, "lev choice probabilities sum to 1")
            check(0 <= js["answers"].get("fire", {}).get("noul", -1) <= 1, "lev noul (9-level rating) in [0, 1]")
            log = s.text()
            check("decision model type: lev" in log, "server logged the lev decision type")
    finally:
        s.stop()

    print(f"{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
