# REWRITE-SAFE: this test starts a server and POSTs requests; it writes only its own log files and reads no data store.
"""E2E served checks for the Kobold / ooba / SillyTavern / TabbyAPI sampler request names and the unknown-key warning.

Usage: pythonw e2e_sampler_aliases_served.py <llama-server.exe> <small.gguf> <logdir> [port]
       (the caller redirects stdout to a file; exit 0 = every check passed)

Row: S69-ENGREV-TABBYAPI-SAMPLER-ALIASES-AND-UNKNOWN-KEY-WARN-2026-10-27-01
Method: every request is sequential with cache_prompt false, seed fixed (row S69-FORK-GATE-TEXT-IDENTITY-TESTS-MUST-BE-SEQUENTIAL-CACHE-OFF-2026-11-08-01).
For each alias A of canonical C the check is  out(A) == out(C)  AND  out(C) != out(baseline)  where the second half proves the
setting is effective on this model; a binary that ignores A gives out(A) == out(baseline) != out(C) and FAILS (fail-before).
DEGRADED PROXY: a 0.5B model on CPU proves wiring (the alias reaches the same sampler value), not sampling quality.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

EXE, MODEL, LOGDIR = sys.argv[1], sys.argv[2], sys.argv[3]
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 18190
os.makedirs(LOGDIR, exist_ok=True)
fails = []
LOGP = os.path.join(LOGDIR, "aliases_server_%d.log" % PORT)


def check(name, ok, detail=""):
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        fails.append(name)


def http(method, path, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def jb(text):
    try:
        return json.loads(text)
    except ValueError:
        return {}


PROMPT = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox"
BASE = {"prompt": PROMPT, "temperature": 0, "seed": 1, "n_predict": 40, "cache_prompt": False, "return_tokens": True}


def comp(extra, base=None):
    b = dict(BASE if base is None else base)
    b.update(extra)
    st, txt = http("POST", "/completion", b)
    j = jb(txt)
    return st, j.get("tokens"), j


def toks(extra, base=None):
    st, t, j = comp(extra, base)
    return st, t


def same_and_effective(label, canon, alias, base=None, effective=True):
    """alias request == canonical request; canonical differs from the baseline (the setting does something)."""
    s0, t0 = toks({}, base)
    sc, tc = toks(canon, base)
    sa, ta = toks(alias, base)
    ok_status = (s0 == 200 and sc == 200 and sa == 200)
    check(label + " status 200 x3", ok_status, "%s/%s/%s" % (s0, sc, sa))
    check(label + " alias == canonical", ok_status and ta == tc and ta is not None)
    if effective:
        check(label + " canonical != baseline (setting effective)", ok_status and tc != t0)
        check(label + " alias != baseline (alias not ignored)", ok_status and ta != t0)
    return t0, tc, ta


f = open(LOGP, "wb")
proc = subprocess.Popen([EXE, "--host", "127.0.0.1", "--port", str(PORT), "-m", MODEL, "-c", "2048", "-np", "1", "-t", "8"],
                        stdout=f, stderr=subprocess.STDOUT, creationflags=0x08000000)
try:
    t0 = time.time()
    ready = False
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            break
        try:
            if http("GET", "/health", timeout=3)[0] == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(1)
    check("server ready", ready, "pid %s" % proc.pid)
    if not ready:
        raise SystemExit(2)

    # ---- plain renames
    same_and_effective("rep_pen", {"repeat_penalty": 1.8, "repeat_last_n": 64}, {"rep_pen": 1.8, "repeat_last_n": 64})
    # max_length is an alias of n_predict, like max_tokens: when a request carries BOTH names the canonical one wins (alias lookup only
    # runs for an absent canonical key), so the alias is compared on a base WITHOUT n_predict (MEASURED 2026-11-08 gate b1108g4: the old
    # check sent BASE's n_predict 40 together with max_length 5 and got 40 tokens on CPU and CUDA alike -- the test, not the binary)
    np_base = {k: v for k, v in BASE.items() if k != "n_predict"}
    s_c5, t_c5 = toks({"n_predict": 5}, np_base)
    s_a5, t_a5 = toks({"max_length": 5}, np_base)
    s_b40, t_b40 = toks({})
    check("max_length status 200 x3", s_c5 == 200 and s_a5 == 200 and s_b40 == 200, "%s/%s/%s" % (s_c5, s_a5, s_b40))
    check("max_length alias == canonical", t_a5 is not None and t_a5 == t_c5)
    check("max_length yields 5 tokens", t_a5 is not None and len(t_a5) == 5, str(len(t_a5 or [])))
    check("max_length alias != baseline n_predict 40 (alias not ignored)", t_a5 is not None and t_b40 is not None and t_a5 != t_b40)
    t_both = toks({"max_length": 5})[1]
    check("both names sent: canonical n_predict 40 wins (same rule as max_tokens)", t_both is not None and len(t_both) == 40,
          str(len(t_both or [])))

    # min_tokens: end-of-generation masked; use a short natural ending: a stop-prone prompt
    short = dict(BASE, prompt="Q: What is 2+2?\nA: 4\n\nQ: What is 3+3?\nA:", n_predict=48)
    # sampler settings need a prompt whose next-token distribution is NOT peaked: on the repeated-fox prompt the 0.5B model gives the
    # greedy text at temperature 1.0 whatever typical / mirostat / dynatemp say (MEASURED 2026-11-08, CPU and CUDA). The baseline is the
    # SAME temperature and seed without the setting, so a difference proves the setting itself, not the temperature.
    creative = dict(BASE, prompt="Here is a short story about a dragon and a robot who become friends. Once upon a time,")
    sampled = dict(creative, temperature=1.0, seed=7)
    same_and_effective("typical", {"typical_p": 0.2}, {"typical": 0.2}, sampled)
    # mirostat must be repeatable before an alias comparison means anything (work/mirostat-mu-1108: mu was never initialised)
    check("mirostat repeatable (two identical seeded requests, same tokens)",
          toks({"mirostat": 2}, sampled)[1] == toks({"mirostat": 2}, sampled)[1])
    same_and_effective("mirostat_mode", {"mirostat": 2}, {"mirostat_mode": 2}, sampled)
    s_a, t_a = toks({"dynatemp_range": 0.9}, sampled)
    s_b, t_b = toks({"dynatemp_range": 0.9, "temp_exponent": 8.0}, sampled)
    s_c, t_c = toks({"dynatemp_range": 0.9, "dynatemp_exponent": 8.0}, sampled)
    check("temp_exponent alias == dynatemp_exponent", s_b == 200 and s_c == 200 and t_b == t_c and t_b is not None)
    check("temp_exponent effective (exponent 8 != default exponent)", t_b != t_a)

    # ---- penalty range names (-1 = whole context accepted for the alias family)
    s0, t_rl = toks({"repeat_penalty": 1.8, "repeat_last_n": 8})
    names = ["penalty_range", "repetition_range", "repetition_penalty_range", "rep_pen_range"]
    outs = [toks({"repeat_penalty": 1.8, n: 8}) for n in names]
    check("penalty-range family == repeat_last_n 8", all(o[0] == 200 and o[1] == t_rl for o in outs), str([o[0] for o in outs]))
    s_all, t_all = toks({"repeat_penalty": 1.8, "repeat_last_n": 4096})
    s_m1, t_m1 = toks({"repeat_penalty": 1.8, "rep_pen_range": -1})
    check("rep_pen_range -1 accepted (200)", s_m1 == 200)
    check("rep_pen_range -1 == whole-context window", s_m1 == 200 and t_m1 == t_all)
    s_neg, _ = toks({"repeat_penalty": 1.8, "repeat_last_n": -1})
    check("repeat_last_n -1 still refused (existing bound unchanged)", s_neg == 400, str(s_neg))

    # ---- dynatemp low/high pair -> temperature (hi+lo)/2, range (hi-lo)/2
    s_d, t_d = toks({"temperature": 1.0, "dynatemp_range": 0.5, "seed": 7}, creative)
    s_p, t_p = toks({"dynatemp_low": 0.5, "dynatemp_high": 1.5, "seed": 7}, creative)
    s_q, t_q = toks({"min_temp": 0.5, "max_temp": 1.5, "seed": 7}, creative)
    s_b0, t_b0 = toks({"seed": 7}, creative)
    check("dynatemp_low/high == temperature 1.0 + range 0.5", s_p == 200 and t_p == t_d and t_p is not None)
    check("min_temp/max_temp == dynatemp_low/high", s_q == 200 and t_q == t_p)
    check("dynatemp pair != greedy baseline (effective)", t_p != t_b0)
    s_e, t_e = toks({"dynatemp_low": 1.0, "dynatemp_high": 1.0, "temperature": 0.0, "seed": 7}, creative)
    check("equal low/high = static temperature, temperature untouched (greedy)", s_e == 200 and t_e == t_b0)
    s_bad, _ = toks({"dynatemp_low": 2.0, "dynatemp_high": 1.0})
    check("low > high refused (400)", s_bad == 400, str(s_bad))

    # ---- stop_sequence: native, OAI completions, chat
    # the stop string must occur in the greedy continuation: ' 6\n\nQ: What is 4+4?...' has no '.', so the old '.' stop never fired
    s0, t_s0 = toks({"stop": ["Q:"]}, short)
    s1, t_s1 = toks({"stop_sequence": ["Q:"]}, short)
    s2, t_s2 = toks({}, short)
    check("stop_sequence == stop (/completion)", s0 == 200 and s1 == 200 and t_s0 == t_s1 and t_s0 is not None)
    check("stop effective (shorter than baseline)", t_s0 is not None and t_s2 is not None and len(t_s0) < len(t_s2))
    st, txt = http("POST", "/v1/completions", dict(prompt=short["prompt"], temperature=0, seed=1, max_tokens=48, stop_sequence=["."], cache_prompt=False))
    st2, txt2 = http("POST", "/v1/completions", dict(prompt=short["prompt"], temperature=0, seed=1, max_tokens=48, stop=["."], cache_prompt=False))
    c1 = (jb(txt).get("choices") or [{}])[0].get("text")
    c2 = (jb(txt2).get("choices") or [{}])[0].get("text")
    check("stop_sequence == stop (/v1/completions)", st == 200 and st2 == 200 and c1 == c2 and c1 is not None, "%r | %r" % (c1, c2))

    # ---- token-id bans
    t_first = toks({"n_predict": 6})[1]
    victim = t_first[0] if t_first else 0
    s_lb, t_lb = toks({"logit_bias": [[victim, False]], "n_predict": 6})
    s_cb, t_cb = toks({"custom_token_bans": [victim], "n_predict": 6})
    s_cs, t_cs = toks({"custom_token_bans": "%d" % victim, "n_predict": 6})
    s_bt, t_bt = toks({"banned_tokens": [victim], "n_predict": 6})
    check("custom_token_bans ids == logit_bias -inf", s_cb == 200 and t_cb == t_lb and t_lb is not None and victim not in t_lb)
    check("custom_token_bans comma string == logit_bias -inf", s_cs == 200 and t_cs == t_lb)
    check("banned_tokens int ids == logit_bias -inf (TabbyAPI shape)", s_bt == 200 and t_bt == t_lb)
    check("ban effective (first token changed)", t_lb is not None and t_first is not None and t_lb[0] != victim)
    s_str, t_str = toks({"banned_tokens": ["the"], "n_predict": 6})
    check("banned_tokens string array still accepted (Kobold shape, unchanged)", s_str == 200)

    # ---- allowed_tokens
    allowed = sorted(set((toks({"n_predict": 12})[1] or [0])))[:3]
    s_ai, t_ai = toks({"allowed_token_ids": allowed, "n_predict": 12})
    s_at, t_at = toks({"allowed_tokens": allowed, "n_predict": 12})
    check("allowed_tokens == allowed_token_ids", s_ai == 200 and s_at == 200 and t_ai == t_at and t_at is not None)
    check("allowed_tokens restricts output to the set", t_at is not None and all(x in allowed for x in t_at), str(allowed))

    # ---- ban_eos_token
    eos_short = dict(BASE, prompt="Hello", n_predict=64)
    s_i, t_i = toks({"ignore_eos": True}, eos_short)
    s_b, t_b = toks({"ban_eos_token": True}, eos_short)
    check("ban_eos_token == ignore_eos", s_i == 200 and s_b == 200 and t_i == t_b and t_b is not None)
    check("ban_eos_token runs to n_predict", t_b is not None and len(t_b) == 64, str(len(t_b or [])))
    s_n, t_n = toks({}, eos_short)
    print("INFO ban_eos_token baseline (no ban) generated %d tokens of 64" % len(t_n or []), flush=True)

    # ---- min_length
    s_m, t_m = toks({"min_tokens": 30, "stop": ["\n"]}, short)
    s_l, t_l = toks({"min_length": 30, "stop": ["\n"]}, short)
    check("min_length == min_tokens", s_m == 200 and s_l == 200 and t_m == t_l and t_l is not None)

    # ---- unknown key: one WRN per key, still 200
    def wrn_count(key):
        try:
            with open(LOGP, "rb") as g:
                return g.read().decode("utf-8", "replace").count("request key '%s' is not a known" % key)
        except OSError:
            return -1
    before = wrn_count("top_P")
    check("no WRN before the typo request (control)", before == 0, str(before))
    s_u, t_u = toks({"top_P": 0.1})
    s_u0, t_u0 = toks({})
    check("unknown key still answered 200 and ignored", s_u == 200 and t_u == t_u0)
    time.sleep(0.5)
    after1 = wrn_count("top_P")
    toks({"top_P": 0.1})
    toks({"top_P": 0.5})
    time.sleep(0.5)
    after3 = wrn_count("top_P")
    check("exactly one WRN naming top_P after first request", after1 == 1, str(after1))
    check("repeat requests add no further WRN", after3 == 1, str(after3))
    toks({"zzz_not_a_key": 1, "another_one": 2})
    time.sleep(0.5)
    check("each distinct unknown key gets its own WRN", wrn_count("zzz_not_a_key") == 1 and wrn_count("another_one") == 1)
    # alias and envelope keys must not warn
    toks({"rep_pen": 1.1, "max_length": 3, "stop_sequence": ["x"], "min_temp": 1.0, "max_temp": 1.0, "ban_eos_token": False})
    st, _ = http("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
                                                  "temperature": 0, "stream": False})
    check("chat route answered", st == 200, str(st))
    time.sleep(0.5)
    for k in ["rep_pen", "max_length", "stop_sequence", "min_temp", "max_temp", "ban_eos_token", "messages", "model", "max_tokens"]:
        check("no WRN for known key %s" % k, wrn_count(k) == 0, str(wrn_count(k)))
finally:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    f.close()

print("FAILED %d: %s" % (len(fails), fails))
sys.exit(1 if fails else 0)
