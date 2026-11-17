#!/usr/bin/env python3
"""llama-server greedy decoding with CTranslate2's logits processors, against the REAL CTranslate2 engine on the SAME model.

   python tests/test_ct2_sampling_parity.py <tree-root> <llama-server> [--explore]

Greedy search (CT2 beam_size=1, sampling_topk=1) with, one at a time: repetition_penalty, no_repeat_ngram_size, min_length, disable_unk,
suppress_sequences (single tokens and token sequences). The server's answer must have the SAME token ids as the engine's.
SOURCE-READ (OpenNMT/CTranslate2 @ f2f54b42, MIT): src/decoding.cc:1091-1120 (the processors and their order), src/decoding_utils.cc:34-62
(repetition penalty over the ids of the sequence so far), :64-95 (no-repeat n-gram), :98-150 (suppress sequences: the LAST token of a banned
sequence is disabled when the tokens before it end the history), src/cpu/primitives.cc:411-428 (score < 0 ? score * penalty : score / penalty).
"""
import json
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1]).resolve()
server_bin = sys.argv[2]
explore = "--explore" in sys.argv
sys.path.insert(0, str(root / "tests"))
sys.argv = [sys.argv[0], str(root), server_bin]

import test_ct2_beam_parity as B  # noqa: E402

E = B.E
T = E.T
check = T.check
np = E.np

MAXLEN = 24


def run():
    with tempfile.TemporaryDirectory() as tmp:
        tokdir = Path(tmp) / "tok"
        vocab = E.make_tokenizer(tokdir)
        ct2dir = B.build(tmp, tokdir, vocab, eos_boost=2.5, seed=23, name="m")
        gguf = Path(tmp) / "m.gguf"
        r = E.subprocess.run([sys.executable, str(root / "convert_hf_to_gguf.py"), str(ct2dir), "--ct2-tokenizer-dir", str(tokdir), "--outfile", str(gguf),
                              "--outtype", "f32"], capture_output=True, text=True, cwd=str(root))
        check("convert_hf_to_gguf.py converts the CT2 directory", r.returncode == 0 and gguf.exists(), (r.stderr or r.stdout)[-600:])
        gen = E.ctranslate2.Generator(str(ct2dir), device="cpu", compute_type="float32")
        srv = E.Server(gguf, slots=2)
        prompt = None

        def ct2(include_prompt=False, **kw):
            nonlocal prompt
            kw.setdefault("max_length", MAXLEN)
            res = gen.generate_batch([[vocab[i] for i in prompt]], beam_size=1, sampling_topk=1, include_prompt_in_result=include_prompt,
                                     return_scores=False, **kw)[0]
            ids = list(res.sequences_ids[0])
            return ids[len(prompt) - 1:] if include_prompt else ids   # the result starts at start_tokens[1:] (src/decoding.cc:1161)

        def fork(**kw):
            nonlocal prompt
            req = {"prompt": prompt, "n_predict": MAXLEN, "temperature": 0.0, "return_tokens": True, "cache_prompt": False, "top_k": 1}
            req.update(kw)
            out = srv.post("/completion", req)
            toks = list(out["tokens"])
            return toks[:-1] if out.get("stop_type") == "eos" and toks and toks[-1] == 2 else toks

        def case(name, ref, got, must_differ_from=None):
            ok = ref == got
            check(f"{name}: same token ids as CTranslate2", ok, f"ct2={ref} fork={got}")
            if explore:
                print(f"     {name}: {'SAME' if ok else 'DIFF'}\n        ct2 ={ref}\n        fork={got}")
            if must_differ_from is not None:
                check(f"{name}: CONTROL the option changes the output", ref != must_differ_from)

        try:
            # A prompt on which the options matter: its tokens are ones the model likes to say, so penalising them (prompt in the
            # history) gives another answer than penalising only what was generated. Found with the real engine, deterministic seed.
            rng = np.random.default_rng(3)
            for _ in range(2000):
                cand = [1] + [int(x) for x in rng.integers(3, len(vocab), size=9)]
                prompt = cand
                b0 = ct2()
                if not 8 <= len(b0) <= 16:          # ends on EOS well before max_length, so min_length has something to change
                    continue
                a1, a0 = ct2(True, repetition_penalty=2.0, max_length=MAXLEN), ct2(False, repetition_penalty=2.0)
                if a1 != b0 and a0 != b0 and a1 != a0:
                    break
            else:
                raise SystemExit("no discriminating prompt found")
            print("     prompt:", prompt)
            base = ct2()
            check("plain greedy: same token ids as CTranslate2", base == fork(), f"ct2={base} fork={fork()}")
            check("control: the plain output ends on EOS before max_length", 8 <= len(base) <= 16, str(base))
            # --- max_length. With include_prompt_in_result=True the prompt counts toward max_length (src/decoding.cc:359-381: return_prefix),
            # so the generated part is max_length - (len(prompt) - 1) tokens; with False it is max_length.
            n_pre = len(prompt) - 1
            # --- repetition penalty. include_prompt_in_result=True puts the prompt into the penalised history (like HF generate());
            # include_prompt_in_result=False forwards it at once and only its LAST token is in the history (src/decoding.cc:1139-1176)
            for pen in (1.3, 2.0):
                case(f"repetition_penalty {pen}, prompt in history", ct2(True, repetition_penalty=pen),
                     fork(repeat_penalty=pen, repeat_last_n=4096, n_predict=MAXLEN - n_pre), must_differ_from=base)
            # known difference, not asserted: CT2's include_prompt_in_result=False history is generated tokens (+ the last prompt token), which
            # repeat_penalty cannot express (penalty_scope only changes frequency / presence penalty). DECLINED, see the report.
            if explore:
                for pen in (2.0,):
                    r0, f0 = ct2(False, repetition_penalty=pen), fork(repeat_penalty=pen, repeat_last_n=4096, penalty_scope="generated")
                    print(f"     [info] repetition_penalty {pen}, generated tokens only: {'SAME' if r0 == f0 else 'DIFF'} (declined area)\n        ct2 ={r0}\n        fork={f0}")
            # --- no repeat n-gram
            for n in (2, 3):
                case(f"no_repeat_ngram_size {n}, prompt in history", ct2(True, no_repeat_ngram_size=n), fork(no_repeat_ngram_size=n, no_repeat_ngram_last_n=-1, n_predict=MAXLEN - n_pre))
            # --- min_length: EOS is not allowed before `min_length` tokens (src/decoding.cc:384-408); the plain output is shorter than that
            ml = len(base) + 4
            case(f"min_length {ml}", ct2(min_length=ml), fork(min_tokens=ml), must_differ_from=base)
            case("disable_unk", ct2(disable_unk=True), fork(logit_bias=[[0, False]]))
            # --- suppress sequences
            first = base[0]
            case("suppress_sequences [token]", ct2(suppress_sequences=[[vocab[first]]]), fork(logit_bias=[[first, False]]), must_differ_from=base)
            case("suppress_sequences [token] via the new field", ct2(suppress_sequences=[[vocab[first]]]), fork(suppress_sequences=[[first]]))
            seq = [base[2], base[3]]
            case("suppress_sequences [token, token]", ct2(suppress_sequences=[[vocab[t] for t in seq]]), fork(suppress_sequences=[seq]), must_differ_from=base)
            seq3 = [base[3], base[4], base[5]]
            case("suppress_sequences [token, token, token] + [token]", ct2(suppress_sequences=[[vocab[t] for t in seq3], [vocab[base[7]]]]),
                 fork(suppress_sequences=[seq3, [base[7]]]), must_differ_from=base)
        finally:
            srv.close()


run()
print()
print("FAILED: " + ", ".join(T.failures) if T.failures else "all checks passed")
sys.exit(1 if T.failures and not explore else 0)
