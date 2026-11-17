"""Stub-only checks for the feature-exercised-share guard in speed_bench.py (no server, no datasets).

Run: python test_exercised_share.py
"""
import importlib.util
import os
import sys
import types

for name in ("requests", "datasets", "tqdm"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["datasets"].get_dataset_config_names = lambda *a, **k: []
sys.modules["datasets"].load_dataset = lambda *a, **k: None
sys.modules["tqdm"].tqdm = lambda x=None, **k: x

_spec = importlib.util.spec_from_file_location("speed_bench", os.path.join(os.path.dirname(__file__), "speed_bench.py"))
sb = importlib.util.module_from_spec(_spec)
sys.modules["speed_bench"] = sb  # dataclass resolves its own module by name
_spec.loader.exec_module(sb)


def mk(draft_n, accepted, tokens):
    return sb.RequestResult(
        id="x", category="c", ok=True, turns=1, latency_s=1.0, prompt_tokens=10,
        completion_tokens=tokens, total_tokens=10 + tokens, finish_reason="stop",
        draft_n=draft_n, draft_n_accepted=accepted, prompt_ms=None, predicted_ms=None,
        prompt_per_second=None, predicted_per_second=None, error=None)


def summ(results):
    # summarize_group needs only the fields set above for the counters under test
    return sb.summarize_group("overall", results)


def test_share_and_refusal():
    rs = [mk(0, 0, 100)] * 19 + [mk(10, 5, 100)]
    s = summ(rs)
    assert s["exercised_requests"] == 1 and abs(s["exercised_share"] - 0.05) < 1e-9
    assert abs(s["draft_per_token"] - 10 / 2000) < 1e-9
    assert sb.exercised_share_refusal(s, 0.05) is None  # equal to floor passes
    assert sb.exercised_share_refusal(s, 0.10) is not None


def test_never_drafted_is_not_refused():
    s = summ([mk(0, 0, 50)] * 4)
    assert sb.exercised_share_refusal(s, 0.5) is None
    assert sb.exercised_share_refusal(s, 0.0) is None


if __name__ == "__main__":
    test_share_and_refusal()
    test_never_drafted_is_not_refused()
    print("ok")
