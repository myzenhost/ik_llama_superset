import pytest
from utils import ServerPreset, use_local_stories260k

# adaptive_target with nothing truncating in front of it draws from the whole vocabulary (measured on Llama-3.2-1B: 99.8% of drawn tokens
# below 1e-3 probability). adaptive_untruncated picks the behaviour: "floor" (default) applies min_p = adaptive_floor first, "refuse"
# answers HTTP 400, "allow" is the old behaviour. The request is "untruncated" when top_k, top_p, min_p (and the other truncating
# samplers) are all at their off values. A plain adaptive_target request with the stock defaults is truncated and is not touched.
# CPU test server on stories260K only.

server = ServerPreset.tinyllama2()

UNTRUNC = {"top_k": 0, "top_p": 1.0, "min_p": 0.0, "temperature": 0.8, "adaptive_target": 0.3, "n_predict": 8, "prompt": "Once upon a time"}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    use_local_stories260k(server)
    server.n_ctx = 256
    server.n_slots = 1


def _post(**over):
    body = dict(UNTRUNC)
    body.update(over)
    return server.make_request("POST", "/completion", data=body)


def test_default_mode_floors_an_untruncated_request():
    server.start()
    res = _post(seed=1)
    assert res.status_code == 200, res.body
    gs = res.body["generation_settings"]
    assert gs["adaptive_untruncated"] == "floor", gs
    assert gs["adaptive_p_floor_applied"] is True, gs
    assert abs(gs["adaptive_floor"] - 0.02) < 1e-6, gs


def test_control_truncated_request_is_not_touched():
    server.start()
    res = _post(seed=1, top_k=40)  # control: same request with a truncating sampler on
    assert res.status_code == 200, res.body
    assert res.body["generation_settings"]["adaptive_p_floor_applied"] is False, res.body["generation_settings"]


def test_control_adaptive_off_is_not_touched():
    server.start()
    res = _post(seed=1, adaptive_target=-1.0)
    assert res.status_code == 200, res.body
    assert res.body["generation_settings"]["adaptive_p_floor_applied"] is False, res.body["generation_settings"]


def test_refuse_answers_400_when_untruncated():
    server.start()
    res = _post(adaptive_untruncated="refuse")
    assert res.status_code == 400, (res.status_code, res.body)
    assert "truncating sampler" in res.body["error"]["message"], res.body


def test_top_k_at_or_above_the_vocab_is_neutral():
    # stories260K has a 512-token vocab: top_k = 10000000 drops nothing, so with "refuse" the request is still untruncated.
    # Control: top_k = 40 on the same server truncates and is served.
    server.start()
    res = _post(adaptive_untruncated="refuse", top_k=10000000)
    assert res.status_code == 400, (res.status_code, res.body)
    res = _post(adaptive_untruncated="refuse", top_k=40)
    assert res.status_code == 200, (res.status_code, res.body)


def test_refuse_lets_a_truncated_request_through():
    server.start()
    res = _post(adaptive_untruncated="refuse", min_p=0.05)
    assert res.status_code == 200, (res.status_code, res.body)


def _not_refused(res):
    # "allow" is the old behaviour: it is NOT refused, but an untruncated adaptive draw from the whole vocabulary regularly yields tokens the
    # content parser rejects, so the server answers 500 ("does not match the expected Content-only format"). That 500 is the measured defect
    # the floor removes, so it is accepted here; a 400 would mean the request was refused, which is what "refuse" is for.
    return res.status_code in (200, 500) and "truncating sampler" not in str(res.body)


def test_allow_keeps_the_old_behaviour():
    server.start()
    seen_ok = None
    for seed in range(1, 9):
        res = _post(adaptive_untruncated="allow", seed=seed)
        assert _not_refused(res), (seed, res.status_code, res.body)
        if res.status_code == 200:
            seen_ok = res
    if seen_ok is not None:  # when the draw happened to parse, the settings must still report the mode truthfully
        gs = seen_ok.body["generation_settings"]
        assert gs["adaptive_untruncated"] == "allow" and gs["adaptive_p_floor_applied"] is False, gs


def test_unknown_mode_is_rejected():
    server.start()
    res = _post(adaptive_untruncated="sometimes")
    assert res.status_code == 400, (res.status_code, res.body)
    assert "floor, refuse or allow" in res.body["error"]["message"], res.body


def test_floor_changes_the_draw_and_is_deterministic():
    server.start()
    differ = 0
    for seed in range(1, 9):
        a = _post(seed=seed, adaptive_untruncated="floor", adaptive_floor=0.2, n_predict=24)
        b = _post(seed=seed, adaptive_untruncated="floor", adaptive_floor=0.2, n_predict=24)
        assert a.status_code == 200 and b.status_code == 200, (seed, a.status_code, a.body, b.status_code)
        assert a.body["content"] == b.body["content"], (seed, a.body["content"], b.body["content"])  # control: same mode and seed repeat
        allow = _post(seed=seed, adaptive_untruncated="allow", n_predict=24)
        # a different outcome is either the 500 the whole-vocabulary draw causes or different text
        differ += allow.status_code != 200 or allow.body["content"] != a.body["content"]
    # a floor of 0.2 removes most of a 512-token tail; if it never changed an outcome in 8 seeds the floor is not being applied
    assert differ >= 1, "floor mode matched allow mode on every seed"


def test_server_wide_refuse_with_a_per_request_override():
    server.extra_args = ["--adaptive-p-untruncated", "refuse"]
    server.start()
    res = _post()
    assert res.status_code == 400, (res.status_code, res.body)
    res = _post(adaptive_untruncated="allow", seed=1)
    assert _not_refused(res), (res.status_code, res.body)
    res = _post(top_k=40)
    assert res.status_code == 200, (res.status_code, res.body)
