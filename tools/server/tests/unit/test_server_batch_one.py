# -b 1 is an accepted setting and must not kill the server at startup.
#
# common_context_can_seq_rm probes partial sequence removal by decoding 2 tokens in ONE batch; with -b 1 that
# batch exceeded the context's logical batch and the legacy decoder hit GGML_ASSERT(n_tokens_all <=
# cparams.n_batch), aborting the process before it served anything (2026-09-28, gemma-4-E2B and this model).
# The probe now feeds n_batch-sized pieces and the decoder reports an oversized batch instead of aborting.
import pytest
from utils import ServerPreset, ServerProcess

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.fa = "off"


@pytest.mark.parametrize("extra", [["-b", "1", "-ub", "1"], ["-b", "1"], ["-b", "2", "-ub", "2"]])
def test_small_logical_batch_starts_and_generates(extra):
    global server
    server.extra_args = extra
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is", "temperature": 0.0, "top_k": 1, "n_predict": 8,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    # same greedy text as the default batch: splitting the prompt into 1-token decodes must not change it
    assert res.body["content"].startswith(" to be kind."), res.body["content"]
