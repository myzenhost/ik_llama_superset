"""Binary / int8 quantised output for /v1/embeddings and /embeddings
(row S69-VESPA-PICK-BINARY-INT8-EMBEDDING-OUTPUT-2026-10-15-01; spec in tools/server/server-embd-quant.h).

The reference for every quantised vector is the SAME request's float vector, so the tests check the mechanism (pool + normalise
in float, Matryoshka cut, THEN pack), not a stored number:
  * binary / ubinary : bit i of the packed vector == (float[i] > 0), first value = highest bit, binary is the signed view of ubinary
  * int8             : round(float[i] * 127) clamped to +-127
  * a dimension that is not a multiple of 8 is refused with 400 for the binary modes
"""
import base64
import struct
import pytest
from utils import ServerPreset

server = ServerPreset.bert_bge_small()

PROMPT = "I believe the meaning of life is"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.bert_bge_small()
    server.pooling = 'mean'


def _post(**extra):
    data = {"input": PROMPT}
    data.update(extra)
    return server.make_request("POST", "/v1/embeddings", data=data)


def _unpack(packed):
    """packed bytes (0..255) -> list of bits, first value = highest bit"""
    return [(b >> (7 - j)) & 1 for b in packed for j in range(8)]


@pytest.mark.parametrize("dims", [None, 64, 8])
def test_binary_bits_equal_sign_of_float(dims):
    server.start()
    extra = {} if dims is None else {"dimensions": dims}
    f = _post(**extra)
    b = _post(encoding_format="binary", **extra)
    assert f.status_code == 200 and b.status_code == 200
    fv = f.body['data'][0]['embedding']
    bv = b.body['data'][0]['embedding']
    assert len(bv) == len(fv) // 8                      # packed dimension = dim / 8
    assert all(isinstance(x, int) and -128 <= x <= 127 for x in bv)
    bits = _unpack([x & 0xFF for x in bv])
    assert bits == [1 if x > 0 else 0 for x in fv]       # sign(float) == unpacked binary
    assert b.body['data'][0]['encoding_format'] == "binary"


def test_ubinary_is_the_unsigned_view_of_binary():
    server.start()
    b = _post(encoding_format="binary", dimensions=64).body['data'][0]['embedding']
    u = _post(encoding_format="ubinary", dimensions=64).body['data'][0]['embedding']
    assert len(u) == 8
    assert all(isinstance(x, int) and 0 <= x <= 255 for x in u)
    assert u == [x & 0xFF for x in b]


def test_int8_is_scaled_rounded_float():
    server.start()
    f = _post(dimensions=64).body['data'][0]['embedding']
    q = _post(encoding_format="int8", dimensions=64).body['data'][0]['embedding']
    assert len(q) == len(f) == 64
    for x, y in zip(f, q):
        assert isinstance(y, int) and -127 <= y <= 127
        assert abs(y - round(x * 127)) <= 1               # <= 1: the float JSON is printed with ~6 digits, a .5 edge can flip
    # the vector is unit length, so the largest component must use a good part of the range
    assert max(abs(y) for y in q) > 20


def test_binary_dimension_not_multiple_of_8_is_400():
    server.start()
    for dims in (12, 63, 1):
        r = _post(encoding_format="binary", dimensions=dims)
        assert r.status_code == 400, dims
        assert "error" in r.body
        r = _post(encoding_format="ubinary", dimensions=dims)
        assert r.status_code == 400, dims
    # int8 has no packing constraint
    assert _post(encoding_format="int8", dimensions=12).status_code == 200


def test_binary_matryoshka_head_is_the_prefix_of_the_full_bits():
    """the cut happens BEFORE packing: the first 64 bits of the native binary vector are the 64-dim binary vector only when
    normalisation does not flip a sign -- it cannot, so the bits of the head are a prefix of the full bits"""
    server.start()
    full = _post(encoding_format="ubinary").body['data'][0]['embedding']
    head = _post(encoding_format="ubinary", dimensions=64).body['data'][0]['embedding']
    assert head == full[:8]


def test_multiple_inputs_keep_request_order():
    server.start()
    texts = ["I believe the meaning of life is", "Write a joke about AI", "This is a test", "zzz"]
    fl = server.make_request("POST", "/v1/embeddings", data={"input": texts, "dimensions": 32})
    bn = server.make_request("POST", "/v1/embeddings", data={"input": texts, "dimensions": 32, "encoding_format": "binary"})
    assert fl.status_code == 200 and bn.status_code == 200
    assert [d['index'] for d in bn.body['data']] == [0, 1, 2, 3]
    for df, db in zip(fl.body['data'], bn.body['data']):
        assert _unpack([x & 0xFF for x in db['embedding']]) == [1 if x > 0 else 0 for x in df['embedding']]


def test_non_oai_route_returns_a_one_row_matrix():
    server.start()
    f = server.make_request("POST", "/embeddings", data={"input": PROMPT, "dimensions": 16})
    b = server.make_request("POST", "/embeddings", data={"input": PROMPT, "dimensions": 16, "encoding_format": "binary"})
    assert f.status_code == 200 and b.status_code == 200
    fv = f.body[0]['embedding'][0]
    bv = b.body[0]['embedding']
    assert len(bv) == 1 and len(bv[0]) == 2
    assert _unpack([x & 0xFF for x in bv[0]]) == [1 if x > 0 else 0 for x in fv]


def test_existing_formats_unchanged():
    """controls: float and base64 still work, an unknown format is still refused"""
    server.start()
    f = _post(dimensions=16)
    assert f.status_code == 200
    assert len(f.body['data'][0]['embedding']) == 16
    assert 'encoding_format' not in f.body['data'][0]
    assert _post(encoding_format="float", dimensions=16).status_code == 200
    r = _post(encoding_format="base64", dimensions=16)
    assert r.status_code == 200
    raw = base64.b64decode(r.body['data'][0]['embedding'])
    assert len(raw) == 16 * 4
    floats = struct.unpack('16f', raw)
    for a, b in zip(floats, f.body['data'][0]['embedding']):
        assert abs(a - b) < 1e-3
    bad = _post(encoding_format="hex")
    assert bad.status_code == 400
    assert "error" in bad.body
