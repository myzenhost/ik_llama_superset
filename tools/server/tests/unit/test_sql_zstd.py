import os
import pytest
from utils import ServerPreset

# G4: sqlite-zstd transparent compression over the G3 SQL store (row
# S69-G4-ZSTD-ROUTES-PORT-USER-APPROVED-2026-09-27-01). SOURCE-READ: examples/server/server.cpp:2097-2127,
# 2236-2243 @ 35f0cac75.
#
# Three tiers, so the security fix is tested on EVERY build:
#  1. gating: without --sql-save-file the 4 routes refuse with 501 (any binary).
#  2. request validation, the injection fix: with a working SQL store but WITHOUT the extension, a
#     malicious "table"/"column" is refused with 400 BEFORE any SQL runs. The pre-merge source
#     concatenated those strings into its SQL, so this is the fail-before/pass-after case.
#     Needs a -DLLAMA_SERVER_SQLITE3=ON binary; skipped otherwise.
#  3. end to end with the real extension: only when LLAMA_TEST_SQLITE_ZSTD_EXT names the sqlite-zstd
#     library file (it is a separately distributed LGPL-3.0 binary, not vendored; skipped otherwise).

server = ServerPreset.tinyllama2()
ZSTD_EXT = os.environ.get("LLAMA_TEST_SQLITE_ZSTD_EXT", "")


def sql_supported() -> bool:
    global server
    server = ServerPreset.tinyllama2()
    os.makedirs("tmp", exist_ok=True)
    server.sql_save_file = "./tmp/sql_zstd_probe.db"
    server.start()
    res = server.make_request("GET", "/version")
    supported = res.status_code == 200 and res.body.get("features", {}).get("sql") is True
    server.stop()
    return supported


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    os.makedirs("tmp", exist_ok=True)


def test_zstd_routes_gated_without_sql_save_file():
    global server
    server.start()
    assert server.make_request("GET", "/zstd_get_configs").status_code == 501
    assert server.make_request("POST", "/zstd_incremental_maintenance", data={"db_load": 0.5}).status_code == 501
    assert server.make_request("POST", "/zstd_enable_transparent", data={
        "table": "sessions", "column": "data", "compression_level": 3, "train_dict_samples_ratio": 100}).status_code == 501
    assert server.make_request("POST", "/zstd_update_transparent", data={"compression_level": 5}).status_code == 501
    res = server.make_request("GET", "/version")
    assert res.body["features"]["zstd_compression"] is False


SQL_SKIP = pytest.mark.skipif(not sql_supported(), reason=(
    "needs a -DLLAMA_SERVER_SQLITE3=ON server binary (row S69-G4-ZSTD-ROUTES-PORT-USER-APPROVED-2026-09-27-01)"))

INJECTIONS = [
    # closes the JSON string and the SQL literal the pre-merge code built, then appends SQL
    ('sessions", "column": "data"}\'); DROP TABLE sessions; --', "data"),
    ("sessions", 'data", "compression_level": 1}\'); DROP TABLE names; --'),
    ("_zstd_configs", "data"),        # not one of the 3 store tables
    ("sessions", "key"),              # not the compressible column
    ("", "data"),                     # empty table must not default to "sessions"
]


@SQL_SKIP
class TestZstdValidation:
    def setup_method(self):
        global server
        server.sql_save_file = "./tmp/test_sql_zstd.db"
        # SAFE-DELETE-OK: scratch SQLite file this test class creates itself under tests/tmp (gitignored).
        if os.path.exists(server.sql_save_file):
            os.remove(server.sql_save_file)

    @pytest.mark.parametrize("table,column", INJECTIONS)
    def test_enable_refuses_injection_before_any_sql(self, table, column):
        global server
        server.start()
        # a row in `sessions` and `names` so a successful DROP would be visible
        assert server.make_request("POST", "/save", data={"key": "k1", "data": {"x": 1}}).status_code == 200
        assert server.make_request("POST", "/save", data={"storeName": "names", "key": "k1", "data": "n"}).status_code == 200
        res = server.make_request("POST", "/zstd_enable_transparent", data={
            "table": table, "column": column, "compression_level": 3, "train_dict_samples_ratio": 100})
        assert res.status_code == 400, res.body
        # both tables still intact
        assert server.make_request("POST", "/load", data={"key": "k1"}).status_code == 200
        assert server.make_request("POST", "/load", data={"storeName": "names", "key": "k1"}).status_code == 200

    @pytest.mark.parametrize("body", [
        {"compression_level": "3); DROP TABLE sessions; --"},
        {"compression_level": 0},
        {"compression_level": 23},
        {"train_dict_samples_ratio": -1},
        {},
    ])
    def test_update_refuses_bad_values(self, body):
        global server
        server.start()
        res = server.make_request("POST", "/zstd_update_transparent", data=body)
        assert res.status_code == 400, res.body

    def test_valid_request_without_extension_says_how_to_enable(self):
        global server
        server.start()
        res = server.make_request("POST", "/zstd_enable_transparent", data={
            "table": "sessions", "column": "data", "compression_level": 3, "train_dict_samples_ratio": 100})
        assert res.status_code == 501
        assert "--sqlite-zstd-ext-file" in res.body["error"]["message"]


@SQL_SKIP
@pytest.mark.skipif(not ZSTD_EXT, reason="set LLAMA_TEST_SQLITE_ZSTD_EXT to the sqlite-zstd extension library to run end to end")
class TestZstdEndToEnd:
    def setup_method(self):
        global server
        server.sql_save_file = "./tmp/test_sql_zstd_e2e.db"
        server.extra_args = ["--sqlite-zstd-ext-file", ZSTD_EXT]
        # SAFE-DELETE-OK: scratch SQLite file this test class creates itself under tests/tmp (gitignored).
        if os.path.exists(server.sql_save_file):
            os.remove(server.sql_save_file)

    def test_enable_compress_and_roundtrip(self):
        global server
        server.start()
        assert server.make_request("GET", "/version").body["features"]["zstd_compression"] is True
        payload = {"text": "the quick brown fox " * 400}
        for i in range(20):
            assert server.make_request("POST", "/save", data={"key": "s%d" % i, "data": payload}).status_code == 200
        res = server.make_request("POST", "/zstd_enable_transparent", data={
            "table": "sessions", "column": "data", "compression_level": 19, "train_dict_samples_ratio": 100})
        assert res.status_code == 200, res.body
        res = server.make_request("GET", "/zstd_get_configs")
        assert res.status_code == 200 and len(res.body["configs"]) == 1, res.body
        res = server.make_request("POST", "/zstd_incremental_maintenance", data={"duration": None, "db_load": 1.0})
        assert res.status_code == 200, res.body
        # data reads back identically through the transparent decompression view
        for i in (0, 7, 19):
            res = server.make_request("POST", "/load", data={"key": "s%d" % i})
            assert res.status_code == 200 and res.body["result"] == payload
        res = server.make_request("POST", "/zstd_update_transparent", data={"compression_level": 5})
        assert res.status_code == 200 and res.body["updated"] == 1, res.body
