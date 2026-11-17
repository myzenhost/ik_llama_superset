import os
import pytest
from utils import *

# G3: SQL session store, ported from the pre-merge fork's examples/server/server.cpp
# (row S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01).
# SOURCE-READ: examples/server/server.cpp:43-54, 570-571, 1991-2096, 2225-2235 @ 35f0cac75.
#
# This feature is compiled in only when the server binary under test was built with
# -DLLAMA_SERVER_SQLITE3=ON (which defines SQLITE3_MODERN_CPP_SUPPORT). The 7 routes are always
# mounted regardless (see server.cpp), so test_sql_routes_gated_without_sql_save_file below passes
# against EITHER binary: without --sql-save-file, every route refuses with a 501 whose message
# differs (compiled-but-not-configured vs not-compiled-at-all) but whose status code does not.
# Every other test here needs the SQLITE3-enabled binary AND --sql-save-file, so it is skipped
# (not failed) against a binary built without the option -- see sql_supported() below.

server = ServerPreset.tinyllama2()


def sql_supported() -> bool:
    """True if this binary answers /version with sql: true once --sql-save-file is set.

    Cheaper and more direct than parsing a CMake cache or grepping the binary: start a server with
    --sql-save-file pointed at a scratch file and ask it. Cached per test session isn't worth the
    complexity here -- this runs once per test via the fixture below, and a real server start is
    already what every other test in this suite pays for its assertions.
    """
    global server
    server.sql_save_file = "./tmp/sql_supported_probe.db"
    server.start()
    res = server.make_request("GET", "/version")
    supported = res.status_code == 200 and res.body.get("features", {}).get("sql") is True
    server.stop()
    return supported


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    # tools/server/tests/.gitignore excludes tmp/, so a fresh checkout has none (same reasoning as
    # test_slot_prompts.py's own fixture for --slot-save-path).
    os.makedirs("tmp", exist_ok=True)


def test_sql_routes_gated_without_sql_save_file():
    # all 7 G3 routes must refuse with 501 when --sql-save-file was not given, regardless of
    # whether this binary was even built with -DLLAMA_SERVER_SQLITE3=ON.
    global server
    server.start()

    res = server.make_request("POST", "/load", data={"key": "x"})
    assert res.status_code == 501
    res = server.make_request("POST", "/save", data={"key": "x", "data": "y"})
    assert res.status_code == 501
    res = server.make_request("POST", "/rename", data={"key": "x", "newName": "y"})
    assert res.status_code == 501
    res = server.make_request("POST", "/all", data={})
    assert res.status_code == 501
    res = server.make_request("GET", "/sessions")
    assert res.status_code == 501
    res = server.make_request("POST", "/sessions", data={})
    assert res.status_code == 501
    res = server.make_request("POST", "/delete", data={"key": "x"})
    assert res.status_code == 501
    res = server.make_request("GET", "/vacuum")
    assert res.status_code == 501

    # /version must report the feature as unavailable either way.
    res = server.make_request("GET", "/version")
    assert res.status_code == 200
    assert res.body["features"]["sql"] is False


@pytest.mark.skipif(not sql_supported(), reason=(
    "this server binary was not built with -DLLAMA_SERVER_SQLITE3=ON (row "
    "S69-FORK-SERVER-FEATURES-STRANDED-IN-UNBUILT-EXAMPLES-SERVER-2026-09-27-01, group G3); "
    "build build (or any build configured the same way) and point "
    "LLAMA_SERVER_BIN_PATH at its bin/llama-server(.exe) to run the rest of this file"
))
class TestSqlSessionStore:
    def setup_method(self):
        global server
        server.sql_save_file = "./tmp/test_sql_sessions.db"
        # a fresh db per test avoids cross-test key collisions without adding a reset route.
        # SAFE-DELETE-OK: this exact path is a scratch SQLite file this test class itself creates
        # and repopulates every run (tools/server/tests/.gitignore excludes tmp/ entirely) -- never
        # a path supplied by the user or containing any data from outside this test file.
        if os.path.exists(server.sql_save_file):
            os.remove(server.sql_save_file)

    def test_version_reports_sql_true(self):
        global server
        server.start()
        res = server.make_request("GET", "/version")
        assert res.status_code == 200
        assert res.body["features"]["sql"] is True
        assert res.body["features"]["zstd_compression"] is False  # no --sqlite-zstd-ext-file given (G4: test_sql_zstd.py)

    def test_save_load_roundtrip_default_store(self):
        global server
        server.start()

        res = server.make_request("POST", "/save", data={"key": "session-1", "data": {"messages": ["hi"]}})
        assert res.status_code == 200
        assert res.body["ok"] is True

        res = server.make_request("POST", "/load", data={"key": "session-1"})
        assert res.status_code == 200
        assert res.body["ok"] is True
        assert res.body["result"] == {"messages": ["hi"]}

    def test_load_missing_key_404(self):
        global server
        server.start()
        res = server.make_request("POST", "/load", data={"key": "does-not-exist"})
        assert res.status_code == 404

    def test_save_overwrites_existing_key(self):
        global server
        server.start()
        server.make_request("POST", "/save", data={"key": "k", "data": {"v": 1}})
        res = server.make_request("POST", "/save", data={"key": "k", "data": {"v": 2}})
        assert res.status_code == 200
        res = server.make_request("POST", "/load", data={"key": "k"})
        assert res.body["result"] == {"v": 2}

    def test_names_store_round_trips_raw_string_not_json(self):
        # the `names` table holds session display names as a raw string (SOURCE-READ:
        # old server.cpp:2056, 2049) -- unlike every other store, /save does not json-dump it and
        # /load does not json-parse it back.
        global server
        server.start()
        res = server.make_request("POST", "/save", data={
            "storeName": "names", "key": "session-1", "data": "My First Chat",
        })
        assert res.status_code == 200

        res = server.make_request("POST", "/load", data={"storeName": "names", "key": "session-1"})
        assert res.status_code == 200
        assert res.body["result"] == "My First Chat"

    def test_unknown_store_name_rejected(self):
        global server
        server.start()
        res = server.make_request("POST", "/save", data={"storeName": "bogus", "key": "k", "data": "v"})
        assert res.status_code == 400
        res = server.make_request("POST", "/load", data={"storeName": "bogus", "key": "k"})
        assert res.status_code == 400
        res = server.make_request("POST", "/all", data={"storeName": "bogus"})
        assert res.status_code == 400
        res = server.make_request("POST", "/delete", data={"storeName": "bogus", "key": "k"})
        assert res.status_code == 400

    def test_all_lists_every_entry_in_a_store(self):
        global server
        server.start()
        server.make_request("POST", "/save", data={"key": "a", "data": {"n": 1}})
        server.make_request("POST", "/save", data={"key": "b", "data": {"n": 2}})

        res = server.make_request("POST", "/all", data={})
        assert res.status_code == 200
        assert res.body["result"] == {"a": {"n": 1}, "b": {"n": 2}}

    def test_all_on_names_store_does_not_json_parse(self):
        # regression test for a bug found while porting: the fork's own handle_all called
        # json::parse() unconditionally, which would throw on a plain display name in the `names`
        # store (see the comment on post_sql_all in server-context.cpp).
        global server
        server.start()
        server.make_request("POST", "/save", data={"storeName": "names", "key": "s1", "data": "Plain Name With Spaces"})

        res = server.make_request("POST", "/all", data={"storeName": "names"})
        assert res.status_code == 200
        assert res.body["result"] == {"s1": "Plain Name With Spaces"}

    def test_sessions_get_and_post_reflect_names_store(self):
        global server
        server.start()
        server.make_request("POST", "/save", data={"storeName": "names", "key": "s1", "data": "Chat One"})

        res = server.make_request("GET", "/sessions")
        assert res.status_code == 200
        assert res.body["result"] == {"s1": "Chat One"}

        res = server.make_request("POST", "/sessions", data={})
        assert res.status_code == 200
        assert res.body["result"] == {"s1": "Chat One"}

    def test_rename_updates_names_store(self):
        global server
        server.start()
        server.make_request("POST", "/save", data={"storeName": "names", "key": "s1", "data": "Old Name"})

        res = server.make_request("POST", "/rename", data={"key": "s1", "newName": "New Name"})
        assert res.status_code == 200
        assert res.body["ok"] is True

        res = server.make_request("GET", "/sessions")
        assert res.body["result"] == {"s1": "New Name"}

    def test_rename_missing_key_404(self):
        # regression test for a silent-no-op bug found while porting: the fork's own handle_rename
        # ran an UPDATE with no existence check, so renaming a key that was never saved answered
        # "Session renamed successfully" despite touching 0 rows.
        global server
        server.start()
        res = server.make_request("POST", "/rename", data={"key": "never-saved", "newName": "x"})
        assert res.status_code == 404

    def test_delete_removes_key(self):
        global server
        server.start()
        server.make_request("POST", "/save", data={"key": "k", "data": {"v": 1}})

        res = server.make_request("POST", "/delete", data={"key": "k"})
        assert res.status_code == 200
        assert res.body["ok"] is True

        res = server.make_request("POST", "/load", data={"key": "k"})
        assert res.status_code == 404

    def test_delete_missing_key_404(self):
        # regression test for the same silent-no-op shape as rename above (old server.cpp:2085-2089).
        global server
        server.start()
        res = server.make_request("POST", "/delete", data={"key": "never-saved"})
        assert res.status_code == 404

    def test_vacuum_ok(self):
        global server
        server.start()
        res = server.make_request("GET", "/vacuum")
        assert res.status_code == 200
        assert res.body["ok"] is True

    def test_key_can_be_a_json_value_not_just_a_string(self):
        # SOURCE-READ: old server.cpp:2035-2037 (get_key_string) -- a non-string JSON key is
        # re-dumped to a string and used as the literal sqlite key.
        global server
        server.start()
        res = server.make_request("POST", "/save", data={"key": 42, "data": {"v": 1}})
        assert res.status_code == 200
        res = server.make_request("POST", "/load", data={"key": 42})
        assert res.status_code == 200
        assert res.body["result"] == {"v": 1}
