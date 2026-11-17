import os
import pytest
from filelock import FileLock
from utils import *


@pytest.fixture(scope="session", autouse=True)
def configure_worker_port(request):
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    worker_num = 0 if worker_id == "master" else int(worker_id[2:])
    # LLAMA_SERVER_TEST_PORT_BASE decides; without it a PORT the caller set is the base (it used to be THE way to
    # pick the port, and every runner and wrapper of this fork still sets it). Before this fallback PORT was
    # overwritten with 8080, so two test sessions on one host met on that port: on 2026-11-14 a second session's
    # requests reached the first one's server and two of its tests failed.
    port_base = int(os.environ.get("LLAMA_SERVER_TEST_PORT_BASE") or os.environ.get("PORT") or "8080")
    os.environ["PORT"] = str(port_base + worker_num * 10)


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


@pytest.fixture(scope="session", autouse=True)
def load_server_presets(configure_worker_port, tmp_path_factory):
    # this will be run once per test session, before any tests

    # serialize model downloads across parallel workers.
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    with FileLock(str(root_tmp_dir / "load_all.lock")):
        ServerPreset.load_all()
