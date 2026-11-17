#!/usr/bin/env python3
"""Router loaded-set snapshot (--models-snapshot FILE) in the built llama-server, on the CPU.

usage: test-server-models-snapshot.py <llama-server(.exe)> <models dir with model-a/b/c.gguf> <port> <workdir>

Row S69-TORCHSERVE-PICK-ROUTER-LOADED-SET-SNAPSHOT-2026-10-18-01 (TorchServe docs/snapshot.md, Apache-2.0; behaviour only).
Every run uses a fresh <workdir>/run-<time> directory, so nothing from an earlier run is read or has to be deleted.
   A. router run 1: no snapshot file yet; load model-a -> FILE = [model-a]; load model-b -> [model-a, model-b];
      unload model-b -> [model-a]; a graceful stop (Ctrl-C) stops every child and FILE still says [model-a]
      (shutdown must not record its own unloads).
   B. router run 2, same flags: model-a is loaded with no load request, model-b / model-c are not, and the log says so.
   C. explicit preset wins: FILE = [model-a, model-zz, model-b] and a preset with load-on-startup = false for model-a:
      model-b is restored (positive control), model-a stays unloaded and the log names the preset, model-zz is reported
      as not configured.
Before the feature the router refuses the unknown argument, so run A fails at start.
"""
import http.client
import json
import os
import signal
import subprocess
import sys
import time

if len(sys.argv) >= 2 and sys.argv[1] == '--console-host':
    # Windows: a hidden console of our own, so a console stop event reaches the router and nothing else. Starts the
    # router in this console (own process group), waits for <stop file>, then sends it CTRL_BREAK (see below).
    stop_file, rc_file, log_path = sys.argv[2], sys.argv[3], sys.argv[4]
    argv = sys.argv[5:]
    # DIFFERENT-APPROACH: the first version set SIGINT to SIG_IGN here, and the router never stopped on Ctrl-C (all
    # three phases ended killed-after-timeout, with no router clean-up line). Ignoring Ctrl-C in a parent is inherited
    # by its children on Windows, so this host now takes the Ctrl-C itself as KeyboardInterrupt and keeps waiting.
    # DIFFERENT-APPROACH (second): catching KeyboardInterrupt alone still left the router running (pa2: killed-after-
    # timeout, no router clean-up line). The ignore-Ctrl-C attribute is inherited from whatever launched this host
    # (SetConsoleCtrlHandler(NULL, TRUE) state passes to child processes), so restore normal Ctrl-C processing here
    # before the router is started; the router inherits the restored state.
    # DIFFERENT-APPROACH (third): restoring it did not help either. MEASURED (pa5 probe, three arms): the group-wide
    # Ctrl-C reached this host every time and reached NO child -- not the router, and not a plain python child that
    # only waits for KeyboardInterrupt. So nothing router-specific blocks it. The router is now started in a process
    # group of its own and sent CTRL_BREAK, which GenerateConsoleCtrlEvent can aim at one group; the server treats
    # CTRL_BREAK as the same graceful stop as Ctrl-C (server.cpp console_ctrl_handler).
    import ctypes
    restored = ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
    got_ctrl_c = False
    with open(log_path, 'w', encoding='utf-8') as lf:
        p = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        while p.poll() is None and not os.path.exists(stop_file):
            time.sleep(0.2)
        sent = False
        if p.poll() is None:
            try:
                os.kill(p.pid, signal.CTRL_BREAK_EVENT)
                sent = True
            except KeyboardInterrupt:
                got_ctrl_c = True
        rc = None
        deadline = time.time() + 120
        while rc is None and time.time() < deadline:  # PROGRESS-OK: bounded wait, a timeout is reported as such
            try:
                rc = p.wait(timeout=max(0.1, deadline - time.time()))
            except KeyboardInterrupt:
                got_ctrl_c = True
                continue
            except subprocess.TimeoutExpired:
                break
        if rc is None:
            p.kill()
            rc = 'killed-after-timeout' if sent else 'killed-no-ctrl-c'
    with open(rc_file + '.diag', 'w', encoding='utf-8') as f:
        f.write(f'ctrl_handler_restored={restored} sent={sent} host_got_ctrl_c={got_ctrl_c}\n')
    with open(rc_file, 'w', encoding='utf-8') as f:
        f.write(str(rc))
    sys.exit(0)

server_exe, models_dir, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
work = os.path.join(sys.argv[4], time.strftime('run-%Y%m%d_%H%M%S'))
os.makedirs(work, exist_ok=True)
SNAP = os.path.join(work, 'snap.txt')
HEADER = '# llama-server models snapshot v1'
fails, passes = [], 0


def check(cond, msg):
    global passes
    if cond:
        passes += 1
        print('PASS', msg, flush=True)
    else:
        fails.append(msg)
        print('FAIL', msg, flush=True)


def call(method, path, body=None, timeout=30):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    c.request(method, path, body=None if body is None else json.dumps(body), headers={'Content-Type': 'application/json'})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def statuses():
    st, data = call('GET', '/models')
    if st != 200:
        return {}
    return {m['id']: m['status']['value'] for m in json.loads(data)['data']}


def wait_status(name, want, secs=180):
    s = None
    for _ in range(secs * 2):  # PROGRESS-OK: bounded poll, prints the last status on timeout
        try:
            s = statuses().get(name)
        except OSError:
            s = None
        if s == want:
            return True
        time.sleep(0.5)
    print(f'  {name}: last status {s!r}, wanted {want!r}', flush=True)
    return False


def snap_names():
    if not os.path.exists(SNAP):
        return None
    with open(SNAP, encoding='utf-8') as f:
        lines = [x.strip() for x in f.read().splitlines() if x.strip()]
    if not lines or lines[0] != HEADER:
        return ['<bad header>'] + lines
    return lines[1:]


def own_lines(text):
    # child instances' output is relayed with a "[<port>]" prefix
    return [x for x in text.splitlines() if not x.lstrip().startswith('[')]


def wait_snap(want, secs=30):
    got = None
    for _ in range(secs * 5):  # PROGRESS-OK: bounded poll, prints the last value on timeout
        got = snap_names()
        if got == want:
            return True
        time.sleep(0.2)
    print(f'  snapshot is {got!r}, wanted {want!r}', flush=True)
    return False


class Router:
    def __init__(self, tag, extra):
        self.log = os.path.join(work, f'router_{tag}.log')
        self.stop_file = os.path.join(work, f'stop_{tag}')
        self.rc_file = os.path.join(work, f'rc_{tag}')
        argv = [server_exe, '--models-dir', models_dir, '--models-snapshot', SNAP, '--models-max', '3',
                '--port', str(port), '--host', '127.0.0.1', '--no-webui', '-c', '512', '-t', '2'] + extra
        if os.name == 'nt':
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0  # SW_HIDE
            self.host = subprocess.Popen([sys.executable, os.path.abspath(__file__), '--console-host', self.stop_file,
                                          self.rc_file, self.log] + argv,
                                         creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=si)
        else:
            self.lf = open(self.log, 'w', encoding='utf-8')
            self.host = subprocess.Popen(argv, stdout=self.lf, stderr=subprocess.STDOUT)

    def up(self, secs=60):
        for _ in range(secs * 2):  # PROGRESS-OK: bounded health poll
            if self.exited():
                return False
            try:
                if call('GET', '/health', timeout=2)[0] == 200:
                    return True
            except OSError:
                pass
            time.sleep(0.5)
        return False

    def exited(self):
        if os.name == 'nt':
            return os.path.exists(self.rc_file)
        return self.host.poll() is not None

    def stop(self):
        if os.name == 'nt':
            open(self.stop_file, 'w').close()
            self.host.wait(timeout=180)
            if os.path.exists(self.rc_file + '.diag'):
                with open(self.rc_file + '.diag', encoding='utf-8') as f:
                    print('  console host:', f.read().strip(), flush=True)
            with open(self.rc_file, encoding='utf-8') as f:
                return f.read().strip()
        self.host.send_signal(signal.SIGINT)
        rc = self.host.wait(timeout=120)
        self.lf.close()
        return str(rc)

    def text(self):
        with open(self.log, encoding='utf-8', errors='replace') as f:
            return f.read()


print('workdir', work, flush=True)

# ---- A ----
print('== A: record', flush=True)
r = Router('a', [])
ok = r.up()
check(ok, 'A router up (the build accepts --models-snapshot)')
if ok:
    check(snap_names() is None, 'A control: no snapshot file before any load')
    st0 = statuses()
    check(st0.get('model-a') == 'unloaded' and st0.get('model-b') == 'unloaded', f'A models listed unloaded {st0}')
    call('POST', '/models/load', {'model': 'model-a'})
    check(wait_status('model-a', 'loaded'), 'A model-a loaded')
    check(wait_snap(['model-a']), 'A snapshot = [model-a] after its load')
    call('POST', '/models/load', {'model': 'model-b'})
    check(wait_status('model-b', 'loaded'), 'A model-b loaded')
    check(wait_snap(['model-a', 'model-b']), 'A snapshot = [model-a, model-b]')
    call('POST', '/models/unload', {'model': 'model-b'})
    check(wait_status('model-b', 'unloaded'), 'A model-b unloaded')
    check(wait_snap(['model-a']), 'A snapshot = [model-a] after the unload')
    rc_a = r.stop()
    print(f'  router A exit {rc_a}', flush=True)
    check(rc_a == '0', f'A router exited 0 on the graceful stop (got {rc_a})')
    # the router's own lines carry no "[port]" prefix; a child instance's "cleaning up" line must not count
    check(any('cleaning up before exit' in x for x in own_lines(r.text())),
          'A graceful stop ran the ROUTER clean-up (unload_all), not only a child\'s')
    check(snap_names() == ['model-a'], f'A snapshot survives the shutdown unloads: {snap_names()}')
else:
    print(r.text()[-2000:], flush=True)
    try:
        print(f'  router A exit {r.stop()}', flush=True)
    except Exception as e:  # SWALLOW-OK: the router never came up; the failure is already recorded above
        print('  stop after failed start:', e, flush=True)

# ---- B ----
if not fails:
    print('== B: restore', flush=True)
    r = Router('b', [])
    check(r.up(), 'B router up')
    check(wait_status('model-a', 'loaded'), 'B model-a restored without a load request')
    st = statuses()
    check(st.get('model-b') == 'unloaded' and st.get('model-c') == 'unloaded', f'B nothing else loaded {st}')
    rc_b = r.stop()
    print(f'  router B exit {rc_b}', flush=True)
    check(rc_b == '0', f'B router exited 0 on the graceful stop (got {rc_b})')
    check('models snapshot: restoring model-a' in r.text(), 'B log names the restored model')

# ---- C ----
if not fails:
    print('== C: explicit preset wins', flush=True)
    with open(SNAP, 'w', encoding='utf-8') as f:
        f.write(HEADER + '\nmodel-a\nmodel-zz\nmodel-b\n')
    ini = os.path.join(work, 'presets.ini')
    with open(ini, 'w', encoding='utf-8') as f:
        f.write('[model-a]\nload-on-startup = false\n')
    r = Router('c', ['--models-preset', ini])
    check(r.up(), 'C router up')
    check(wait_status('model-b', 'loaded'), 'C model-b restored (positive control)')
    time.sleep(3)
    st = statuses()
    check(st.get('model-a') == 'unloaded', f'C model-a NOT loaded: its preset says load-on-startup = false ({st})')
    rc_c = r.stop()
    print(f'  router C exit {rc_c}', flush=True)
    check(rc_c == '0', f'C router exited 0 on the graceful stop (got {rc_c})')
    t = r.text()
    check('models snapshot: model-a skipped, its preset sets' in t, 'C log says the preset won for model-a')
    check('models snapshot: model-zz skipped, no such model is configured now' in t, 'C log reports the unknown model-zz')

print(f'{passes} passed, {len(fails)} failed', flush=True)
for f_ in fails:
    print('  FAILED:', f_, flush=True)
sys.exit(1 if fails else 0)
