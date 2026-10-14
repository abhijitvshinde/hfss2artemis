"""h2a: HFSS -> ARTEMIS in one tool.  Design in HFSS, run on the GPU, watch progress, get S-parameters.

  python h2a.py export  DESIGN.aedt [--design NAME] [-o models/NAME.json]   HFSS project -> model.json
  python h2a.py build   models/NAME.json --dx 20 --dy 20 --dz 5 --ns 3 [-o runs/NAME]   model -> ARTEMIS deck
  python h2a.py run     [RUN]  [--cpu] [--ranks N]     start ARTEMIS in the background (survives closing VS Code)
  python h2a.py status  [RUN]  [--watch]               step, %, speed, ETA, errors, GPU load
  python h2a.py stop    [RUN]                          stop a running simulation (checkpoints are kept)
  python h2a.py resume  [RUN]                          continue from the newest checkpoint
  python h2a.py sparams [RUN]                          S11/S21 -> RUN/sparams.csv + sparams.png
  python h2a.py list                                   all runs and their state
  python h2a.py go      DESIGN.aedt --dx .. --dy .. --dz .. --ns .. [--name NAME]    export + build + run

RUN is a folder under runs/ (name or path); leave it out to use the most recently active run.
Windows: ARTEMIS is started inside WSL (Ubuntu). Linux / HPC: it is started directly.
Machine-specific paths (ARTEMIS binaries, WSL distro, MPI ranks) live in h2a_config.json.
"""
import argparse
import datetime as _dt
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
MODELS = os.path.join(HERE, "models")
IS_WIN = os.name == "nt"
STATE = "h2a_state.json"
NOWIN = {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WIN else {}   # no console flashes from the GUI

DEFAULT_CONFIG = {
    "wsl_distro": "Ubuntu",
    "gpu_bin": "~/artemis_src/artemis/build_gpu/bin/warpx.3d.MPI.CUDA.DP.PDP.LLG",
    "cpu_bin": "~/artemis_src/artemis/build/bin/warpx.3d.MPI.OMP.DP.PDP.LLG",
    "gpu_ranks": 1,
    "cpu_ranks": 4,
    "omp_threads": 2,
    "keep_checkpoints": 2,
}


# ----------------------------------------------------------------------------- helpers
def config():
    path = os.path.join(HERE, "h2a_config.json")
    if not os.path.isfile(path):
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=1)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(json.load(open(path)))
    return cfg


def to_linux(path):
    """Windows path -> path ARTEMIS sees (WSL /mnt/<drive>/...). Unchanged on Linux."""
    path = os.path.abspath(path)
    if not IS_WIN:
        return path
    drive, rest = os.path.splitdrive(path)
    return "/mnt/" + drive.rstrip(":").lower() + rest.replace("\\", "/")


def shell(cmd, cfg, capture=True, **kw):
    """Run a bash command in Linux (WSL on Windows)."""
    # --exec: no extra shell between wsl.exe and bash, so quotes and $(...) arrive intact
    argv = (["wsl", "-d", cfg["wsl_distro"], "--exec", "bash", "-lc", cmd] if IS_WIN else ["bash", "-lc", cmd])
    kw = {**NOWIN, **kw}
    if capture:
        return subprocess.run(argv, capture_output=True, text=True, **kw)
    return subprocess.Popen(argv, **kw)


def resolve_run(name):
    """Run folder from a name/path, or the most recently active run."""
    if name:
        for cand in (name, os.path.join(RUNS, name)):
            if os.path.isfile(os.path.join(cand, "inputs")):
                return os.path.abspath(cand)
        sys.exit(f"no ARTEMIS run folder '{name}' (looked in . and {RUNS})")
    runs = [d for d in glob.glob(os.path.join(RUNS, "*")) if os.path.isfile(os.path.join(d, "inputs"))]
    if not runs:
        sys.exit(f"no runs in {RUNS}; create one with 'h2a.py build' or 'h2a.py go'")

    def activity(d):
        files = [os.path.join(d, f) for f in ("run.log", "inputs", STATE)]
        return max(os.path.getmtime(f) for f in files if os.path.exists(f))
    return max(runs, key=activity)


def load_state(run):
    p = os.path.join(run, STATE)
    return json.load(open(p)) if os.path.isfile(p) else {}


def save_state(run, **kw):
    st = load_state(run)
    st.update(kw)
    tmp = os.path.join(run, STATE + ".tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(run, STATE))
    return st


def pid_alive(pid):
    if not pid:
        return False
    if IS_WIN:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True,
                             **NOWIN).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def artemis_pids(run, cfg):
    """PIDs of ARTEMIS processes whose working directory is this run folder."""
    d = to_linux(run)
    cmd = (f'd=$(readlink -f "{d}"); for p in $(pgrep -f "warpx\\.3d"); do '
           f'[ "$(readlink -f /proc/$p/cwd 2>/dev/null)" = "$d" ] && echo $p; done; true')
    return [int(x) for x in shell(cmd, cfg).stdout.split()]


def any_artemis_running(cfg):
    return shell('pgrep -f "warpx\\.3d" >/dev/null && echo yes || echo no', cfg).stdout.strip() == "yes"


def tail(path, nbytes=400_000):
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode(errors="ignore")


def checkpoints(run):
    """Complete checkpoints (WarpX writes WarpXHeader last), oldest first."""
    cps = [d for d in glob.glob(os.path.join(run, "diags", "chk*"))
           if re.fullmatch(r"chk\d+", os.path.basename(d)) and os.path.isfile(os.path.join(d, "WarpXHeader"))]
    return sorted(cps, key=lambda d: int(os.path.basename(d)[3:]))


def max_step(run):
    m = re.search(r"^max_step\s*=\s*(\d+)", open(os.path.join(run, "inputs")).read(), re.M)
    return int(m.group(1)) if m else None


def fmt_dur(sec):
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


# ----------------------------------------------------------------------------- progress
def progress(run):
    """Parse the end of run.log: last step, sim time, s/step, errors."""
    txt = tail(os.path.join(run, "run.log"))
    steps = re.findall(r"STEP (\d+) ends\. TIME = ([0-9.eE+-]+)", txt)
    avg = re.findall(r"Avg\. per step = ([0-9.eE+-]+) s", txt)
    errs = [ln.strip() for ln in txt.splitlines()
            if re.search(r"Abort|ERROR|Segfault|Segmentation|CUDA error|out of memory|red flag", ln)
            and "Abort is in progress" not in ln]
    done = "Total Time" in txt or bool(re.search(r"AMReX \(.*\) finalized", txt))
    return {"step": int(steps[-1][0]) if steps else 0, "t_sim": float(steps[-1][1]) if steps else 0.0,
            "sec_per_step": float(avg[-1]) if avg else None, "errors": errs[-3:], "finished_log": done}


def gpu_line():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    r = subprocess.run([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, **NOWIN)
    if r.returncode or not r.stdout.strip():
        return None
    name, util, used, tot, temp = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    return f"{name}: {util}% busy, {int(used)/1024:.1f}/{int(tot)/1024:.1f} GB, {temp} C"


def run_state(run, cfg):
    st, pr = load_state(run), progress(run)
    ms = max_step(run)
    alive = pid_alive(st.get("watcher_pid")) if st.get("state") == "running" else False
    state = st.get("state", "not started" if not os.path.isfile(os.path.join(run, "run.log")) else "unknown")
    if state == "running" and st.get("stop_requested"):
        state = "stopping" if alive else "stopped"
    elif state == "running" and not alive:
        state = "stopped unexpectedly"
    if state in ("unknown", "stopped unexpectedly") and artemis_pids(run, cfg):
        state = "running (started outside h2a)"
    return state, pr, ms, st


def cmd_status(a):
    cfg = config()
    run = resolve_run(a.run)
    while True:
        state, pr, ms, st = run_state(run, cfg)
        pct = 100.0 * pr["step"] / ms if ms else 0.0
        lines = [f"run      : {run}", f"state    : {state}"]
        lines.append(f"progress : step {pr['step']:,} of {ms:,} ({pct:.1f}%), {pr['t_sim']*1e9:.3f} ns simulated")
        if pr["sec_per_step"]:
            left = (ms - pr["step"]) * pr["sec_per_step"]
            eta = (_dt.datetime.now() + _dt.timedelta(seconds=left)).strftime("%a %H:%M")
            lines.append(f"speed    : {pr['sec_per_step']:.3f} s/step, "
                         + (f"about {fmt_dur(left)} left (done ~{eta})" if state.startswith("running") else
                            f"{fmt_dur(left)} of work left"))
        if st.get("started"):
            lines.append(f"started  : {st['started']}" + (f"  (resumed from {st['resumed_from']})" if st.get("resumed_from") else ""))
        if st.get("ended"):
            lines.append(f"ended    : {st['ended']}  exit code {st.get('exit_code')}")
        cps = checkpoints(run)
        lines.append(f"restart  : {os.path.basename(cps[-1]) if cps else 'no checkpoint yet'}")
        red = os.path.join(run, "diags", "reducedfiles")
        nsamp = sum(1 for _ in open(os.path.join(red, "V_port1.txt"))) - 1 if os.path.isfile(os.path.join(red, "V_port1.txt")) else 0
        nslice = len(glob.glob(os.path.join(run, "diags", "slice*")))
        lines.append(f"outputs  : {nsamp} port samples, {nslice} field slices")
        g = gpu_line()
        if g:
            lines.append(f"GPU      : {g}")
        for e in pr["errors"]:
            lines.append(f"PROBLEM  : {e}")
        if state == "stopped unexpectedly" or pr["errors"]:
            lines.append("hint     : see the end of run.log; 'h2a.py resume' continues from the last checkpoint")
        if a.watch:
            os.system("cls" if IS_WIN else "clear")
        print("\n".join(lines))
        if not a.watch or not state.startswith("running"):
            break
        time.sleep(a.every)


# ----------------------------------------------------------------------------- run / watch / stop
def launch(run, restart_from=None, cpu=False, ranks=None):
    cfg = config()
    if any_artemis_running(cfg):
        sys.exit("another ARTEMIS simulation is already running on this machine ('h2a.py list' / 'h2a.py stop')")
    st = save_state(run, state="starting", cpu=cpu, ranks=ranks or (cfg["cpu_ranks"] if cpu else cfg["gpu_ranks"]),
                    resumed_from=os.path.basename(restart_from) if restart_from else None,
                    stop_requested=False, ended=None, exit_code=None)
    log = open(os.path.join(run, "h2a_watcher.log"), "a")
    kw = {"stdout": log, "stderr": log, "stdin": subprocess.DEVNULL, "cwd": HERE}
    if IS_WIN:
        kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    argv = [sys.executable, os.path.abspath(__file__), "_watch", run]
    if restart_from:
        argv += ["--restart", restart_from]
    p = subprocess.Popen(argv, **kw)
    for _ in range(40):                                  # wait until ARTEMIS writes its first step
        time.sleep(1.5)
        if progress(run)["step"] > 0 or p.poll() is not None:
            break
    pr = progress(run)
    if p.poll() is not None and load_state(run).get("state") != "running":
        print(tail(os.path.join(run, "run.log"), 3000))
        sys.exit("ARTEMIS stopped right away; see run.log above")
    print(f"started {'CPU' if cpu else 'GPU'} run in the background: {run}")
    print(f"  now at step {pr['step']}; check with:  python h2a.py status   (or  status --watch)")
    print("  closing VS Code/terminal is fine; the laptop must stay on (plugged in, lid open)")
    return st


def cmd_run(a):
    run = resolve_run(a.run)
    if any_artemis_running(config()):      # check before touching any files of a possibly running run
        sys.exit("an ARTEMIS simulation is already running on this machine ('h2a.py list' / 'h2a.py stop')")
    if os.path.isfile(os.path.join(run, "run.log")) and not a.fresh and checkpoints(run):
        sys.exit(f"{run} already has checkpoints; use 'h2a.py resume' to continue or 'run --fresh' to start over")
    if a.fresh:
        for d in glob.glob(os.path.join(run, "diags", "*")):
            shutil.rmtree(d, ignore_errors=True)
    launch(run, cpu=a.cpu, ranks=a.ranks)


def cmd_resume(a):
    run = resolve_run(a.run)
    cps = checkpoints(run)
    if not cps:
        sys.exit("no checkpoint to resume from; start over with 'h2a.py run --fresh'")
    st = load_state(run)
    launch(run, restart_from=cps[-1], cpu=a.cpu if a.cpu else st.get("cpu", False), ranks=a.ranks)


def cmd_watch(a):
    """Internal: owns one ARTEMIS run, keeps the PC awake, prunes old checkpoints, records the outcome."""
    run, cfg = os.path.abspath(a.run), config()
    st = load_state(run)
    cpu = st.get("cpu", False)
    exe = cfg["cpu_bin"] if cpu else cfg["gpu_bin"]
    ranks = st.get("ranks") or (cfg["cpu_ranks"] if cpu else cfg["gpu_ranks"])
    extra = f" amr.restart={to_linux(a.restart)}" if a.restart else ""
    redirect = ">>" if a.restart else ">"
    cmd = (f'cd "{to_linux(run)}" && export OMP_NUM_THREADS={cfg["omp_threads"]} && '
           f'exec mpirun -n {ranks} {exe} inputs{extra} {redirect} run.log 2>&1')
    if IS_WIN:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS|ES_SYSTEM_REQUIRED
    proc = shell(cmd, cfg, capture=False, stdin=subprocess.DEVNULL)
    save_state(run, state="running", watcher_pid=os.getpid(), cmd=cmd,
               started=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    tick = 0
    while proc.poll() is None:
        if tick % 6 == 0:
            for old in checkpoints(run)[:-cfg["keep_checkpoints"]]:
                shutil.rmtree(old, ignore_errors=True)
        tick += 1
        time.sleep(5)
    for old in checkpoints(run)[:-cfg["keep_checkpoints"]]:
        shutil.rmtree(old, ignore_errors=True)
    rc = proc.returncode
    pr, st = progress(run), load_state(run)
    if st.get("stop_requested"):
        state = "stopped"
    elif rc == 0 and pr["step"] >= (max_step(run) or 0):
        state = "finished"
    else:
        state = "failed"
    save_state(run, state=state, exit_code=rc, ended=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def cmd_stop(a):
    cfg = config()
    run = resolve_run(a.run)
    pids = artemis_pids(run, cfg)
    if not pids:
        state, pr, ms, _ = run_state(run, cfg)
        print(f"nothing running in {run} (state: {state}, step {pr['step']:,} of {ms:,})")
        return
    save_state(run, stop_requested=True)
    shell("kill " + " ".join(map(str, pids)), cfg)
    for _ in range(20):
        time.sleep(1)
        if not artemis_pids(run, cfg):
            break
    pr = progress(run)
    cps = checkpoints(run)
    print(f"stopped at step {pr['step']:,}; " +
          (f"'h2a.py resume' continues from {os.path.basename(cps[-1])}" if cps else "no checkpoint yet, a rerun starts from 0"))


# ----------------------------------------------------------------------------- list / export / build / sparams / go
def cmd_list(a):
    cfg = config()
    runs = sorted(d for d in glob.glob(os.path.join(RUNS, "*")) if os.path.isfile(os.path.join(d, "inputs")))
    if not runs:
        print(f"no runs in {RUNS}")
        return
    print(f"{'run':24s} {'state':24s} {'progress':>18s}  last activity")
    for d in runs:
        state, pr, ms, _ = run_state(d, cfg)
        log = os.path.join(d, "run.log")
        when = _dt.datetime.fromtimestamp(os.path.getmtime(log)).strftime("%Y-%m-%d %H:%M") if os.path.exists(log) else "-"
        pct = f"{pr['step']:,}/{ms:,} {100*pr['step']/ms:5.1f}%" if ms else "-"
        print(f"{os.path.basename(d):24s} {state:24s} {pct:>18s}  {when}")


def cmd_export(a, extra):
    import hfss_export
    out = a.out or os.path.join(MODELS, os.path.splitext(os.path.basename(a.aedt))[0] + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    hfss_export.main([a.aedt, "-o", out] + (["--design", a.design] if a.design else []) + extra)
    return out


def cmd_build(a, extra):
    import build_inputs
    out = a.out or os.path.join(RUNS, os.path.splitext(os.path.basename(a.model))[0])
    if os.path.isfile(os.path.join(out, "run.log")) and not a.force:
        sys.exit(f"{out} already holds a run; pick another -o/--name or add --force")
    build_inputs.main([a.model, "-o", out] + extra)
    print(f"\nnext:  check {os.path.join(out, 'geometry_preview.png')}, then  python h2a.py run {os.path.basename(out)}")
    return out


def cmd_sparams(a):
    import sparams
    sparams.main([resolve_run(a.run)])


def cmd_go(a, extra):
    name = a.name or os.path.splitext(os.path.basename(a.aedt))[0]
    model = cmd_export(argparse.Namespace(aedt=a.aedt, design=a.design, out=os.path.join(MODELS, name + ".json")), [])
    out = cmd_build(argparse.Namespace(model=model, out=os.path.join(RUNS, name), force=a.force), extra)
    if a.no_run:
        return
    launch(out, cpu=a.cpu)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="HFSS .aedt -> model.json")
    p.add_argument("aedt")
    p.add_argument("--design")
    p.add_argument("-o", "--out")
    p = sub.add_parser("build", help="model.json -> ARTEMIS run folder (extra options go to build_inputs.py)")
    p.add_argument("model")
    p.add_argument("-o", "--out")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("go", help="export + build + run in one step")
    p.add_argument("aedt")
    p.add_argument("--design")
    p.add_argument("--name")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--force", action="store_true")
    for name, hlp in (("run", "start a simulation in the background"), ("resume", "continue from the newest checkpoint")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("run", nargs="?")
        p.add_argument("--cpu", action="store_true", help="use the CPU build instead of the GPU")
        p.add_argument("--ranks", type=int, help="MPI ranks (default from h2a_config.json)")
        if name == "run":
            p.add_argument("--fresh", action="store_true", help="delete old outputs and start from step 0")
    p = sub.add_parser("status", help="progress of a run")
    p.add_argument("run", nargs="?")
    p.add_argument("--watch", action="store_true", help="refresh until the run ends (Ctrl+C to quit)")
    p.add_argument("--every", type=int, default=60, help="refresh interval for --watch (s)")
    for name, hlp in (("stop", "stop a running simulation"), ("sparams", "compute S-parameters")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("run", nargs="?")
    sub.add_parser("list", help="all runs")
    p = sub.add_parser("_watch")
    p.add_argument("run")
    p.add_argument("--restart")

    a, extra = ap.parse_known_args()
    if extra and a.cmd not in ("build", "go"):
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    sys.path.insert(0, HERE)
    {"export": lambda: cmd_export(a, extra), "build": lambda: cmd_build(a, extra), "go": lambda: cmd_go(a, extra),
     "run": lambda: cmd_run(a), "resume": lambda: cmd_resume(a), "status": lambda: cmd_status(a),
     "stop": lambda: cmd_stop(a), "sparams": lambda: cmd_sparams(a), "list": lambda: cmd_list(a),
     "_watch": lambda: cmd_watch(a)}[a.cmd]()


if __name__ == "__main__":
    main()
