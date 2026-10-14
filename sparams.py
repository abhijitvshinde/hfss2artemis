"""S-parameters from an ARTEMIS run made with build_inputs.py.

    python sparams.py run_dir            # reads run_dir/ports.json + run_dir/diags/reducedfiles/*.txt

Port k voltage V_k(t) = (volume integral of E over the port sheet) / (port cross-section). The driven port is a
Norton source of known current Is(t) in parallel with Z0, every port is terminated in Z0, so
    S11 = 2 V1(f) / (Z0 Is(f)) - 1        Sk1 = 2 Vk(f) / (Z0 Is(f))
Writes run_dir/sparams.csv and run_dir/sparams.png (time signals + |S| in dB).
"""
import argparse
import json
import os

import numpy as np


def port_voltage(path, axis, area):
    """RawEFieldReduction file (step, time, int Ex dV, int Ey dV, int Ez dV) -> (t, V = int E dV / A)."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 1], data[:, 2 + "xyz".index(axis)] / area


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="run folder (default: most recently updated folder in ./runs next to this script)")
    ap.add_argument("--pad", type=int, default=8, help="zero-padding factor for the FFT")
    args = ap.parse_args(argv)
    if args.run_dir is None:
        runs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        cands = [os.path.join(runs, d) for d in os.listdir(runs)
                 if os.path.isfile(os.path.join(runs, d, "diags", "reducedfiles", "V_port1.txt"))]
        if not cands:
            print(f"no finished or running ARTEMIS runs found in {runs}")
            return
        args.run_dir = max(cands, key=lambda d: os.path.getmtime(os.path.join(d, "diags", "reducedfiles", "V_port1.txt")))
        print("using", args.run_dir)

    meta = json.load(open(os.path.join(args.run_dir, "ports.json")))
    red = os.path.join(args.run_dir, "diags", "reducedfiles")
    ports = meta["ports"]
    drv = next(p for p in ports if p["driven"])
    Z0, tau, t0, f0 = drv["Z0"], meta["tau"], meta["t0"], meta["f0"]

    sig = {}
    for p in ports:
        t, v = port_voltage(os.path.join(red, p["probe"] + ".txt"), p["axis"], p["A"])
        sig[p["name"]] = (t, v)
    t = sig[drv["name"]][0]
    if len(t) < 16:
        last = 0
        log = os.path.join(args.run_dir, "run.log")
        if os.path.isfile(log):
            with open(log, errors="ignore") as fh:
                for line in fh:
                    if line.startswith("STEP ") and " ends." in line:
                        last = int(line.split()[1])
        print(f"only {len(t)} port sample(s) so far (step {last} of {meta['max_step']}, "
              f"{100 * last / meta['max_step']:.1f}%); run again later")
        return
    Is_amp = meta["Is_amp"]
    if not meta.get("Is_includes_sigma_gain"):
        # decks built before 2026-10-07 assumed J = eps*f/dt; the real soft-source current through the
        # resistive port sheet is (eps/dt + sigma)*f, i.e. larger by 1 + sigma*dt/eps0
        Is_amp *= 1 + drv["sigma"] * meta["dt"] / 8.8541878128e-12
    Is = Is_amp * np.exp(-(t - t0) ** 2 / (2 * tau ** 2)) * np.sin(2 * np.pi * f0 * (t - t0))

    dt_s = np.median(np.diff(t))
    nfft = int(2 ** np.ceil(np.log2(len(t) * args.pad)))
    f = np.fft.rfftfreq(nfft, dt_s)
    Isf = np.fft.rfft(Is, nfft)
    band = (f >= meta["fmin"]) & (f <= meta["fmax"])
    out = {"f_GHz": f[band] / 1e9}
    for p in ports:
        tv, v = sig[p["name"]]
        Vf = np.fft.rfft(np.interp(t, tv, v), nfft)
        s = 2 * Vf / (Z0 * Isf)
        if p["name"] == drv["name"]:
            s = s - 1
        key = f"S{p['name']}{drv['name']}"
        out[f"dB_{key}"] = 20 * np.log10(np.abs(s[band]) + 1e-30)
        out[f"deg_{key}"] = np.degrees(np.angle(s[band]))

    t_sim = t[-1]
    decay = {p["name"]: float(np.abs(sig[p["name"]][1][-max(3, len(t) // 20):]).max() /
                              (np.abs(sig[p["name"]][1]).max() + 1e-30)) for p in ports}
    print(f"simulated {t_sim*1e9:.2f} ns, {len(t)} samples, frequency resolution {1/t_sim/1e6:.0f} MHz")
    for k, r in decay.items():
        print(f"  port {k}: last-5% |V| / max |V| = {r:.2e}" + ("  (not decayed: S-params are truncated)" if r > 1e-2 else ""))

    keys = list(out)
    with open(os.path.join(args.run_dir, "sparams.csv"), "w") as fcsv:
        fcsv.write(",".join(keys) + "\n")
        for i in range(len(out["f_GHz"])):
            fcsv.write(",".join(f"{out[k][i]:.6g}" for k in keys) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.plot(t * 1e9, Is * Z0 / 2, "k--", lw=1, label="incident (Z0*Is/2)")
    for p in ports:
        a1.plot(sig[p["name"]][0] * 1e9, sig[p["name"]][1], label=f"V port {p['name']}")
    a1.set_xlabel("t (ns)")
    a1.set_ylabel("V")
    a1.legend()
    a1.grid(alpha=0.3)
    for k in keys:
        if k.startswith("dB_"):
            a2.plot(out["f_GHz"], out[k], label=k[3:])
    a2.set_xlabel("f (GHz)")
    a2.set_ylabel("dB")
    a2.set_title(f"ARTEMIS, {t_sim*1e9:.2f} ns simulated")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.run_dir, "sparams.png"), dpi=110)
    print("wrote sparams.csv and sparams.png")


if __name__ == "__main__":
    main()
