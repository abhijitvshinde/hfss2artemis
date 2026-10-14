"""Turn model.json (from hfss_export.py) into a runnable ARTEMIS input deck.

    python build_inputs.py model.json --dx 40 --dy 40 --dz 5 --ns 3 -o run_dir

Grid sizes are in micrometres. The geometry is voxelised onto the Yee grid, materials are written as
sums of boxes (ARTEMIS parser functions), and every lumped port becomes
    * a matched resistive sheet (sigma chosen so the gap is Z0), plus
    * on the driven port, a soft E-field source = Norton current source of known current Is(t),
    * a FieldProbe line across the gap (port voltage).
sparams.py then gives  S11 = 2 V1/(Z0 Is) - 1  and  S21 = 2 V2/(Z0 Is).

Writes run_dir/inputs, run_dir/ports.json and run_dir/run.sh. Needs only numpy.
"""
import argparse
import json
import math
import os

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-06
C0 = 299792458.0
GAMMA_E = 1.759e11          # |gamma| for g = 2, rad/(s*T)


# ----------------------------------------------------------------------------- geometry helpers
def raster(segments, xs, ys):
    """Even-odd point-in-polygon on the node lattice. Nodes are nudged by +1e-6 cell so that
    shapes come out half-open [lo, hi) and touching shapes never overlap or leave a gap."""
    if not segments:
        return np.zeros((len(xs), len(ys)), bool)
    seg = np.asarray(segments, float)                       # (n, 2, 2)
    px = xs + 1e-6 * (xs[1] - xs[0])
    py = ys + 1.3e-6 * (ys[1] - ys[0])
    x0, y0, x1, y1 = seg[:, 0, 0], seg[:, 0, 1], seg[:, 1, 0], seg[:, 1, 1]
    inside = np.zeros((len(xs), len(ys)), bool)
    for j, y in enumerate(py):
        crosses = (y0 > y) != (y1 > y)
        if not crosses.any():
            continue
        xc = x0[crosses] + (y - y0[crosses]) * (x1[crosses] - x0[crosses]) / (y1[crosses] - y0[crosses])
        inside[:, j] = (np.sum(px[:, None] < xc[None, :], axis=1) % 2) == 1
    return inside


def rects_2d(mask):
    """Greedy decomposition of a 2D bool mask into index rectangles (i0, i1, j0, j1), inclusive."""
    out, open_ = [], {}
    for j in range(mask.shape[1] + 1):
        runs = set()
        if j < mask.shape[1]:
            col = np.concatenate(([False], mask[:, j], [False]))
            d = np.flatnonzero(np.diff(col.astype(np.int8)))
            runs = {(int(a), int(b) - 1) for a, b in zip(d[::2], d[1::2])}
        for key in list(open_):
            if key not in runs:
                out.append((key[0], key[1], open_.pop(key), j - 1))
        for key in runs:
            open_.setdefault(key, j)
    return out


def boxes_3d(mask3):
    """Merge per-layer rectangles across z into boxes (i0, i1, j0, j1, k0, k1), inclusive."""
    out, open_ = [], {}
    for k in range(mask3.shape[2] + 1):
        rs = set(rects_2d(mask3[:, :, k])) if k < mask3.shape[2] else set()
        for key in list(open_):
            if key not in rs:
                out.append(key + (open_.pop(key), k - 1))
        for key in rs:
            open_.setdefault(key, k)
    return out


class Grid:
    def __init__(self, lo, d, n):
        self.lo, self.d, self.n = np.asarray(lo, float), np.asarray(d, float), np.asarray(n, int)
        self.x = [self.lo[a] + self.d[a] * np.arange(self.n[a] + 1) for a in range(3)]

    def box_expr(self, b):
        """Parser indicator for node box b: covers the nodes and the half-edges above them."""
        terms = []
        for a, v in enumerate("xyz"):
            i0, i1 = b[2 * a], b[2 * a + 1]
            lo = self.x[a][i0] - 0.5 * self.d[a] + 1e-3 * self.d[a]
            hi = self.x[a][i1] + 0.5 * self.d[a] + 1e-3 * self.d[a]
            terms.append(f"({v}>{lo:.9e})*({v}<{hi:.9e})")
        return "*".join(terms)


def sum_expr(grid, background, regions):
    """background + sum_k (value_k - background) * indicator(boxes of k)."""
    parts = [f"{background:.9e}"]
    for value, boxes in regions:
        if abs(value - background) < 1e-300 or not boxes:
            continue
        ind = " + ".join(grid.box_expr(b) for b in boxes)
        parts.append(f"({value - background:.9e})*({ind})")
    return "\n    + ".join(parts)


def preview(labels, table, g, ports, path):
    """Top views of every z-layer that holds a ferrite, metal or port, plus an x-z cut."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return
    xs, ys, zs = (v * 1e3 for v in g.x)
    special = [i for i, t in enumerate(table) if t.get("ferrite") or t["name"].startswith(("metal", "port"))]
    layers = [k for k in range(labels.shape[2]) if np.isin(labels[:, :, k], special).any()]
    names = [t["name"] for t in table]
    cmap = ListedColormap(plt.cm.tab10(np.arange(len(table)) % 10))
    ncol = len(layers) + 1
    fig, axs = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.4), squeeze=False)
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    for ax, k in zip(axs[0], layers):
        ax.imshow(labels[:, :, k].T, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=len(table) - 0.5,
                  interpolation="nearest")
        for p in ports:
            (a, b) = p["line"]
            ax.plot([a[0] * 1e3, b[0] * 1e3], [a[1] * 1e3, b[1] * 1e3], "w-", lw=2)
        ax.set_title(f"z = {zs[k]:.4f} mm")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    jm = labels.shape[1] // 2
    if ports:
        jm = int(np.argmin(abs(ys - ports[0]["line"][0][1] * 1e3)))
    ax = axs[0][-1]
    ax.imshow(labels[:, jm, :].T, origin="lower", aspect="auto", extent=[xs[0], xs[-1], zs[0], zs[-1]],
              cmap=cmap, vmin=-0.5, vmax=len(table) - 0.5, interpolation="nearest")
    ax.set_title(f"x-z cut at y = {ys[jm]:.3f} mm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(len(table))]
    fig.legend(handles, names, loc="lower center", ncol=min(len(names), 8))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("-o", "--out", default="artemis_run")
    ap.add_argument("--dx", type=float, default=20.0, help="cell size x (um)")
    ap.add_argument("--dy", type=float, default=20.0, help="cell size y (um)")
    ap.add_argument("--dz", type=float, default=5.0, help="cell size z (um)")
    ap.add_argument("--pad-xy", type=float, default=0.2, help="air margin around the device in x/y (mm)")
    ap.add_argument("--pad-below", type=float, default=0.1, help="air below the substrate (mm)")
    ap.add_argument("--pad-above", type=float, default=0.3, help="air above the top metal (mm)")
    ap.add_argument("--metal-cells", type=int, default=2, help="thickness of sheet conductors in cells")
    ap.add_argument("--ns", type=float, default=None, help="simulated time (ns); default = pulse + 20 periods")
    ap.add_argument("--max-step", type=int, default=None, help="overrides --ns")
    ap.add_argument("--fmin", type=float, default=None, help="GHz (default: HFSS sweep start)")
    ap.add_argument("--fmax", type=float, default=None, help="GHz (default: HFSS sweep stop)")
    ap.add_argument("--bias-oe", type=float, default=None, help="override bias magnitude (Oe)")
    ap.add_argument("--cfl", type=float, default=0.8)
    ap.add_argument("--drive", default=None, help="port name to excite (default: first port)")
    ap.add_argument("--max-grid", type=int, default=128)
    ap.add_argument("--slice-every-ns", type=float, default=0.25, help="z-slice plotfile interval (ns)")
    ap.add_argument("--checkpoint-every-ns", type=float, default=0.25,
                    help="restart checkpoint interval (ns); h2a.py keeps only the newest few")
    ap.add_argument("--sec-per-cell-step", type=float, default=6.0e-8,
                    help="measured cost used for the run-time estimate (RTX 3050 laptop ~6e-8)")
    args = ap.parse_args(argv)

    m = json.load(open(args.model))
    warnings = list(m.get("warnings", []))
    fmin = (args.fmin * 1e9) if args.fmin else m.get("fstart_Hz", 1e9)
    fmax = (args.fmax * 1e9) if args.fmax else m.get("fstop_Hz", 10e9)
    f0, bw = 0.5 * (fmin + fmax), fmax - fmin

    # ---- domain + grid (z nodes aligned to the lowest conductor plane)
    pts = []
    for s in m["solids"]:
        seg = np.asarray(s["segments"])
        pts += [[seg[..., 0].min(), seg[..., 1].min(), s["zmin"]], [seg[..., 0].max(), seg[..., 1].max(), s["zmax"]]]
    for c in m["conductors"]:
        for f in c["faces"]:
            seg = np.asarray(f["segments"])
            pts += [[seg[..., 0].min(), seg[..., 1].min(), f["z"]], [seg[..., 0].max(), seg[..., 1].max(), f["z"]]]
    pts = np.asarray(pts)
    bmin, bmax = pts.min(0), pts.max(0)
    d = np.array([args.dx, args.dy, args.dz]) * 1e-6
    zref = min((f["z"] for c in m["conductors"] for f in c["faces"]), default=bmax[2])
    lo = np.array([bmin[0] - args.pad_xy * 1e-3, bmin[1] - args.pad_xy * 1e-3, 0.0])
    lo[0] = bmin[0] - math.ceil(args.pad_xy * 1e-3 / d[0]) * d[0]
    lo[1] = bmin[1] - math.ceil(args.pad_xy * 1e-3 / d[1]) * d[1]
    lo[2] = zref - math.ceil((zref - bmin[2] + args.pad_below * 1e-3) / d[2] - 1e-9) * d[2]
    hi = bmax + np.array([args.pad_xy, args.pad_xy, args.pad_above]) * 1e-3 + [0, 0, args.metal_cells * d[2]]
    n = np.ceil((hi - lo) / d - 1e-9).astype(int)
    n += n % 2                                         # even, for blocking_factor = 2
    g = Grid(lo, d, n)
    xs, ys, zs = g.x
    tol = 1e-6 * d[2]

    # ---- label lattice: 0 = air
    labels = np.zeros(tuple(n + 1), np.int16)
    table = [{"name": "air", "eps": EPS0, "sigma": 0.0, "mu": MU0}]

    def new_label(rec):
        table.append(rec)
        return len(table) - 1

    mats = m["materials"]
    solids = sorted(m["solids"], key=lambda s: mats[s["material"]].get("ferrite", False))
    for s in solids:
        mt = mats[s["material"]]
        sig = mt["sigma"] + 2 * math.pi * f0 * EPS0 * mt["eps_r"] * mt["tan_d"]
        rec = {"name": s["name"], "eps": EPS0 * mt["eps_r"], "sigma": sig, "mu": MU0 * mt["mu_r"]}
        if mt.get("ferrite"):
            g_fac = mt.get("lande_g", 2.0)
            gyro_hz_per_t = GAMMA_E / (2 * math.pi) * g_fac / 2.0
            rec.update(mu=MU0, ferrite=True, Ms=mt["Ms_Apm"], gamma=-GAMMA_E * g_fac / 2.0,
                       alpha=gyro_hz_per_t * 1e-4 * mt.get("delta_H_Oe", 0.0) / (2 * f0))
            if abs(mt["mu_r"] - 1) > 1e-9:
                warnings.append(f"{s['material']}: HFSS mu_r={mt['mu_r']} ignored; LLG supplies the ferrite permeability")
            bias = [b for b in m["bias"] if s["name"] in b["objects"]]
            if bias:
                h = bias[0]["H_Apm"] if args.bias_oe is None else args.bias_oe * 1000 / (4 * math.pi)
                rec["H_bias"] = [h * c for c in bias[0]["direction"]]
                rec["M_init"] = [rec["Ms"] * c for c in bias[0]["direction"]]
            else:
                warnings.append(f"ferrite '{s['name']}' has no bias; M starts along +z")
                rec["H_bias"], rec["M_init"] = [0, 0, 0], [0, 0, rec["Ms"]]
        lab = new_label(rec)
        mask2 = raster(s["segments"], xs, ys)
        kz = np.flatnonzero((zs + tol >= s["zmin"]) & (zs + tol < s["zmax"]))
        if kz.size == 0:
            kz = np.array([int(np.argmin(abs(zs - 0.5 * (s["zmin"] + s["zmax"]))))])
            warnings.append(f"solid '{s['name']}' thinner than dz; forced to one cell layer")
        for k in kz:
            labels[:, :, k][mask2] = lab
        thick = s["zmax"] - s["zmin"]
        if thick < 2 * d[2] and rec.get("ferrite"):
            warnings.append(f"'{s['name']}' is {thick*1e6:.2f} um thick = {thick/d[2]:.1f} cell(s) in z")

    cond_label = {}
    for c in m["conductors"]:
        sig = min(c["sigma"], 1e10)
        if sig not in cond_label:
            cond_label[sig] = new_label({"name": f"metal_{sig:.3g}", "eps": EPS0, "sigma": sig, "mu": MU0})
        lab = cond_label[sig]
        ztop = max(f["z"] for f in c["faces"])
        for f in c["faces"]:
            mask2 = raster(f["segments"], xs, ys)
            kz = np.flatnonzero((zs >= f["z"] - tol) & (zs <= ztop + (args.metal_cells - 1) * d[2] + tol))
            for k in kz:
                labels[:, :, k][mask2] = lab
    is_metal = np.isin(labels, list(cond_label.values()))

    # ---- ports
    ports, drive = [], args.drive or (m["ports"][0]["name"] if m["ports"] else None)
    for p in m["ports"]:
        segs = p["segments"] or [[[p["bbox"][0], p["bbox"][1]], [p["bbox"][3], p["bbox"][1]]],
                                 [[p["bbox"][3], p["bbox"][1]], [p["bbox"][3], p["bbox"][4]]],
                                 [[p["bbox"][3], p["bbox"][4]], [p["bbox"][0], p["bbox"][4]]],
                                 [[p["bbox"][0], p["bbox"][4]], [p["bbox"][0], p["bbox"][1]]]]
        mask2 = raster(segs, xs, ys)
        if not mask2.any():
            raise SystemExit(f"port {p['name']} smaller than one cell; refine the grid")
        ii, jj = np.nonzero(mask2)
        i0, i1, j0, j1 = ii.min(), ii.max(), jj.min(), jj.max()
        kp = int(np.argmin(abs(zs - p["z"])))
        im, jm = (i0 + i1) // 2, (j0 + j1) // 2
        axis = None
        if i0 > 0 and i1 + 1 <= n[0] and is_metal[i0 - 1, jm, kp] and is_metal[i1 + 1, jm, kp]:
            axis = 0
        elif j0 > 0 and j1 + 1 <= n[1] and is_metal[im, j0 - 1, kp] and is_metal[im, j1 + 1, kp]:
            axis = 1
        if axis is None:
            raise SystemExit(f"port {p['name']}: no conductor on two opposite sides at the port plane")
        before = (i0 - 1, jm) if axis == 0 else (im, j0 - 1)
        after = (i1 + 1, jm) if axis == 0 else (im, j1 + 1)
        kz = np.flatnonzero(is_metal[before[0], before[1], :] & is_metal[after[0], after[1], :])
        kz = kz[(kz >= kp) & (kz <= kp + args.metal_cells + 1)]
        n_along = (i1 - i0 + 1) if axis == 0 else (j1 - j0 + 1)
        n_trans = (j1 - j0 + 1) if axis == 0 else (i1 - i0 + 1)
        L = n_along * d[axis]
        A = n_trans * len(kz) * d[1 - axis] * d[2]
        sigma_p = L / (p["Z0"] * A)
        lab = new_label({"name": f"port_{p['name']}", "eps": EPS0, "sigma": sigma_p, "mu": MU0})
        for k in kz:
            labels[i0:i1 + 1, j0:j1 + 1, k][mask2[i0:i1 + 1, j0:j1 + 1]] = lab
        a0 = xs[before[0]] if axis == 0 else ys[before[1]]
        a1 = xs[after[0]] if axis == 0 else ys[after[1]]
        tc = ys[jm] if axis == 0 else xs[im]
        zc = zs[kz].mean()
        ports.append({"name": p["name"], "Z0": p["Z0"], "axis": "xyz"[axis], "label": lab, "L": L, "A": A,
                      "sigma": sigma_p, "driven": p["name"] == drive, "probe": f"V_port{p['name']}",
                      "line": [[a0, tc, zc], [a1, tc, zc]] if axis == 0 else [[tc, a0, zc], [tc, a1, zc]],
                      "resolution": int(round((a1 - a0) / d[axis])) + 1})

    # ---- boxes per label
    boxes = {lab: boxes_3d(labels == lab) for lab in range(1, len(table))}

    def regions(key, default=0.0):
        return [(table[lab].get(key, default), boxes[lab]) for lab in boxes]

    def comp_regions(key, c):
        return [(table[lab][key][c], boxes[lab]) for lab in boxes if key in table[lab]]

    # ---- time step + source
    dt = args.cfl / (C0 * math.sqrt(sum(1 / di ** 2 for di in d)))
    tau = math.sqrt(2 * math.log(10)) / (math.pi * bw)       # spectrum -20 dB at band edges
    t0 = 4 * tau
    t_end = (args.ns * 1e-9) if args.ns else (2 * t0 + 20 / f0)
    max_step = args.max_step or int(math.ceil(t_end / dt))
    probe_every = max(1, int(1 / (20 * fmax * dt)))           # >= 20 samples per period at fmax
    slice_every = max(1, int(args.slice_every_ns * 1e-9 / dt))
    Is_amp = 2.0 / ports[0]["Z0"] if ports else 0.0          # 1 V incident wave

    lines = []
    w = lines.append
    w(f"# ARTEMIS input generated by hfss2artemis from {m.get('source')} [{m.get('design')}]")
    w(f"# grid {args.dx} x {args.dy} x {args.dz} um, band {fmin/1e9:.3g}-{fmax/1e9:.3g} GHz")
    w(f"max_step = {max_step}")
    w(f"amr.n_cell = {n[0]} {n[1]} {n[2]}")
    w(f"amr.max_grid_size = {args.max_grid}")
    w("amr.blocking_factor = 2")
    w("amr.max_level = 0")
    w("geometry.dims = 3")
    w(f"geometry.prob_lo = {lo[0]:.9e} {lo[1]:.9e} {lo[2]:.9e}")
    w(f"geometry.prob_hi = {lo[0]+n[0]*d[0]:.9e} {lo[1]+n[1]*d[1]:.9e} {lo[2]+n[2]*d[2]:.9e}")
    bc = "pml pml pml" if m.get("open_boundary") else "pec pec pec"
    w(f"boundary.field_lo = {bc}")
    w(f"boundary.field_hi = {bc}")
    w("warpx.verbose = 1")
    w("warpx.use_filter = 0")
    w(f"warpx.const_dt = {dt:.12e}")
    w("particles.nspecies = 0")
    w("algo.em_solver_medium = macroscopic")
    w("algo.macroscopic_sigma_method = backwardeuler")
    w(f'macroscopic.epsilon_function(x,y,z) = "{sum_expr(g, EPS0, regions("eps", EPS0))}"')
    w(f'macroscopic.sigma_function(x,y,z) = "{sum_expr(g, 0.0, regions("sigma"))}"')
    w(f'macroscopic.mu_function(x,y,z) = "{sum_expr(g, MU0, regions("mu", MU0))}"')

    ferr = [lab for lab in boxes if table[lab].get("ferrite")]
    if ferr:
        w("warpx.mag_M_normalization = 1")
        w("warpx.mag_LLG_coupling = 1")
        w("warpx.mag_time_scheme_order = 2")
        w("macroscopic.mag_max_iter = 100")
        w("macroscopic.mag_tol = 1.e-7")
        w("macroscopic.mag_normalized_error = 0.1")
        for key, nm in (("Ms", "Ms"), ("alpha", "alpha"), ("gamma", "gamma")):
            w(f'macroscopic.mag_{nm}_init_style = "parse_mag_{nm}_function"')
            w(f'macroscopic.mag_{nm}_function(x,y,z) = "{sum_expr(g, 0.0, regions(key))}"')
        w("warpx.H_bias_ext_grid_init_style = parse_H_bias_ext_grid_function")
        for c, ax in enumerate("xyz"):
            w(f'warpx.H{ax}_bias_external_grid_function(x,y,z) = "{sum_expr(g, 0.0, comp_regions("H_bias", c))}"')
        w("warpx.M_ext_grid_init_style = parse_M_ext_grid_function")
        for c, ax in enumerate("xyz"):
            w(f'warpx.M{ax}_external_grid_function(x,y,z) = "{sum_expr(g, 0.0, comp_regions("M_init", c))}"')

    w('warpx.E_excitation_on_grid_style = "parse_E_excitation_grid_function"')
    for ax in "xyz":
        drv = [p for p in ports if p["driven"] and p["axis"] == ax]
        if drv:
            p = drv[0]
            # ARTEMIS adds the soft source after the backward-Euler sigma update, so the injected
            # current density is (eps/dt + sigma) * f, not eps/dt * f.
            kf = Is_amp * dt / (EPS0 * p["A"] * (1 + p["sigma"] * dt / EPS0))
            ind = " + ".join(g.box_expr(b) for b in boxes[p["label"]])
            w(f'warpx.E{ax}_excitation_flag_function(x,y,z) = "2*({ind})"')
            w(f'warpx.E{ax}_excitation_grid_function(x,y,z,t) = "{kf:.9e}*exp(-(t-{t0:.9e})**2/(2*{tau:.9e}**2))'
              f'*sin(2*3.14159265358979*{f0:.9e}*(t-{t0:.9e}))"')
        else:
            w(f'warpx.E{ax}_excitation_flag_function(x,y,z) = "0"')
            w(f'warpx.E{ax}_excitation_grid_function(x,y,z,t) = "0."')

    w("warpx.reduced_diags_names = " + " ".join(p["probe"] for p in ports))
    # Port voltage: volume integral of E over the port sheet = V * A  (FieldProbe reads B, which the
    # LLG build does not allocate, so it segfaults; RawEFieldReduction is what ARTEMIS' CPW examples use)
    for p in ports:
        ind = " + ".join(g.box_expr(b) for b in boxes[p["label"]])
        w(f"{p['probe']}.type = RawEFieldReduction")
        w(f"{p['probe']}.reduction_type = integral")
        w(f"{p['probe']}.integration_type = volume")
        w(f"{p['probe']}.intervals = {probe_every}")
        w(f"{p['probe']}.reduced_function(x,y,z) = {ind}")
    zslice = zs[int(np.argmin(abs(zs - zref)))]
    if ferr:
        kf_ = np.flatnonzero((labels == ferr[0]).any(axis=(0, 1)))
        zslice = zs[kf_[len(kf_) // 2]] + 0.5 * d[2]
    w("diagnostics.diags_names = mat slice chk")
    w(f"chk.intervals = {max(1, int(args.checkpoint_every_ns * 1e-9 / dt))}")
    w("chk.diag_type = Full")
    w("chk.format = checkpoint")
    w("mat.intervals = 1:1")
    w("mat.diag_type = Full")
    w("mat.fields_to_plot = epsilon sigma mu")
    w(f"slice.intervals = {slice_every}")
    w("slice.diag_type = Full")
    w(f"slice.diag_lo = {lo[0]:.9e} {lo[1]:.9e} {zslice:.9e}")
    w(f"slice.diag_hi = {lo[0]+n[0]*d[0]:.9e} {lo[1]+n[1]*d[1]:.9e} {zslice:.9e}")
    w("slice.fields_to_plot = Ex Ey Ez Hx Hy Hz" + (" Mx_zface My_zface Mz_zface" if ferr else ""))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "inputs"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    meta = {"dt": dt, "max_step": max_step, "f0": f0, "fmin": fmin, "fmax": fmax, "tau": tau, "t0": t0,
            "Is_amp": Is_amp, "Is_includes_sigma_gain": True, "ports": ports, "grid": {"lo": lo.tolist(), "d": d.tolist(), "n": n.tolist()},
            "materials": [{k: v for k, v in t.items()} for t in table], "warnings": warnings}
    with open(os.path.join(args.out, "ports.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(args.out, "run.sh"), "w", newline="\n") as f:
        f.write("#!/bin/bash\n# usage: ./run.sh /path/to/warpx.3d...LLG [n_mpi_ranks]\n"
                'cd "$(dirname "$0")"\nmpirun -n "${2:-1}" "$1" inputs > run.log 2>&1\n')

    preview(labels, table, g, ports, os.path.join(args.out, "geometry_preview.png"))

    cells = int(np.prod(n))
    pml = 2 * 10 * (n[0] * n[1] + n[1] * n[2] + n[0] * n[2]) if m.get("open_boundary") else 0
    print(f"grid {n[0]} x {n[1]} x {n[2]} = {cells/1e6:.2f} M cells (+{pml/1e6:.2f} M PML), "
          f"dt = {dt*1e15:.2f} fs, {max_step} steps = {max_step*dt*1e9:.2f} ns")
    print(f"estimated GPU memory ~{(cells+pml)*450/1e9:.1f} GB, run time ~{max_step*(cells+pml)*args.sec_per_cell_step/3600:.1f} h "
          f"(at {args.sec_per_cell_step:g} s/cell/step)")
    for t in table:
        extra = f" Ms={t['Ms']:.4g} alpha={t['alpha']:.3g} H={t.get('H_bias')}" if t.get("ferrite") else ""
        print(f"  {t['name']:16s} eps_r={t['eps']/EPS0:6.2f} sigma={t['sigma']:.3g}{extra}")
    for p in ports:
        print(f"  port {p['name']}: E{p['axis']} across {p['L']*1e6:.0f} um, area {p['A']*1e12:.0f} um^2, "
              f"sigma={p['sigma']:.3g} S/m{' (driven)' if p['driven'] else ''}")
    print(f"  {sum(len(b) for b in boxes.values())} material boxes")
    for wn in warnings:
        print("  warning:", wn)


if __name__ == "__main__":
    main()
