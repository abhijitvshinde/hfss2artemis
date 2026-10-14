"""Export an HFSS design to a neutral model.json that build_inputs.py turns into an ARTEMIS input.

Runs on Windows with Ansys AEDT + pyaedt. The project is COPIED to a temp folder first, so the
original .aedt is never opened for writing.

    python hfss_export.py "C:/Users/me/Documents/Ansoft/my_design.aedt" --design HFSSDesign1 -o model.json

What gets exported (everything converted to SI metres):
  * 3D solids      -> material + z range + outline of their top face (extruded prisms)
  * 2D sheets      -> conductors (Finite Conductivity / Perfect E) and lumped-port sheets
  * materials      -> eps_r, tan(delta), sigma, mu_r, ferrite Ms / g / delta_H
  * magnetic bias  -> field (A/m) and direction, incl. the value from a parametric sweep
  * setup / sweep  -> frequency band
  * radiation/PML  -> open boundary flag
Air/region boxes and HFSS PML objects are dropped; build_inputs.py makes its own box + PML.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile

from ansys.aedt.core import Hfss

LEN = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9, "mil": 25.4e-6, "in": 25.4e-3}
OE_TO_APM = 1000.0 / (4.0 * math.pi)


def to_float(text, default=None):
    """'1750gauss' -> 1750.0, '3GHz' -> 3e9, '500Oe' -> 500, '58000000' -> 5.8e7."""
    if text is None:
        return default
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().strip("'\"")
    m = re.match(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Za-z_]*)$", s)
    if not m:
        return default
    val, unit = float(m.group(1)), m.group(2).lower()
    scale = {"": 1, "hz": 1, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12,
             "gauss": 1, "oe": 1, "a_per_meter": 1, "tesla": 1, "ohm": 1, "deg": 1,
             "siemens/m": 1}
    scale.update(LEN)
    return val * scale.get(unit, 1)


def field_to_apm(text):
    """Bias value as written in HFSS ('500Oe', '39788A_per_meter', '0.05tesla') -> A/m."""
    s = str(text).strip().lower()
    v = to_float(s, 0.0)
    if s.endswith("oe"):
        return v * OE_TO_APM
    if s.endswith("tesla"):
        return v / (4e-7 * math.pi)
    return v


def rot_z_axis(ax_deg, ay_deg, az_deg):
    """HFSS bias direction: +z rotated about x, then y, then z (R = Rz*Ry*Rx)."""
    ax, ay, az = (math.radians(a) for a in (ax_deg, ay_deg, az_deg))
    v = [0.0, 0.0, 1.0]
    v = [v[0], v[1] * math.cos(ax) - v[2] * math.sin(ax), v[1] * math.sin(ax) + v[2] * math.cos(ax)]
    v = [v[0] * math.cos(ay) + v[2] * math.sin(ay), v[1], -v[0] * math.sin(ay) + v[2] * math.cos(ay)]
    v = [v[0] * math.cos(az) - v[1] * math.sin(az), v[0] * math.sin(az) + v[1] * math.cos(az), v[2]]
    return [round(c, 12) for c in v]


def edge_segments(face, scale):
    """Face outline as straight segments [[x0,y0],[x1,y1]] (arcs approximated start-mid-end)."""
    segs = []
    for e in face.edges:
        pts = [v.position for v in e.vertices]
        try:
            mid = e.midpoint
        except Exception:
            mid = None
        if len(pts) == 2:
            a, b = pts
            chord_mid = [(a[i] + b[i]) / 2 for i in range(3)]
            if mid and math.dist(mid[:2], chord_mid[:2]) > 1e-9:
                segs += [[a[:2], mid[:2]], [mid[:2], b[:2]]]
            else:
                segs.append([a[:2], b[:2]])
        elif len(pts) == 1 and mid:  # full circle: approximate as a 16-gon
            c = [(pts[0][i] + mid[i]) / 2 for i in range(2)]
            r = math.dist(pts[0][:2], c)
            ring = [[c[0] + r * math.cos(2 * math.pi * k / 16), c[1] + r * math.sin(2 * math.pi * k / 16)]
                    for k in range(17)]
            segs += [[ring[k], ring[k + 1]] for k in range(16)]
    return [[[p[0] * scale, p[1] * scale] for p in s] for s in segs]


def horizontal_faces(obj, scale):
    out = []
    for f in obj.faces:
        n = f.normal
        if n is None or abs(abs(n[2]) - 1.0) > 1e-6:
            continue
        out.append({"z": f.center[2] * scale, "normal_z": 1 if n[2] > 0 else -1,
                    "segments": edge_segments(f, scale)})
    return out


def material_record(hfss, name):
    rec = {"name": name, "eps_r": 1.0, "tan_d": 0.0, "sigma": 0.0, "mu_r": 1.0, "ferrite": False}
    if not name or name.lower() == "vacuum":
        return rec
    mat = hfss.materials.material_keys.get(name.lower())
    if mat is None:
        return rec
    props = {k.lower(): v for k, v in dict(mat._props).items()}
    rec["eps_r"] = to_float(props.get("permittivity"), 1.0)
    rec["tan_d"] = to_float(props.get("dielectric_loss_tangent"), 0.0)
    rec["sigma"] = to_float(props.get("conductivity"), 0.0)
    rec["mu_r"] = to_float(props.get("permeability"), 1.0)
    ms = props.get("saturation_mag")
    if ms is not None and to_float(ms, 0.0) > 0:
        rec["ferrite"] = True
        rec["four_pi_Ms_gauss"] = to_float(ms)            # 4*pi*Ms in gauss
        rec["Ms_Apm"] = to_float(ms) * OE_TO_APM           # 1 G of 4piMs = 1000/4pi A/m
        rec["lande_g"] = to_float(props.get("lande_g_factor"), 2.0)
        rec["delta_H_Oe"] = to_float(props.get("delta_h"), 0.0)
    return rec


def parametric_value(hfss, var):
    """First value a parametric (Optimetrics) setup assigns to `var`, or None."""
    try:
        setups = hfss.parametrics.setups
    except Exception:
        return None
    for ps in setups:
        sweeps = ps.props.get("Sweeps", {}).get("SweepDefinition", [])
        if isinstance(sweeps, dict):
            sweeps = [sweeps]
        for sd in sweeps:
            if sd.get("Variable") == var:
                first = str(sd.get("Data", "")).split()
                if first and first[0].upper() in ("LIN", "LINC", "DEC", "ESTP"):
                    first = first[1:]
                if first:
                    return first[0]
    return None


def _aedt_pids():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq ansysedt.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return {int(m) for m in re.findall(r'"ansysedt\.exe","(\d+)"', out)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aedt")
    ap.add_argument("--design", default=None, help="design name (default: first HFSS design)")
    ap.add_argument("--version", default="2024.1")
    ap.add_argument("-o", "--out", default="model.json")
    args = ap.parse_args(argv)

    tmp = tempfile.mkdtemp(prefix="hfss2artemis_")
    # AEDT's scripting interface refuses project names with spaces/odd characters
    # ("access to the requested resource is not permitted"), so the copy gets a safe name.
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(args.aedt))[0]) + ".aedt"
    copy = os.path.join(tmp, safe)
    shutil.copy(args.aedt, copy)
    before = _aedt_pids()
    try:
        hfss = Hfss(project=copy, design=args.design, version=args.version,
                    non_graphical=True, new_desktop=True, close_on_exit=True)
    except Exception as e:
        for pid in _aedt_pids() - before:      # the AEDT we started would otherwise linger and block the GUI
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        log = open("batch.log", errors="ignore").read() if os.path.isfile("batch.log") else ""
        shutil.rmtree(tmp, ignore_errors=True)
        if "FlexNet Licensing error" in log or "vendor daemon is down" in log:
            raise SystemExit("AEDT could not get a valid HFSS license, so it opened the design read-only and "
                             "refused the export.\nThe export step needs a working, licensed HFSS; "
                             "everything after it (build, run, results) does not.")
        raise SystemExit(f"AEDT could not open design '{args.design}' in {args.aedt}: {e}\nSee batch.log for details.")
    try:
        scale = LEN[hfss.modeler.model_units.lower()]
        by_id = {o.id: o for o in hfss.modeler.object_list}
        model = {"source": os.path.abspath(args.aedt), "design": hfss.design_name,
                 "solids": [], "conductors": [], "ports": [], "bias": [], "warnings": [],
                 "open_boundary": False, "materials": {}}

        cond_ids, port_ids = {}, {}
        for b in hfss.boundaries:
            p = dict(b.props)
            ids = p.get("Objects", []) or []
            if b.type in ("Finite Conductivity", "Perfect E"):
                sig = 1e30 if b.type == "Perfect E" else to_float(p.get("Conductivity"), 5.8e7)
                for i in ids:
                    cond_ids[i] = sig
            elif b.type == "Lumped Port":
                for i in ids:
                    port_ids[i] = {"name": b.name, "Z0": to_float(p.get("Impedance"), 50.0)}
            elif b.type in ("Radiation", "PML"):
                model["open_boundary"] = True
            elif b.type == "Magnetic Bias":
                expr = str(p.get("Bias", "0"))
                h = field_to_apm(hfss.get_evaluated_value(expr) if not re.match(r"^[\d.]", expr) else expr)
                if abs(h) < 1e-12:
                    pv = parametric_value(hfss, expr)
                    if pv is not None:
                        h = field_to_apm(pv)
                        model["warnings"].append(f"bias {expr}=0 in nominal design; using parametric value {pv}")
                direction = rot_z_axis(to_float(p.get("XAngle"), 0), to_float(p.get("YAngle"), 0),
                                       to_float(p.get("ZAngle"), 0))
                model["bias"].append({"name": b.name, "objects": [by_id[i].name for i in ids if i in by_id],
                                      "H_Apm": h, "direction": direction, "expr": expr})
            else:
                model["warnings"].append(f"boundary '{b.name}' of type '{b.type}' ignored")

        for o in hfss.modeler.object_list:
            name, mat = o.name, (o.material_name or "")
            if name.startswith("PML_") or mat.lower().startswith("pmlgroup"):
                model["open_boundary"] = True
                continue
            faces = horizontal_faces(o, scale)
            if o.is_3d:
                bb = [c * scale for c in o.bounding_box]
                if mat.lower() in ("vacuum", "air", ""):
                    continue  # air box / radiation region: the converter builds its own
                top = [f for f in faces if f["normal_z"] > 0 and abs(f["z"] - bb[5]) < 1e-12]
                if len(faces) < 2 or not top:
                    model["warnings"].append(f"solid '{name}' is not a vertical prism; using its bounding box")
                    top = [{"z": bb[5], "segments": [[[bb[0], bb[1]], [bb[3], bb[1]]], [[bb[3], bb[1]], [bb[3], bb[4]]],
                                                     [[bb[3], bb[4]], [bb[0], bb[4]]], [[bb[0], bb[4]], [bb[0], bb[1]]]]}]
                model["materials"][mat] = material_record(hfss, mat)
                model["solids"].append({"name": name, "material": mat, "zmin": bb[2], "zmax": bb[5],
                                        "segments": [s for f in top for s in f["segments"]]})
            elif o.id in port_ids:
                pr = port_ids[o.id]
                bb = [c * scale for c in o.bounding_box]
                model["ports"].append({"name": pr["name"], "object": name, "Z0": pr["Z0"],
                                       "z": faces[0]["z"] if faces else bb[2], "bbox": bb,
                                       "segments": faces[0]["segments"] if faces else []})
            elif o.id in cond_ids:
                model["conductors"].append({"name": name, "sigma": cond_ids[o.id], "faces": faces})
            else:
                model["warnings"].append(f"sheet '{name}' has no conductor/port boundary; ignored")

        try:
            setup = hfss.setups[0]
            model["setup_freq_Hz"] = to_float(setup.props.get("Frequency"))
            sw = setup.sweeps[0].props if setup.sweeps else {}
            model["fstart_Hz"] = to_float(sw.get("RangeStart"), model["setup_freq_Hz"] * 0.5)
            model["fstop_Hz"] = to_float(sw.get("RangeEnd"), model["setup_freq_Hz"] * 1.5)
        except Exception as e:
            model["warnings"].append(f"no setup/sweep found ({e}); set --fmin/--fmax in build_inputs.py")

        with open(args.out, "w") as f:
            json.dump(model, f, indent=1)
        print(f"wrote {args.out}: {len(model['solids'])} solids, {len(model['conductors'])} conductors, "
              f"{len(model['ports'])} ports, {len(model['bias'])} bias region(s)")
        for w in model["warnings"]:
            print("  warning:", w)
    finally:
        hfss.release_desktop(close_projects=True, close_desktop=True)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
