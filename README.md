# hfss2artemis (h2a)

Design in HFSS, simulate in ARTEMIS on the GPU (time domain, full Maxwell + LLG magnetization),
watch progress, get S-parameters. Everything goes through one program: **`h2a.py`**.

```
HFSS .aedt --export--> models/NAME.json --build--> runs/NAME/inputs --run--> ARTEMIS (GPU, background)
                                                                       status / stop / resume
                                                                       sparams --> sparams.png / .csv
```

## Desktop app (no command prompt)

Double-click **HFSS to ARTEMIS** on the Desktop (or `h2a_gui.pyw` in this folder).

1. **New simulation**: Browse to the `.aedt`, pick the design, set dx / dy / dz / simulated time,
   press *Convert & preview*, check the geometry picture and the memory / run-time estimate, press *Start simulation*.
2. **Runs & progress**: every run with a progress bar, time left and finish time, GPU load and any problem;
   buttons Start / Stop / Resume / Start over / Show results / Open folder / Delete. Refreshes every 15 s.
3. **Results**: port voltages and S11 / S21 for any run (also while it is still running), zoomable; Open CSV.

Closing the app does not stop a running simulation; open it again to see the progress.

## Quick start (command line)

From PowerShell in this folder (`python` = `C:\Users\abhij\AppData\Local\Programs\Python\Python313\python.exe`):

```
python h2a.py go "C:\Users\abhij\Documents\Ansoft\MyDesign.aedt" --design HFSSDesign1 --name mydesign --dx 20 --dy 20 --dz 2.5 --ns 3
python h2a.py status            # how far along, speed, time left, errors, GPU load
python h2a.py sparams           # S11/S21 plot (works on a partial run too)
```

In VS Code: open this folder, press **Ctrl+Shift+D** (Run and Debug), pick an `h2a: ...` entry from the
drop-down and press the green play button. "convert + run an HFSS project" asks for the file and grid.

## All commands

| command | what it does |
|---|---|
| `h2a.py export DESIGN.aedt [--design NAME] [-o models/X.json]` | read the HFSS project (opens a copy, never your original) |
| `h2a.py build models/X.json --dx .. --dy .. --dz .. --ns .. [-o runs/X]` | make the ARTEMIS deck, `geometry_preview.png`, memory and time estimate |
| `h2a.py run [RUN] [--cpu] [--ranks N] [--fresh]` | start in the background; closing VS Code / the terminal is fine |
| `h2a.py status [RUN] [--watch]` | step, %, simulated ns, s/step, time left + finish time, checkpoint, GPU load, errors |
| `h2a.py stop [RUN]` | stop the simulation (checkpoints are kept) |
| `h2a.py resume [RUN]` | continue from the newest checkpoint instead of starting over |
| `h2a.py sparams [RUN]` | `RUN/sparams.png` + `sparams.csv` (columns `f_GHz, dB_S11, deg_S11, dB_S21, deg_S21`) |
| `h2a.py list` | every run with its state and progress |
| `h2a.py go DESIGN.aedt ... [--no-run]` | export + build + run in one go |

`RUN` is the run's folder name under `runs\` (e.g. `horn_test`); leave it out to use the most recently active run.
All `build_inputs.py` options can be passed to `build` and `go`, e.g. `--fmin 2 --fmax 4` (GHz), `--bias-oe 500`,
`--pad-xy 0.2 --pad-below 0.1 --pad-above 0.3` (mm), `--metal-cells 2`, `--checkpoint-every-ns 0.25`.

## While a run is going

* It runs inside WSL Ubuntu; `h2a.py` keeps the PC from idle-sleeping while it runs, but **keep it plugged in
  with the lid open** (a closed lid still sleeps).
* Only one simulation at a time per GPU; `run` refuses to start a second one.
* A checkpoint is written every 0.25 ns of simulated time (the newest 2 are kept). If the PC restarts or you
  `stop`, `resume` loses at most that much work.
* Files in `runs\NAME\`: `run.log` (ARTEMIS output), `diags\reducedfiles\V_port*.txt` (port voltages),
  `diags\slice*` (field snapshots of the ferrite layer), `diags\chk*` (checkpoints), `h2a_state.json`.

## Choosing the grid

* thinnest ferrite film >= 2 cells in z (`--dz` <= thickness / 2); smallest gap or slot >= 2-3 cells in x/y;
* memory ~450 bytes/cell: RTX 3050 laptop (6 GB) ~10 M cells; a 24 GB card ~50 M; an 80 GB A100/H100 ~150 M;
* speed on the RTX 3050 laptop ~6e-8 s per cell per step; `build` prints the estimate;
* simulated time (`--ns`) must cover the slowest physics: spin waves crossing the device plus ring-down.
  `sparams` warns "not decayed" when the run was too short.
* Always look at `runs\NAME\geometry_preview.png` before a long run.

## New / high-performance PC

1. Install WSL Ubuntu (Windows) or use Linux directly, then build ARTEMIS (CUDA, `-DWarpX_MAG_LLG=ON`).
   Under WSL2 also apply `amrex_wsl_curand.patch` (see below).
2. Copy this folder. Edit `h2a_config.json`: `gpu_bin` / `cpu_bin` paths, `gpu_ranks` = number of GPUs.
3. Same commands as above. On Linux `h2a.py` starts ARTEMIS directly (no WSL).

## What is converted

| HFSS | ARTEMIS |
|---|---|
| 3D solids (vertical prisms) | voxelised boxes: eps, sigma (incl. tan d at band centre), mu |
| ferrite (4piMs, g, delta_H) + uniform Magnetic Bias | LLG: Ms, gamma = g/2 * 1.759e11, alpha = gamma' * delta_H / (2 f0), H_bias, M along bias |
| Finite Conductivity / Perfect E sheets | conductor `--metal-cells` cells thick |
| Lumped Port (Z0) | matched resistive sheet; driven port = soft-source Norton current Is(t) |
| Radiation / PML | PML on all faces (own air box, `--pad-*`) |
| sweep start/stop | Gaussian-modulated pulse covering the band |
| parametric bias value | used when the nominal bias variable is 0 |

S-parameters: `S11 = 2 V1/(Z0 Is) - 1`, `S21 = 2 V2/(Z0 Is)` with V = port voltage, Is = known source current.

## Limitations (v1)

* Non-prismatic solids (spheres, sloped sidewalls) fall back to their bounding box (warned).
* Wave ports are not converted: use lumped ports in HFSS.
* Arcs are approximated by start-mid-end segments.
* Exchange and anisotropy are off (fine for dipolar spin waves on >= 1 um cells).
* Uniform grid: features smaller than one cell vanish (check the preview).

## ARTEMIS notes for this laptop

* GPU binary `~/artemis_src/artemis/build_gpu/bin/warpx.3d.MPI.CUDA.DP.PDP.LLG`, CPU `~/artemis_src/artemis/build/bin/...OMP...LLG`.
* WSL2 + CUDA 12.4: AMReX's cuRAND host generator fails (error 203); `~/artemis_src/amrex_wsl_curand.patch`
  makes that non-fatal (ARTEMIS never uses it). Re-apply if AMReX is re-downloaded.
* ARTEMIS' `FieldProbe` diagnostic crashes in LLG builds; port voltages use `RawEFieldReduction` instead.

## Files

`h2a_gui.pyw` (desktop app) · `h2a.py` (command-line front end) · `hfss_export.py` · `build_inputs.py` · `sparams.py` · `h2a_config.json` ·
`models\` (exported designs) · `runs\` (simulations) · `.vscode\launch.json` (Run and Debug entries)
