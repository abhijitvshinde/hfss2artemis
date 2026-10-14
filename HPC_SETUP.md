# Setting up on a new / high-performance PC

Tested combination (laptop, 2026-10): ARTEMIS commit `4fa5cde3d7bd`, AMReX `9d2b74990c60` (fetched by CMake),
g++ 13 + CUDA 12.4, OpenMPI, Ubuntu (WSL2 on Windows 11).

## 1. Linux environment

* **Linux HPC**: use it directly.
* **Windows PC**: `wsl --install -d Ubuntu` in an Administrator PowerShell, reboot, create the Linux user.
  Never install an NVIDIA *Linux* driver inside WSL; it uses the Windows driver.

```bash
sudo apt update && sudo apt install -y build-essential cmake git libopenmpi-dev openmpi-bin python3-numpy python3-matplotlib
sudo apt install -y nvidia-cuda-toolkit      # GPU build; on a cluster use its CUDA module instead
nvidia-smi                                    # must list the GPU(s)
```

## 2. Build ARTEMIS (GPU)

```bash
mkdir -p ~/artemis_src && cd ~/artemis_src
git clone https://github.com/AMReX-Microelectronics/artemis.git
cd artemis && git checkout 4fa5cde3d7bd
cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release -DWarpX_COMPUTE=CUDA -DWarpX_MPI=ON -DWarpX_MAG_LLG=ON \
      -DWarpX_DIMS=3 -DWarpX_OPENPMD=OFF -DWarpX_QED=OFF -DAMReX_CUDA_ARCH=8.6
cmake --build build_gpu -j 8
```

* `-DAMReX_CUDA_ARCH`: 8.6 = RTX 30xx / A-series Ampere, 8.9 = RTX 40xx, 8.0 = A100, 9.0 = H100.
* CUDA 12.4 needs g++ <= 13: add `-DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_CUDA_HOST_COMPILER=g++-13` if the
  default g++ is newer.
* **WSL2 only**: if the binary aborts at start with `CURAND error 203`, apply the patch and rebuild:
  ```bash
  cd ~/artemis_src/artemis/build_gpu/_deps/fetchedamrex-src
  git apply /path/to/hfss2artemis/patches/amrex_wsl_curand.patch
  cd ~/artemis_src/artemis && cmake --build build_gpu -j 8
  ```
* CPU build (optional): same cmake line with `-B build -DWarpX_COMPUTE=OMP` and without the CUDA options.

Result: `~/artemis_src/artemis/build_gpu/bin/warpx.3d.MPI.CUDA.DP.PDP.LLG`

## 3. Get this tool

```bash
git clone https://github.com/abhijitvshinde/hfss2artemis.git
cd hfss2artemis
pip install -r requirements.txt        # or the apt packages above
python3 h2a.py list                    # creates h2a_config.json with defaults
```

Edit `h2a_config.json`: `gpu_bin` / `cpu_bin` (paths from step 2), `gpu_ranks` = number of GPUs to use.
On Windows run the same commands from PowerShell (or double-click `h2a_gui.pyw`); ARTEMIS is then started inside WSL.

## 4. Run a design that was already exported

The exported designs are in `models/` (no HFSS needed for anything below):

```bash
python3 h2a.py build models/YIG_horntransducer_GGG_09072026.json -o runs/horn_fine --dx 10 --dy 10 --dz 2.5 --ns 100
#   -> prints cells, GPU memory, run time; check runs/horn_fine/geometry_preview.png
python3 h2a.py run horn_fine
python3 h2a.py status --watch
python3 h2a.py sparams horn_fine
```

`h2a.py export` (reading a new `.aedt`) needs a machine with a licensed Ansys HFSS + `pip install pyaedt`;
copy the resulting `models/NAME.json` here.

## Sizing

~450 bytes of GPU memory per cell and ~6e-8 s per cell per step on an RTX 3050 laptop GPU (faster cards scale
roughly with memory bandwidth). `build` prints both estimates before anything runs; pass the measured cost of
the new machine with `--sec-per-cell-step` to make the time estimate accurate.
