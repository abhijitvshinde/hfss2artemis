"""HFSS -> ARTEMIS desktop app.  Double-click this file (or the desktop shortcut) to open it.

Tab 1  New simulation   pick an HFSS .aedt, choose the grid, convert, check the preview, start
Tab 2  Runs & progress  every run with live progress, time left, GPU load; start / stop / resume / delete
Tab 3  Results          S11 / S21 and the port voltages of any run (works while it is still running)

All work is done by h2a.py (same folder); this window only drives it.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h2a  # noqa: E402

PY = sys.executable.replace("pythonw.exe", "python.exe")
ANSOFT = os.path.join(os.path.expanduser("~"), "Documents", "Ansoft")
LAPTOP_GPU_GB = 6.0


# ----------------------------------------------------------------------------- background workers
class ProcWorker(QThread):
    """Runs one 'h2a.py ...' command and streams its output."""
    line = Signal(str)
    finished_rc = Signal(int, str)

    def __init__(self, args, tag):
        super().__init__()
        self.args, self.tag, self.output = args, tag, []

    def run(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            p = subprocess.Popen([PY, "-u", os.path.join(HERE, "h2a.py")] + self.args, cwd=HERE, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 text=True, encoding="utf-8", errors="replace", **h2a.NOWIN)
        except OSError as e:
            self.line.emit(f"could not start h2a.py: {e}")
            self.finished_rc.emit(1, self.tag)
            return
        for ln in p.stdout:
            ln = ln.rstrip()
            self.output.append(ln)
            if "PyAEDT INFO" not in ln and not ln.startswith("next:") and "check with:" not in ln:
                self.line.emit(ln)
        self.finished_rc.emit(p.wait(), self.tag)


class StatusWorker(QThread):
    """Collects the state of every run (may call into WSL, so never on the GUI thread)."""
    result = Signal(list, object)

    def run(self):
        cfg, rows = h2a.config(), []
        for d in sorted(glob.glob(os.path.join(h2a.RUNS, "*"))):
            if not os.path.isfile(os.path.join(d, "inputs")):
                continue
            try:
                state, pr, ms, st = h2a.run_state(d, cfg)
            except Exception as e:  # keep the table alive even if one run is odd
                state, pr, ms, st = f"error: {e}", {"step": 0, "t_sim": 0, "sec_per_step": None, "errors": []}, 1, {}
            red = os.path.join(d, "diags", "reducedfiles", "V_port1.txt")
            cps = h2a.checkpoints(d)
            rows.append({"dir": d, "name": os.path.basename(d), "state": state, "step": pr["step"],
                         "max": ms or 1, "t_sim": pr["t_sim"], "sps": pr["sec_per_step"], "errors": pr["errors"],
                         "started": st.get("started"), "ended": st.get("ended"),
                         "resumed": st.get("resumed_from"), "checkpoint": os.path.basename(cps[-1]) if cps else None,
                         "samples": max(0, sum(1 for _ in open(red)) - 1) if os.path.isfile(red) else 0,
                         "slices": len(glob.glob(os.path.join(d, "diags", "slice*"))),
                         "mtime": max((os.path.getmtime(os.path.join(d, f)) for f in ("run.log", "inputs")
                                       if os.path.exists(os.path.join(d, f))), default=0)})
        self.result.emit(rows, h2a.gpu_line())


def hfss_designs(path):
    """Design names inside an .aedt file (read as text; no AEDT needed)."""
    try:
        txt = open(path, errors="ignore").read()
    except OSError:
        return []
    names = []
    for m in re.finditer(r"SolutionType='", txt):
        found = re.findall(r"^\t\tName='([^']+)'", txt[max(0, m.start() - 4000):m.start()], re.M)
        if found and found[-1] not in names:
            names.append(found[-1])
    return names


def fmt_dur(sec):
    return h2a.fmt_dur(sec)


def spin(lo, hi, val, step, dec=1, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(" " + suffix)
    return s


# ----------------------------------------------------------------------------- tab 1: new simulation
class NewTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.build_lines = []
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        g = QGroupBox("1. HFSS design")
        f = QFormLayout(g)
        self.aedt = QLineEdit()
        self.aedt.setPlaceholderText("choose an .aedt file")
        b = QPushButton("Browse...")
        b.clicked.connect(self.browse)
        row = QHBoxLayout()
        row.addWidget(self.aedt)
        row.addWidget(b)
        f.addRow("Project", row)
        self.design = QComboBox()
        self.design.setEditable(True)
        f.addRow("Design", self.design)
        self.name = QLineEdit()
        f.addRow("Run name", self.name)
        left.addWidget(g)

        g = QGroupBox("2. Grid and time")
        f = QFormLayout(g)
        self.dx = spin(0.5, 500, 20, 5, 1, "um")
        self.dy = spin(0.5, 500, 20, 5, 1, "um")
        self.same = QCheckBox("dy = dx")
        self.same.setChecked(True)
        self.same.toggled.connect(lambda on: self.dy.setEnabled(not on))
        self.dx.valueChanged.connect(lambda v: self.same.isChecked() and self.dy.setValue(v))
        self.dy.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self.dx)
        row.addWidget(QLabel("dy"))
        row.addWidget(self.dy)
        row.addWidget(self.same)
        f.addRow("dx (in-plane)", row)
        self.dz = spin(0.1, 500, 2.5, 0.5, 2, "um")
        f.addRow("dz (height)", self.dz)
        self.ns = spin(0.01, 10000, 3, 1, 2, "ns")
        f.addRow("Simulated time", self.ns)
        hint = QLabel("dz <= thinnest film / 2,  dx <= smallest gap / 2-3.\n"
                      "Time: ~3-5 ns for S11, 100+ ns for spin-wave S21.")
        hint.setStyleSheet("color: gray")
        f.addRow("", hint)
        left.addWidget(g)

        g = QGroupBox("3. Options (leave as is to use the HFSS values)")
        f = QFormLayout(g)
        self.use_band = QCheckBox("from HFSS")
        self.use_band.setChecked(True)
        self.fmin = spin(0.01, 1000, 2, 0.5, 3, "GHz")
        self.fmax = spin(0.01, 1000, 4, 0.5, 3, "GHz")
        for w in (self.fmin, self.fmax):
            w.setEnabled(False)
        self.use_band.toggled.connect(lambda on: [w.setEnabled(not on) for w in (self.fmin, self.fmax)])
        row = QHBoxLayout()
        row.addWidget(self.use_band)
        row.addWidget(self.fmin)
        row.addWidget(QLabel("to"))
        row.addWidget(self.fmax)
        f.addRow("Band", row)
        self.bias_on = QCheckBox("override")
        self.bias = spin(0, 100000, 500, 50, 1, "Oe")
        self.bias.setEnabled(False)
        self.bias_on.toggled.connect(self.bias.setEnabled)
        row = QHBoxLayout()
        row.addWidget(self.bias_on)
        row.addWidget(self.bias)
        f.addRow("Bias field", row)
        self.drive = QSpinBox()
        self.drive.setRange(1, 16)
        f.addRow("Excited port", self.drive)
        self.pad_xy = spin(0, 50, 0.2, 0.05, 2, "mm")
        self.pad_below = spin(0, 50, 0.1, 0.05, 2, "mm")
        self.pad_above = spin(0, 50, 0.3, 0.05, 2, "mm")
        row = QHBoxLayout()
        for lab, w in (("sides", self.pad_xy), ("below", self.pad_below), ("above", self.pad_above)):
            w.setMinimumWidth(95)
            row.addWidget(QLabel(lab))
            row.addWidget(w, 1)
        f.addRow("Air margin", row)
        self.chk = spin(0.001, 100, 0.25, 0.05, 3, "ns")
        f.addRow("Checkpoint every", self.chk)
        self.cpu = QCheckBox("run on CPU instead of GPU")
        f.addRow("", self.cpu)
        left.addWidget(g)

        row = QHBoxLayout()
        self.btn_convert = QPushButton("Convert && preview")
        self.btn_convert.setMinimumHeight(36)
        self.btn_convert.clicked.connect(self.convert)
        self.btn_start = QPushButton("Start simulation")
        self.btn_start.setMinimumHeight(36)
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        row.addWidget(self.btn_convert)
        row.addWidget(self.btn_start)
        left.addLayout(row)
        left.addStretch(1)
        w = QWidget()
        w.setLayout(left)
        w.setMinimumWidth(560)
        w.setMaximumWidth(640)
        lay.addWidget(w)

        right = QVBoxLayout()
        self.estimate = QLabel("Convert a design to see the grid size, memory and run time here.")
        self.estimate.setWordWrap(True)
        self.estimate.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.estimate.setStyleSheet("QLabel { background: #f4f6f8; padding: 8px; border-radius: 4px; }")
        right.addWidget(self.estimate)
        self.preview = QLabel("geometry preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(400, 250)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.preview)
        right.addWidget(scroll, 1)
        lay.addLayout(right, 1)
        self._pix = None

    # -- helpers
    def browse(self):
        start = os.path.dirname(self.aedt.text()) if self.aedt.text() else ANSOFT
        path, _ = QFileDialog.getOpenFileName(self, "HFSS project", start, "HFSS projects (*.aedt)")
        if not path:
            return
        self.aedt.setText(os.path.normpath(path))
        self.design.clear()
        self.design.addItems(hfss_designs(path))
        self.name.setText(re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(path))[0]))
        self.btn_start.setEnabled(False)

    def run_dir(self):
        return os.path.join(h2a.RUNS, self.name.text().strip())

    def model_path(self):
        return os.path.join(h2a.MODELS, self.name.text().strip() + ".json")

    def build_args(self):
        a = ["build", self.model_path(), "-o", self.run_dir(), "--force",
             "--dx", f"{self.dx.value():g}", "--dy", f"{(self.dx if self.same.isChecked() else self.dy).value():g}",
             "--dz", f"{self.dz.value():g}", "--ns", f"{self.ns.value():g}",
             "--pad-xy", f"{self.pad_xy.value():g}", "--pad-below", f"{self.pad_below.value():g}",
             "--pad-above", f"{self.pad_above.value():g}", "--checkpoint-every-ns", f"{self.chk.value():g}",
             "--drive", str(self.drive.value())]
        if not self.use_band.isChecked():
            a += ["--fmin", f"{self.fmin.value():g}", "--fmax", f"{self.fmax.value():g}"]
        if self.bias_on.isChecked():
            a += ["--bias-oe", f"{self.bias.value():g}"]
        return a

    # -- actions
    def convert(self):
        aedt, design, name = self.aedt.text().strip(), self.design.currentText().strip(), self.name.text().strip()
        if not os.path.isfile(aedt):
            return QMessageBox.warning(self, "HFSS project", "Choose an existing .aedt file first.")
        if not design:
            return QMessageBox.warning(self, "HFSS design", "Enter the design name.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            return QMessageBox.warning(self, "Run name", "Use only letters, digits, _ and - in the run name.")
        rd = self.run_dir()
        if os.path.isdir(rd) and os.path.exists(os.path.join(rd, "run.log")):
            if self.main.is_running(name):
                return QMessageBox.warning(self, "Run name", f"'{name}' is running right now. Pick another name.")
            if QMessageBox.question(self, "Replace run?",
                                    f"A simulation called '{name}' already exists.\nDelete its results and replace it?"
                                    ) != QMessageBox.Yes:
                return
            shutil.rmtree(rd, ignore_errors=True)
        self.btn_start.setEnabled(False)
        self.build_lines = []
        model = self.model_path()
        fresh = (os.path.isfile(model) and os.path.getmtime(model) > os.path.getmtime(aedt)
                 and json.load(open(model)).get("design") == design)
        if fresh:
            self.main.log(f"using the existing export {model} (the .aedt has not changed)")
            self.do_build()
        else:
            self.estimate.setText("Reading the HFSS project with AEDT (about a minute)...")
            self.main.run_h2a(["export", aedt, "--design", design, "-o", model], "export", self.after_export)

    def after_export(self, rc, worker):
        if rc != 0:
            self.estimate.setText("<b style='color:#b00'>Export failed</b> - see the log below.")
            return
        self.do_build()

    def do_build(self):
        self.estimate.setText("Building the ARTEMIS input...")
        self.main.run_h2a(self.build_args(), "build", self.after_build)

    def after_build(self, rc, worker):
        out = worker.output
        if rc != 0:
            self.estimate.setText("<b style='color:#b00'>Build failed</b> - see the log below.")
            return
        text = "\n".join(ln for ln in out if not ln.startswith("next:"))
        gb = re.search(r"estimated GPU memory ~([\d.]+) GB", text)
        warn = ""
        if gb and float(gb.group(1)) > LAPTOP_GPU_GB * 0.85 and not self.cpu.isChecked():
            warn = (f"<p style='color:#b00'><b>Needs ~{gb.group(1)} GB of GPU memory - more than this laptop's "
                    f"{LAPTOP_GPU_GB:g} GB.</b> Use larger cells or the high-performance PC.</p>")
        warnings = [ln.strip() for ln in out if ln.strip().startswith("warning:")]
        head = "<br>".join(ln for ln in out[:2])
        mats = "\n".join(ln[2:] for ln in out[2:] if ln.startswith("  ") and "warning" not in ln)
        self.estimate.setText(f"<b>{head}</b>{warn}<pre style='font-family:Consolas; font-size:9pt'>{mats}</pre>"
                              + ("<p style='color:#a60'>" + "<br>".join(warnings) + "</p>" if warnings else "")
                              + "<p>Check the preview, then press <b>Start simulation</b>.</p>")
        png = os.path.join(self.run_dir(), "geometry_preview.png")
        if os.path.isfile(png):
            self._pix = QPixmap(png)
            self.show_pix()
        self.btn_start.setEnabled(True)

    def show_pix(self):
        if self._pix and not self._pix.isNull():
            w = max(400, self.preview.parentWidget().width() - 20)
            self.preview.setPixmap(self._pix.scaledToWidth(w, Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.show_pix()

    def start(self):
        name = self.name.text().strip()
        args = ["run", name] + (["--cpu"] if self.cpu.isChecked() else [])
        self.btn_start.setEnabled(False)
        self.main.run_h2a(args, "run", lambda rc, w: self.main.after_start(rc, name))


# ----------------------------------------------------------------------------- tab 2: runs & progress
class RunsTab(QWidget):
    COLS = ["Run", "State", "Progress", "Simulated", "Time left", "Last activity"]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.rows = []
        lay = QVBoxLayout(self)
        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_detail)
        split.addWidget(self.table)

        box = QGroupBox("Selected run")
        g = QGridLayout(box)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(True)
        self.bar.setMinimumHeight(26)
        g.addWidget(self.bar, 0, 0, 1, 4)
        self.info = QLabel("select a run")
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info.setAlignment(Qt.AlignTop)
        f = QFont("Consolas")
        f.setPointSize(10)
        self.info.setFont(f)
        g.addWidget(self.info, 1, 0, 1, 4)
        self.btns = {}
        for i, (key, text) in enumerate((("run", "Start"), ("stop", "Stop"), ("resume", "Resume"),
                                         ("fresh", "Start over"), ("results", "Show results"),
                                         ("folder", "Open folder"), ("delete", "Delete run"),
                                         ("refresh", "Refresh now"))):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=key: self.action(k))
            g.addWidget(b, 2 + i // 4, i % 4)
            self.btns[key] = b
        split.addWidget(box)
        split.setSizes([260, 300])
        lay.addWidget(split)
        self.gpu = QLabel("")
        self.gpu.setStyleSheet("color: gray")
        lay.addWidget(self.gpu)

    def selected(self):
        r = self.table.currentRow()
        return self.rows[r] if 0 <= r < len(self.rows) else None

    def update_rows(self, rows, gpu):
        keep = self.selected()["name"] if self.selected() else None
        self.rows = rows
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            pct = 100.0 * r["step"] / r["max"]
            left = fmt_dur((r["max"] - r["step"]) * r["sps"]) if r["sps"] and r["state"].startswith("running") else ""
            import datetime as dt
            when = dt.datetime.fromtimestamp(r["mtime"]).strftime("%m-%d %H:%M") if r["mtime"] else ""
            vals = [r["name"], r["state"], f"{pct:5.1f}%  ({r['step']:,}/{r['max']:,})",
                    f"{r['t_sim']*1e9:.3f} ns", left, when]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 1:
                    color = {"running": "#0a7", "finished": "#06c", "failed": "#c00",
                             "stopped unexpectedly": "#c00"}.get(r["state"].split(" (")[0], "#555")
                    it.setForeground(QColor(color))
                self.table.setItem(i, j, it)
            if r["name"] == keep:
                self.table.selectRow(i)
        if keep is None and rows:
            newest = max(range(len(rows)), key=lambda k: rows[k]["mtime"])
            self.table.selectRow(newest)
        self.gpu.setText("GPU: " + gpu if gpu else "")
        self.show_detail()

    def show_detail(self):
        r = self.selected()
        busy = self.main.busy()
        if not r:
            self.info.setText("select a run")
            for b in self.btns.values():
                b.setEnabled(False)
            self.btns["refresh"].setEnabled(True)
            return
        pct = r["step"] / r["max"]
        self.bar.setValue(int(1000 * pct))
        self.bar.setFormat(f"{100*pct:.1f}%   step {r['step']:,} of {r['max']:,}")
        lines = [f"state      {r['state']}",
                 f"simulated  {r['t_sim']*1e9:.3f} ns"]
        if r["sps"]:
            left = (r["max"] - r["step"]) * r["sps"]
            import datetime as dt
            lines.append(f"speed      {r['sps']:.3f} s per step")
            if r["state"].startswith("running"):
                lines.append(f"time left  {fmt_dur(left)}   (finishes ~"
                             f"{(dt.datetime.now() + dt.timedelta(seconds=left)).strftime('%a %H:%M')})")
            elif r["step"] < r["max"]:
                lines.append(f"work left  {fmt_dur(left)}")
        if r["started"]:
            lines.append(f"started    {r['started']}" + (f"   (resumed from {r['resumed']})" if r["resumed"] else ""))
        if r["ended"]:
            lines.append(f"ended      {r['ended']}")
        lines.append(f"checkpoint {r['checkpoint'] or 'none yet'}")
        lines.append(f"outputs    {r['samples']} port samples, {r['slices']} field slices")
        for e in r["errors"]:
            lines.append(f"PROBLEM    {e}")
        self.info.setText("\n".join(lines))
        running = r["state"].startswith(("running", "starting", "stopping"))
        self.btns["run"].setEnabled(not busy and not running and r["step"] == 0)
        self.btns["stop"].setEnabled(not busy and running)
        self.btns["resume"].setEnabled(not busy and not running and r["checkpoint"] is not None
                                       and r["step"] < r["max"])
        self.btns["fresh"].setEnabled(not busy and not running)
        self.btns["results"].setEnabled(r["samples"] > 0)
        self.btns["folder"].setEnabled(True)
        self.btns["delete"].setEnabled(not busy and not running)
        self.btns["refresh"].setEnabled(True)

    def action(self, key):
        r = self.selected()
        if key == "refresh":
            return self.main.refresh()
        if not r:
            return
        name = r["name"]
        if key == "folder":
            return os.startfile(r["dir"])
        if key == "results":
            return self.main.show_results(name)
        if key == "stop":
            if QMessageBox.question(self, "Stop", f"Stop '{name}'?\nYou can resume it later from the last "
                                                  f"checkpoint ({r['checkpoint'] or 'none yet'}).") == QMessageBox.Yes:
                self.main.run_h2a(["stop", name], "stop", lambda rc, w: self.main.refresh())
        elif key == "resume":
            self.main.run_h2a(["resume", name], "resume", lambda rc, w: self.main.after_start(rc, name))
        elif key == "run":
            self.main.run_h2a(["run", name], "run", lambda rc, w: self.main.after_start(rc, name))
        elif key == "fresh":
            if QMessageBox.question(self, "Start over", f"Delete the results of '{name}' and run it again "
                                                        f"from step 0?") == QMessageBox.Yes:
                self.main.run_h2a(["run", name, "--fresh"], "run", lambda rc, w: self.main.after_start(rc, name))
        elif key == "delete":
            if QMessageBox.question(self, "Delete run", f"Permanently delete '{name}' and all its results?\n{r['dir']}"
                                    ) == QMessageBox.Yes:
                shutil.rmtree(r["dir"], ignore_errors=True)
                self.main.log(f"deleted {r['dir']}")
                self.main.refresh()


# ----------------------------------------------------------------------------- tab 3: results
class ResultsTab(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Run"))
        self.pick = QComboBox()
        self.pick.setMinimumWidth(220)
        row.addWidget(self.pick)
        self.btn = QPushButton("Compute / update S-parameters")
        self.btn.clicked.connect(self.compute)
        row.addWidget(self.btn)
        self.btn_csv = QPushButton("Open CSV")
        self.btn_csv.clicked.connect(lambda: self.open_file("sparams.csv"))
        row.addWidget(self.btn_csv)
        self.btn_geo = QPushButton("Geometry preview")
        self.btn_geo.clicked.connect(lambda: self.open_file("geometry_preview.png"))
        row.addWidget(self.btn_geo)
        row.addStretch(1)
        lay.addLayout(row)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.fig = Figure(figsize=(10, 4.5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(NavigationToolbar2QT(self.canvas, self))
        lay.addWidget(self.canvas, 1)

    def set_runs(self, names):
        cur = self.pick.currentText()
        self.pick.blockSignals(True)
        self.pick.clear()
        self.pick.addItems(names)
        if cur in names:
            self.pick.setCurrentText(cur)
        self.pick.blockSignals(False)

    def run_dir(self):
        return os.path.join(h2a.RUNS, self.pick.currentText())

    def open_file(self, fname):
        p = os.path.join(self.run_dir(), fname)
        if os.path.isfile(p):
            os.startfile(p)
        else:
            QMessageBox.information(self, "Not there yet", f"{fname} does not exist yet for this run.")

    def compute(self):
        if not self.pick.currentText():
            return
        self.note.setText("computing...")
        self.main.run_h2a(["sparams", self.pick.currentText()], "sparams", self.after_compute)

    def after_compute(self, rc, worker):
        msgs = [ln for ln in worker.output if ln.strip()]
        self.note.setText("<br>".join(msgs[-4:]).replace("(not decayed",
                                                         "<b style='color:#a60'>(not decayed") .replace(
            "truncated)", "truncated)</b>"))
        self.plot()

    def plot(self):
        import numpy as np
        import sparams
        rd = self.run_dir()
        self.fig.clear()
        a1, a2 = self.fig.add_subplot(1, 2, 1), self.fig.add_subplot(1, 2, 2)
        try:
            meta = json.load(open(os.path.join(rd, "ports.json")))
            red = os.path.join(rd, "diags", "reducedfiles")
            for p in meta["ports"]:
                t, v = sparams.port_voltage(os.path.join(red, p["probe"] + ".txt"), p["axis"], p["A"])
                a1.plot(t * 1e9, v, label=f"port {p['name']}" + (" (driven)" if p["driven"] else ""))
        except Exception as e:
            a1.text(0.5, 0.5, f"no port data yet\n{e}", ha="center", transform=a1.transAxes)
        a1.set_xlabel("time (ns)")
        a1.set_ylabel("port voltage (V)")
        a1.set_title("Port voltages")
        a1.grid(alpha=0.3)
        if a1.get_legend_handles_labels()[0]:
            a1.legend(loc="upper right")
        csv = os.path.join(rd, "sparams.csv")
        if os.path.isfile(csv):
            data = np.genfromtxt(csv, delimiter=",", names=True)
            for col in data.dtype.names:
                if col.startswith("dB_"):
                    a2.plot(data["f_GHz"], data[col], label=col[3:])
            a2.legend()
        else:
            a2.text(0.5, 0.5, "press 'Compute / update S-parameters'", ha="center", transform=a2.transAxes)
        a2.set_xlabel("frequency (GHz)")
        a2.set_ylabel("dB")
        a2.set_title(f"S-parameters - {os.path.basename(rd)}")
        a2.grid(alpha=0.3)
        self.canvas.draw_idle()


# ----------------------------------------------------------------------------- main window
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HFSS -> ARTEMIS")
        self.resize(1300, 860)
        self.worker = None
        self.status_worker = None
        self.last_rows = []

        central = QWidget()
        lay = QVBoxLayout(central)
        split = QSplitter(Qt.Vertical)
        self.tabs = QTabWidget()
        self.new_tab, self.runs_tab, self.results_tab = NewTab(self), RunsTab(self), ResultsTab(self)
        self.tabs.addTab(self.new_tab, "1   New simulation")
        self.tabs.addTab(self.runs_tab, "2   Runs && progress")
        self.tabs.addTab(self.results_tab, "3   Results")
        self.tabs.currentChanged.connect(lambda i: i == 2 and self.results_tab.plot())
        split.addWidget(self.tabs)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        self.console.setFont(QFont("Consolas", 9))
        split.addWidget(self.console)
        split.setSizes([680, 160])
        lay.addWidget(split)
        self.setCentralWidget(central)
        self.statusBar().showMessage("ready")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(15000)
        self.refresh()

    # -- plumbing
    def log(self, text):
        self.console.appendPlainText(text)

    def busy(self):
        return self.worker is not None and self.worker.isRunning()

    def run_h2a(self, args, tag, callback=None):
        if self.busy():
            QMessageBox.information(self, "Busy", "Another step is still working; wait for it to finish.")
            return
        self.log(f"> h2a.py {' '.join(args)}")
        self.statusBar().showMessage(f"working: {tag} ...")
        self.worker = ProcWorker(args, tag)
        self.worker.line.connect(self.log)
        self.worker.finished_rc.connect(lambda rc, t, w=self.worker, cb=callback: self.proc_done(rc, t, w, cb))
        self.worker.start()
        self.runs_tab.show_detail()

    def proc_done(self, rc, tag, worker, callback):
        self.statusBar().showMessage(f"{tag}: {'done' if rc == 0 else 'failed (see log)'}", 15000)
        if callback:
            callback(rc, worker)
        self.runs_tab.show_detail()
        self.refresh()

    def refresh(self):
        if self.status_worker is not None and self.status_worker.isRunning():
            return
        self.status_worker = StatusWorker()
        self.status_worker.result.connect(self.got_status)
        self.status_worker.start()

    def got_status(self, rows, gpu):
        self.last_rows = rows
        self.runs_tab.update_rows(rows, gpu)
        self.results_tab.set_runs([r["name"] for r in rows])

    def is_running(self, name):
        return any(r["name"] == name and r["state"].startswith(("running", "starting")) for r in self.last_rows)

    def after_start(self, rc, name):
        if rc == 0:
            self.tabs.setCurrentWidget(self.runs_tab)
        else:
            QMessageBox.warning(self, "Not started", "The simulation did not start; see the log at the bottom.")

    def show_results(self, name):
        self.results_tab.set_runs([r["name"] for r in self.last_rows])
        self.results_tab.pick.setCurrentText(name)
        self.tabs.setCurrentWidget(self.results_tab)
        self.results_tab.compute()

    def closeEvent(self, ev):
        if self.busy():
            if QMessageBox.question(self, "Quit", "A step is still working (export/build). Quit anyway?\n"
                                                  "Running simulations are not affected.") != QMessageBox.Yes:
                ev.ignore()
                return
        ev.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("HFSS -> ARTEMIS")
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
