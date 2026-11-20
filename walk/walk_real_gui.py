"""walk_real_gui.py - the trained connectome network walks the fly with a REAL fly's gait.

Run:  python walk_real_gui.py        (flywire environment, from this folder)

Same GUI as walk_trained_gui.py, for walk_real.py: the network learned flybody's real-fly
walking (motion capture) instead of the hand-made tripod.

Left    start / pause, speed (incl. slow motion), reset; which controller drives the legs;
        targets; camera; "real-fly ghost" (a translucent fly that does exactly what the real
        flies in the dataset do at this stride phase and turn, for comparison)
Centre  live MuJoCo rendering: red posts = targets, orange trail, blue ghost
Right   map: click = walk there, Shift-click = add a waypoint; motor neuron spikes (500 ms);
        joint angles of one leg, model (solid) vs real fly (dashed); network layer activity
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")   # off-screen rendering into the Tk window

import queue  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import tkinter as tk  # noqa: E402
from pathlib import Path  # noqa: E402
from tkinter import ttk  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import walk_real as WR  # noqa: E402

VIEW_SIZE = (600, 760)
ARENA_CM = 5.0
SPEEDS = {"real time": 1.0, "half speed": 0.5, "quarter speed": 0.25, "slow motion (1/10)": 0.1,
          "as fast as possible": None}
COURSE = [(2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]
START_TARGET = (3.0, 0.0)
GHOST = "G_"
CONTROLLERS = {   # label: policy file (None = the real-gait teacher, no network)
    "Trained network, real gait (imitation + CMA-ES round 2)": WR.RUNS / "cma_robust" / "policy.npz",
    "Trained network, real gait (imitation + CMA-ES)": WR.RUNS / "cma" / "policy.npz",
    "Trained network, real gait (imitation only)": WR.RUNS / "imitate" / "policy.npz",
    "Real-gait teacher, no network": None,
}
NOTES = {
    "Trained network, real gait (imitation + CMA-ES round 2)":
        "The connectome circuit (weights fixed) drives every motor neuron. It first learned to copy the "
        "real-gait teacher, then CMA-ES tuned it in MuJoCo to reach targets, stay upright and stay close "
        "to the real fly's joint angles; round 2 with more targets and longer walks.",
    "Trained network, real gait (imitation + CMA-ES)":
        "After the first CMA-ES round.",
    "Trained network, real gait (imitation only)":
        "Stage 1 only: the network copies the real-gait teacher's motor drive (BPTT).",
    "Real-gait teacher, no network":
        "What the network learns from: muscle drive that tracks real flies' joint angles (flybody motion "
        "capture, real_gait.py), sent straight to the motor neurons, no connectome.",
}
JOINT_COLOURS = {"coxa": "#d62728", "femur": "#1f77b4", "tibia": "#2ca02c", "tarsus": "#9467bd"}
PLOT_LEGS = ["T1_left", "T2_left", "T3_left", "T1_right", "T2_right", "T3_right"]


def _add(scn, gtype, size, pos, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


class RealWalkGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fly walking with a real fly's gait, driven by a trained connectome network")
        self.geometry("1760x960")
        self.circuit = WR.Circuit()
        self.controllers = {k: v for k, v in CONTROLLERS.items() if v is None or v.exists()}
        first = next(iter(self.controllers))
        self.sim = WR.Sim(self._theta(first), self.circuit)
        self.sim.targets, self.sim.loop_targets = [np.array(START_TARGET)], False
        self.seg = None
        self.last_seg = None
        self.cmds = queue.Queue()
        self.running, self.alive = False, True
        self.speed, self.rate = 1.0, 0.0
        self.mn_hist, self.layer_hist, self.joint_hist, self.kin_hist = [], [], [], []
        self.plot_leg = "T1_left"
        self._leg_cols()
        self._publish()
        # render model: the fly + a kinematic ghost (never simulated, only posed)
        self.rmodel, self.rdata, _ = WR.WT.build(1.0, flies=[(WR.PREFIX, 0.0, "fly", "torque", [1, 1, 1, 1]),
                                                              (GHOST, 0.0, "ghost", "torque", [1, 1, 1, 1])])
        self._ghost_setup()
        self.renderer = mujoco.Renderer(self.rmodel, *VIEW_SIZE)
        self.cams = self._make_cameras()
        mot = self.circuit.motor
        order = np.lexsort((mot.direction.to_numpy(), mot.joint.to_numpy(), mot.leg.to_numpy()))
        self.mn_y = np.empty(len(order), int)
        self.mn_y[order] = np.arange(len(order))
        self.mn_col = np.array([JOINT_COLOURS[j] for j in mot.joint])
        self.leg_ticks = [(self.mn_y[(mot.leg == leg).to_numpy()].mean(), leg) for leg in sorted(mot.leg.unique())]
        self.role_idx = [self.circuit.cmd, np.flatnonzero(self.circuit.nodes.role == "premotor"), self.circuit.mn]
        self._build_layout(first)
        self.physics = threading.Thread(target=self._physics_loop, daemon=True)
        self.physics.start()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(50, self._render_tick)
        self.after(300, self._plot_tick)
        self.after(500, self._toggle_run)

    def _theta(self, label):
        f = self.controllers[label]
        return None if f is None else np.load(f)["theta"]

    def _leg_cols(self):
        b = self.sim.body
        self.leg_k = [b.keys.index((self.plot_leg, j)) for j in ("coxa", "femur", "tibia", "tarsus")]

    # ================================================================ ghost
    def _ghost_setup(self):
        m = self.rmodel
        self.g_free = m.joint(GHOST + "free").qposadr[0]
        self.f_free = m.joint(WR.PREFIX + "free").qposadr[0]
        names = WR.GAIT.names
        self.g_qadr = np.array([m.joint(GHOST + n).qposadr[0] for n in names])
        for gi in range(m.ngeom):
            b = m.geom_bodyid[gi]
            if m.body(b).name.startswith(GHOST):
                m.geom_matid[gi] = -1
                m.geom_rgba[gi] = (0.25, 0.55, 1.0, 0.28)
                m.geom_contype[gi] = m.geom_conaffinity[gi] = 0
        self.g_off = np.zeros(3)

    def _pose_ghost(self, snap):
        """Ghost = the fly's body position / orientation, legs = the real fly at this phase."""
        q = self.rdata.qpos
        n = len(snap["qpos"])
        q[self.g_free:self.g_free + n] = snap["qpos"]              # whole pose (wings, head, ...)
        q[self.g_free:self.g_free + 3] += self.g_off
        q[self.g_qadr] = snap["q_ref"]

    # ================================================================ layout
    def _build_layout(self, first):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Big.TButton", font=("TkDefaultFont", 11, "bold"), padding=6)
        style.configure("Hint.TLabel", foreground="#555555")
        left = ttk.Frame(self, padding=8, width=380)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        box = ttk.LabelFrame(left, text="Run", padding=6)
        box.pack(fill="x")
        self.run_btn = ttk.Button(box, text="▶ Start walking", style="Big.TButton", command=self._toggle_run)
        self.run_btn.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(box, text="Speed").grid(row=1, column=0, sticky="w", pady=4)
        self.speed_var = tk.StringVar(value="real time")
        self.speed_var.trace_add("write", lambda *a: setattr(self, "speed", SPEEDS[self.speed_var.get()]))
        ttk.Combobox(box, textvariable=self.speed_var, values=list(SPEEDS), state="readonly",
                     width=20).grid(row=1, column=1, sticky="w")
        ttk.Button(box, text="Reset fly to the origin", command=self._reset).grid(row=2, column=0, columnspan=2,
                                                                                 sticky="ew")

        box = ttk.LabelFrame(left, text="Who controls the gait", padding=6)
        box.pack(fill="x", pady=8)
        self.ctrl_var = tk.StringVar(value=first)
        cb = ttk.Combobox(box, textvariable=self.ctrl_var, values=list(self.controllers), state="readonly", width=48)
        cb.pack(fill="x")
        cb.bind("<<ComboboxSelected>>", lambda e: self._set_controller())
        self.ctrl_note = ttk.Label(box, style="Hint.TLabel", wraplength=350, justify="left", text=NOTES[first])
        self.ctrl_note.pack(anchor="w", pady=(4, 6))
        c = self.circuit
        ttk.Label(box, style="Hint.TLabel", wraplength=350, justify="left", text=(
            f"Circuit: {c.n:,} neurons ({len(c.cmd)} descending, {len(self.role_idx[1]):,} premotor, "
            f"{len(c.mn)} motor, {len(c.sens)} leg sensors), {c.W.nnz:,} connections. Coxa / femur / tibia / "
            f"tarsus (24 joints) are moved only by motor neuron spikes; coxa abduction, coxa twist and "
            f"femur twist (18 joints, no motor neurons in the table) follow the real fly by servo.")
        ).pack(anchor="w")

        box = ttk.LabelFrame(left, text="Targets", padding=6)
        box.pack(fill="x")
        ttk.Label(box, text="Map (right): click = walk there,\nShift-click = add a waypoint.",
                  style="Hint.TLabel").pack(anchor="w")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Straight 3 cm run, facing:").pack(side="left")
        for txt, ang in (("ahead", 0), ("45°", 45), ("-45°", -45)):
            ttk.Button(row, text=txt, width=6, command=lambda a=ang: self._straight(a)).pack(side="left", padx=1)
        ttk.Button(box, text="Walk the square course (loops)", command=self._course).pack(fill="x", pady=(4, 0))
        ttk.Button(box, text="Clear targets (stand still)", command=self._clear).pack(fill="x", pady=4)

        box = ttk.LabelFrame(left, text="View", padding=6)
        box.pack(fill="x", pady=8)
        self.cam_var = tk.StringVar(value="close follow")
        r1 = ttk.Frame(box)
        r1.pack(fill="x")
        for cam in ("close follow", "follow", "side", "legs (low)", "top-down"):
            ttk.Radiobutton(r1, text=cam, value=cam, variable=self.cam_var).pack(side="left", padx=2)
        self.ghost_var = tk.StringVar(value="beside")
        r2 = ttk.Frame(box)
        r2.pack(fill="x", pady=(4, 0))
        ttk.Label(r2, text="Real-fly ghost:").pack(side="left")
        for g in ("off", "overlay", "beside"):
            ttk.Radiobutton(r2, text=g, value=g, variable=self.ghost_var).pack(side="left", padx=2)
        r3 = ttk.Frame(box)
        r3.pack(fill="x", pady=(4, 0))
        ttk.Label(r3, text="Joint plot leg:").pack(side="left")
        self.leg_var = tk.StringVar(value=self.plot_leg)
        lc = ttk.Combobox(r3, textvariable=self.leg_var, values=PLOT_LEGS, state="readonly", width=10)
        lc.pack(side="left", padx=4)
        lc.bind("<<ComboboxSelected>>", lambda e: self._set_plot_leg())

        box = ttk.LabelFrame(left, text="Status", padding=6)
        box.pack(fill="both", expand=True)
        self.status = ttk.Label(box, justify="left", font=("TkFixedFont", 9), wraplength=350)
        self.status.pack(anchor="w")

        self.view = ttk.Label(self)
        self.view.pack(side="left", padx=4, pady=8)

        right = ttk.Frame(self, padding=(0, 8, 8, 8))
        right.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(5.6, 9.0), dpi=100, constrained_layout=True)
        gs = self.fig.add_gridspec(4, 1, height_ratios=[4.2, 2.4, 2.2, 1.1])
        self.ax_map = self.fig.add_subplot(gs[0])
        self.ax_ras = self.fig.add_subplot(gs[1])
        self.ax_jnt = self.fig.add_subplot(gs[2])
        self.ax_lay = self.fig.add_subplot(gs[3])
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_map_click)

    def _make_cameras(self):
        thorax = self.rmodel.body(WR.PREFIX + "thorax").id
        cams = {}
        for name, (dist, elev, az) in {"close follow": (0.75, -20, 210), "follow": (1.3, -25, 200),
                                       "side": (0.8, -6, 90), "legs (low)": (0.6, -3, 160),
                                       "top-down": (7.0, -89, 90)}.items():
            c = mujoco.MjvCamera()
            c.type, c.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, thorax
            c.distance, c.elevation, c.azimuth = dist, elev, az
            cams[name] = c
        return cams

    # ============================================================ physics thread
    def _publish(self):
        s = self.sim
        t = s.t
        recent = [(tt, r) for tt, r in self.mn_hist[-6000:] if tt > t - 500]
        self.snap = {"qpos": s.d.qpos.copy(), "q_ref": s.body.q_ref.copy(), "trail": np.array(s.trail[-3000:]),
                     "targets": [np.array(x) for x in s.targets], "state": s.state(),
                     "raster": np.array(recent) if recent else np.empty((0, 2)),
                     "arrivals": len(s.arrivals), "layer": np.array(self.layer_hist[-500:]),
                     "joints": np.array(self.joint_hist[-150:]),
                     "kin": float(np.mean(self.kin_hist[-100:])) if self.kin_hist else 0.0,
                     "speed": self._speed(), "go": bool(s.targets) and s.go, "fallen": s.fallen(),
                     "seg": None if self.seg is None else [x.copy() if hasattr(x, "copy") else x for x in self.seg],
                     "last_seg": self.last_seg}

    def _speed(self):
        tr = self.sim.trail
        if len(tr) < 12:
            return 0.0
        return float(np.hypot(*(tr[-1] - tr[-11])) / 0.2)     # trail point every 20 ms -> last 200 ms

    def _physics_loop(self):
        wall0, sim0 = time.time(), self.sim.t
        last_pub = 0.0
        while self.alive:
            while not self.cmds.empty():
                self.cmds.get()(self.sim)
                wall0, sim0 = time.time(), self.sim.t
                self._publish()
            if not self.running:
                wall0, sim0 = time.time(), self.sim.t
                time.sleep(0.03)
                continue
            s = self.sim
            n_arr = len(s.arrivals)
            s.step()
            self._track_segment(s, n_arr)
            rows = np.flatnonzero(s.fired)
            if len(rows):
                self.mn_hist.extend((s.t, r) for r in rows)
                if len(self.mn_hist) > 60000:
                    self.mn_hist = self.mn_hist[-40000:]
            if int(s.t) % 4 == 0:
                b = s.body
                q = s.d.qpos[b.qadr][self.leg_k]
                qr = b.q_ref[b.ref_key][self.leg_k]
                self.joint_hist.append((s.t, *q, *qr))
                self.kin_hist.append(b.kin_err(s.d))
                if len(self.joint_hist) > 2000:
                    self.joint_hist = self.joint_hist[-1000:]
                    self.kin_hist = self.kin_hist[-1000:]
            if s.brain is not None and int(s.t) % 5 == 0:
                r = np.maximum(s.brain.x, 0)
                self.layer_hist.append((s.t, *[r[i].mean() for i in self.role_idx]))
                if len(self.layer_hist) > 4000:
                    self.layer_hist = self.layer_hist[-2000:]
            speed = self.speed
            if speed:
                ahead = (s.t - sim0) / 1000 / speed - (time.time() - wall0)
                if ahead > 0.002:
                    time.sleep(ahead)
                elif ahead < -0.5:
                    wall0, sim0 = time.time(), s.t
            now = time.time()
            if now - last_pub > 0.03:
                el = now - wall0
                self.rate = (s.t - sim0) / 1000 / el if el > 0.2 else self.rate
                self._publish()
                last_pub = now

    def _track_segment(self, s, n_arr):
        p = s.pos()
        if len(s.arrivals) > n_arr and self.seg is not None:
            a, b, dmax, path, t0 = self.seg
            self.last_seg = dict(line=np.hypot(*(b - a)), path=path, dmax=dmax, secs=s.t / 1000 - t0)
            self.seg = None
        tgt = s.targets[0] if s.targets and s.go else None
        if tgt is None:
            self.seg = None
            return
        if self.seg is None or not np.allclose(self.seg[1], tgt):
            self.seg = [p.copy(), np.array(tgt, float), 0.0, 0.0, s.t / 1000]
            self._prev = p.copy()
            return
        a, b = self.seg[0], self.seg[1]
        u = (b - a) / max(np.hypot(*(b - a)), 1e-9)
        dev = abs(u[0] * (p[1] - a[1]) - u[1] * (p[0] - a[0]))
        self.seg[2] = max(self.seg[2], dev)
        if int(s.t) % int(WR.PERIOD_MS) == 0:        # once per stride: ignore within-stride sway
            self.seg[3] += np.hypot(*(p - self._prev))
            self._prev = p.copy()

    def _cmd(self, fn):
        self.cmds.put(fn)

    def _clear_hist(self):
        self.mn_hist, self.layer_hist, self.joint_hist, self.kin_hist = [], [], [], []

    def _toggle_run(self):
        self.running = not self.running
        self.run_btn.configure(text="⏸ Pause" if self.running else "▶ Start walking")

    def _reset(self):
        def go(s):
            s.reset()
            s.targets, s.loop_targets = [np.array(START_TARGET)], False
            self._clear_hist()
            self.seg = self.last_seg = None
        self._cmd(go)

    def _straight(self, angle_deg):
        a = np.radians(angle_deg)
        tgt = np.array([3.0 * np.cos(a), 3.0 * np.sin(a)])

        def go(s):
            s.reset(heading_deg0=angle_deg)
            s.targets, s.loop_targets = [tgt], False
            self._clear_hist()
            self.seg = self.last_seg = None
        self._cmd(go)
        if not self.running:
            self._toggle_run()

    def _set_controller(self):
        label = self.ctrl_var.get()
        self.ctrl_note.configure(text=NOTES[label])
        theta = self._theta(label)

        def go(s):
            s.brain = None if theta is None else WR.Brain(self.circuit, theta)
            self.layer_hist = []
        self._cmd(go)

    def _set_plot_leg(self):
        leg = self.leg_var.get()

        def go(s):
            self.plot_leg = leg
            self._leg_cols()
            self.joint_hist = []
        self._cmd(go)

    def _course(self):
        self._cmd(lambda s: (setattr(s, "targets", [np.array(p) for p in COURSE]), setattr(s, "loop_targets", True)))

    def _clear(self):
        self._cmd(lambda s: setattr(s, "targets", []))

    def _on_map_click(self, event):
        if event.inaxes is not self.ax_map or event.xdata is None:
            return
        xy = np.array([round(event.xdata, 2), round(event.ydata, 2)])
        add = event.key is not None and "shift" in event.key

        def go(sim):
            sim.loop_targets = False
            sim.targets = (sim.targets + [xy]) if add else [xy]
        self._cmd(go)
        if not self.running:
            self._toggle_run()

    # ============================================================ drawing
    def _render_tick(self):
        if not self.alive:
            return
        snap = self.snap
        self.rdata.qpos[:] = 0
        self.rdata.qpos[self.f_free:self.f_free + len(snap["qpos"])] = snap["qpos"]
        ghost = self.ghost_var.get()
        if ghost == "off":
            self.g_off = np.array([0.0, 0.0, -5.0])          # parked under the floor
        elif ghost == "overlay":
            self.g_off = np.zeros(3)
        else:
            h = np.radians(snap["state"]["heading"])          # 0.35 cm to the fly's left
            self.g_off = np.array([-0.35 * np.sin(h), 0.35 * np.cos(h), 0.0])
        self._pose_ghost(snap)
        mujoco.mj_forward(self.rmodel, self.rdata)
        self.renderer.update_scene(self.rdata, self.cams[self.cam_var.get()])
        scn = self.renderer.scene
        for k, t in enumerate(snap["targets"]):
            _add(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.12 if k == 0 else 0.07, 0.15, 0],
                 [t[0], t[1], WR.FLOOR_Z + 0.15], (0.9, 0.1, 0.1, 0.9 if k == 0 else 0.45))
        for p in snap["trail"][::-1][:600:2]:
            _add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [p[0], p[1], WR.FLOOR_Z + 0.002],
                 (1.0, 0.55, 0.0, 1.0))
        img = ImageTk.PhotoImage(Image.fromarray(self.renderer.render()))
        self.view.configure(image=img)
        self.view.image = img
        self.after(40, self._render_tick)

    def _plot_tick(self):
        if not self.alive:
            return
        snap, st = self.snap, self.snap["state"]
        ax = self.ax_map
        ax.clear()
        tr = snap["trail"]
        if len(tr):
            ax.plot(tr[:, 0], tr[:, 1], color="#ff8c00", lw=1.4, label="path")
        if snap["targets"]:
            tx = np.array(snap["targets"])
            ax.scatter(tx[:, 0], tx[:, 1], s=80, marker="X", color="red", zorder=4, label="targets")
            ax.plot([st["x"], tx[0, 0]], [st["y"], tx[0, 1]], "--", color="green", lw=1)
        if snap["seg"] is not None:
            a, b = snap["seg"][0], snap["seg"][1]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#888888", lw=6, alpha=0.25, solid_capstyle="round",
                    label="ideal straight line")
        h = np.radians(st["heading"])
        ax.annotate("", xy=(st["x"] + 0.7 * np.cos(h), st["y"] + 0.7 * np.sin(h)), xytext=(st["x"], st["y"]),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
        ax.scatter([st["x"]], [st["y"]], s=35, color="black", zorder=5, label="fly")
        lim = max(ARENA_CM, abs(st["x"]) + 1, abs(st["y"]) + 1)
        ax.set_xlim(-lim / 2, lim); ax.set_ylim(-lim / 2, lim)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title("Map (cm): click = walk here, Shift-click = add waypoint", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")

        ax = self.ax_ras
        ax.clear()
        r = snap["raster"]
        if len(r):
            rows = r[:, 1].astype(int)
            ax.scatter(r[:, 0], self.mn_y[rows], s=2, c=self.mn_col[rows], linewidths=0)
        ax.set_xlim(st["t"] - 500, st["t"])
        ax.set_ylim(-2, len(self.mn_y) + 2)
        ax.set_yticks([y for y, _ in self.leg_ticks], [l for _, l in self.leg_ticks], fontsize=7)
        ax.set_title("Motor neuron spikes, last 500 ms (coxa red, femur blue, tibia green, tarsus purple)",
                     fontsize=8)
        ax.tick_params(labelsize=7)

        ax = self.ax_jnt
        ax.clear()
        J = snap["joints"]
        if len(J):
            for k, j in enumerate(("coxa", "femur", "tibia", "tarsus")):
                ax.plot(J[:, 0], np.degrees(J[:, 1 + k]), color=JOINT_COLOURS[j], lw=1.3, label=j)
                ax.plot(J[:, 0], np.degrees(J[:, 5 + k]), color=JOINT_COLOURS[j], lw=1, ls="--", alpha=0.8)
            ax.legend(fontsize=6, loc="upper left", ncol=4)
        ax.set_xlim(st["t"] - 600, st["t"])
        ax.set_ylabel("deg", fontsize=7)
        ax.set_title(f"{self.plot_leg} joint angles: model (solid) vs real fly (dashed)", fontsize=8)
        ax.tick_params(labelsize=7)

        ax = self.ax_lay
        ax.clear()
        L = snap["layer"]
        if len(L):
            for k, (name, col) in enumerate([("descending", "#444444"), ("premotor", "#ff7f0e"),
                                             ("motor", "#1f77b4")]):
                ax.plot(L[:, 0], L[:, k + 1] / max(L[:, k + 1].max(), 1e-6), color=col, lw=1, label=name)
            ax.legend(fontsize=6, loc="upper left", ncol=3)
        ax.set_xlim(st["t"] - 500, st["t"])
        ax.set_yticks([])
        ax.set_xlabel("time (ms)", fontsize=8)
        ax.set_title("Network layer activity (mean rate, normalised)", fontsize=8)
        ax.tick_params(labelsize=7)
        self.canvas.draw_idle()

        tilt = np.hypot(st["roll"], st["pitch"])
        upright = "FALLEN (press Reset)" if snap["fallen"] else "upright"
        who = self.ctrl_var.get()
        self.status.configure(text=(
            f"{'running' if self.running else 'paused'} - {'walking' if snap['go'] else 'standing (no target)'}\n"
            f"time      {st['t'] / 1000:7.1f} s   ({self.rate:.2f}x real time)\n"
            f"position  x {st['x']:+6.2f}  y {st['y']:+6.2f} cm\n"
            f"heading   {st['heading']:+6.1f} deg   turn input {st['turn']:+.2f}\n"
            f"speed     {snap['speed']:.2f} cm/s (real flies ~{WR.GAIT.v0:.1f})\n"
            f"stride    {1000 / WR.PERIOD_MS:.1f} Hz (real flies' median)\n"
            f"vs real   joint angles off by {np.degrees(snap['kin']):4.1f} deg (RMS, 42 joints)\n"
            f"body      {upright}: height {st['z']:.3f} cm\n"
            f"          roll {st['roll']:+5.1f}  pitch {st['pitch']:+5.1f} deg (tilt {tilt:.0f})\n"
            f"targets   {len(snap['targets'])} queued, {snap['arrivals']} reached\n"
            f"{self._line_text(snap)}"
            f"control   {who.split(' (')[0]}"))
        self.after(450, self._plot_tick)

    @staticmethod
    def _line_text(snap):
        out = ""
        if snap["seg"] is not None:
            a, b, dmax = snap["seg"][0], snap["seg"][1], snap["seg"][2]
            st = snap["state"]
            p = np.array([st["x"], st["y"]])
            u = (b - a) / max(np.hypot(*(b - a)), 1e-9)
            dev = abs(u[0] * (p[1] - a[1]) - u[1] * (p[0] - a[0]))
            out += (f"line      off the line {dev * 10:4.1f} mm now, max {dmax * 10:4.1f} mm\n"
                    f"          {np.hypot(*(b - p)):.2f} cm to go\n")
        r = snap["last_seg"]
        if r is not None:
            out += (f"last run  {r['line']:.2f} cm in {r['secs']:.1f} s, straightness "
                    f"{min(r['line'] / max(r['path'], 1e-9), 1) * 100:3.0f} %, max off-line {r['dmax'] * 10:.1f} mm\n")
        return out

    def _close(self):
        self.alive = False
        self.running = False
        self.physics.join(timeout=2)
        try:
            self.renderer.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    RealWalkGUI().mainloop()
