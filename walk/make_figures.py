#!/usr/bin/env python3
"""Report figures for the walk_real project -> docs/figures/ (PNG, 200 dpi, + PDF).

    python make_figures.py              simulate (cached in runs_real/figdata/) and draw everything
    python make_figures.py --redo       re-run the simulations
    python make_figures.py --only paths,training

Figures
    fig1_training_imitation   imitation loss (runs_real/imitate/log.txt)
    fig2_training_cma         CMA-ES cost per generation, 4 runs (runs_real/*/log.csv)
    fig3_paths                top-down paths to 12 unseen targets, per controller
    fig4_stage_comparison     targets reached, falls, joint error, tilt per controller
    fig5_joint_tracking       one leg's 4 driven joints: network / teacher vs real fly, 3 strides
    fig6_joint_error_map      RMS joint error to the real fly per leg x joint (teacher vs network)
    fig7_gait_diagram         stance (dark) / swing per leg: real fly model, teacher, network
    fig8_neural_activity      motor-neuron spike raster + mean activity of each layer
    fig9_command_following    open-loop yaw rate vs turn command and speed vs go, before / after fine-tune
    fig10_filmstrip           rendered frames of the network walking (one stride)
Also writes docs/figures/results_summary.csv.
"""
import argparse
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
os.environ.setdefault("MUJOCO_GL", "egl")
import walk_real as W  # noqa: E402
import walk_real_cmd as WC  # noqa: E402

FIG = HERE.parent / "docs" / "figures"
CACHE = W.RUNS / "figdata"
POLICIES = {"Teacher (no network)": None, "Imitation": "imitate", "CMA-ES round 1": "cma",
            "CMA-ES round 2": "cma_robust", "Command fine-tune": "cmd2"}
N_TARGETS, N_SEEDS, EP_MS = 12, 3, 6000

# ---- style: reference categorical palette (dataviz skill), light surface, recessive axes ----------------
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                           "#e87ba4", "#008300", "#6250d6", "#e34948")
JOINT_COL = {"coxa": BLUE, "femur": ORANGE, "tibia": AQUA, "tarsus": VIOLET}
REAL = INK2                                   # the real fly / reference: neutral ink, dashed


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 8,
        "axes.titlelocation": "left", "figure.dpi": 100})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  -> docs/figures/{name}.png / .pdf")


def theta_of(run):
    return None if run is None else np.load(W.RUNS / run / "policy.npz")["theta"]


# =========================================================================================================
# simulations (worker processes keep one Sim and one GoSim each; every job reseeds the body's RNG)
# =========================================================================================================
_S = {}


def _init():
    _S["c"] = W.Circuit()


def _sim(kind):
    if kind not in _S:
        _S[kind] = (W.Sim if kind == "sim" else WC.GoSim)(None, _S["c"])
    return _S[kind]


def _set(sim, run, seed):
    th = theta_of(run)
    sim.brain = None if th is None else W.Brain(_S["c"], th)
    sim.body.rng = np.random.default_rng(seed)


def job_path(a):
    label, run, k, seed, target = a
    sim = _sim("sim")
    _set(sim, run, 1000 + k + 100 * seed)
    r = W.course_episode(sim, target, EP_MS)
    return dict(controller=label, k=k, seed=seed, tx=target[0], ty=target[1], trail=np.array(sim.trail), **r)


def job_cmd(a):
    run, go, turn, seed = a
    sim = _sim("go")
    _set(sim, run, seed)
    rng = np.random.default_rng(seed)
    rows, fell, t, kin = WC.episode(sim, [(0.0, go, turn)], 4000, heading=float(rng.uniform(-180, 180)))
    r = rows[0] if rows else dict(yaw=np.nan, fwd=np.nan)
    return dict(run=run, go=go, turn=turn, seed=seed, yaw_deg_s=np.degrees(r["yaw"]), fwd=r["fwd"], fell=fell)


def job_record(a):
    """Straight walk (turn 0) for `ms`: per-ms joint angles, real-fly reference, claw fore-aft position,
    motor spikes and layer activity."""
    run, ms = a
    sim = _sim("sim")
    _set(sim, run, 7)
    sim.reset()
    sim.targets, sim.turn_override, sim.go = [], 0.0, True
    b, d, c = sim.body, sim.d, _S["c"]
    claws = [sim.m.body(f"{W.PREFIX}claw_{leg}").id for leg in W.LEGS]
    pm = np.flatnonzero(c.nodes.role == "premotor")
    out = {k: [] for k in ("q", "qref", "claw", "fired", "layers", "phi")}
    while sim.t < ms and not sim.fallen():
        sim.step()
        R = d.xmat[b.thorax].reshape(3, 3)
        rel = (d.xpos[claws] - d.xpos[b.thorax]) @ R                 # body frame
        out["q"].append(d.qpos[b.qadr].copy())
        out["qref"].append(b.q_ref[b.ref_key].copy())
        out["claw"].append(rel[:, 0])
        out["fired"].append(sim.fired.copy())
        out["phi"].append(W.stride_phase(sim.t))
        if sim.brain is not None:
            r = np.maximum(sim.brain.x, 0)
            out["layers"].append([r[c.cmd].mean(), r[pm].mean(), r[c.mn].mean()])
    o = {k: np.array(v) for k, v in out.items()}
    o["keys"] = np.array([f"{l}|{j}" for l, j in b.keys])
    o["sign"] = b.sign
    return run, o


def simulate(redo=False, workers=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    f_paths, f_cmd, f_rec = CACHE / "paths.pkl", CACHE / "cmd.csv", CACHE / "record.npz"
    if not redo and f_paths.exists() and f_cmd.exists() and f_rec.exists():
        return pd.read_pickle(f_paths), pd.read_csv(f_cmd), dict(np.load(f_rec, allow_pickle=True))
    rng = np.random.default_rng(100)
    targets = [W.random_target(rng, (1.5, 2.0)) for _ in range(N_TARGETS)]
    jp = [(lab, run, k, s, t) for lab, run in POLICIES.items() for s in range(N_SEEDS) for k, t in enumerate(targets)]
    turns = np.round(np.arange(-1, 1.01, 0.25), 2)
    jc = [(run, 1.0, tv, s) for run in ("cma_robust", "cmd2") for tv in turns for s in range(4)]
    jc += [(run, g, 0.0, 10 + s) for run in ("cma_robust", "cmd2") for g in (0.0, 0.5, 0.75) for s in range(4)]
    jr = [(run, 1500) for run in (None, "cma_robust", "cmd2")]
    print(f"simulating: {len(jp)} target runs, {len(jc)} command runs, {len(jr)} recordings on {workers} workers",
          flush=True)
    with get_context("fork").Pool(workers, _init) as pool:
        rec_async = pool.map_async(job_record, jr)
        P = pd.DataFrame(pool.map(job_path, jp))
        C = pd.DataFrame(pool.map(job_cmd, jc))
        R = rec_async.get()
    rec = {}
    for run, o in R:
        for k, v in o.items():
            rec[f"{run or 'teacher'}/{k}"] = v
    P.to_pickle(f_paths)
    C.to_csv(f_cmd, index=False)
    np.savez_compressed(f_rec, **rec)
    return P, C, rec


# =========================================================================================================
# figures
# =========================================================================================================
def fig_training():
    import matplotlib.pyplot as plt
    import re
    txt = (W.RUNS / "imitate" / "log.txt").read_text()
    it = np.array([(int(a), float(b)) for a, b in re.findall(r"iter\s+(\d+)\s+imitation loss ([\d.]+)", txt)])
    it = it[: np.argmax(it[1:, 0] < it[:-1, 0]) + 1] if np.any(it[1:, 0] < it[:-1, 0]) else it   # first round only
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(it[:, 0], it[:, 1], color=BLUE)
    ax.set(xlabel="Adam iteration", ylabel="MSE to teacher's pool drive", ylim=(0, None))
    ax.set_title("Imitation: network copies the teacher's motor drive")
    ax.annotate(f"{it[0, 1]:.3f}", (it[0, 0], it[0, 1]), xytext=(6, -2), textcoords="offset points", color=INK2)
    ax.annotate(f"{it[-1, 1]:.3f}", (it[-1, 0], it[-1, 1]), xytext=(-8, 8), textcoords="offset points",
                color=INK2, ha="right")
    save(fig, "fig1_training_imitation")

    runs = [("cma", "CMA-ES round 1: reach targets", "from imitation, K_REALISM 4"),
            ("cma_robust", "CMA-ES round 2: robust + realistic", "from round 1, K_REALISM 12, 4 targets"),
            ("cmd", "Command fine-tune round 1", "from round 2, go ∈ {0, 1}"),
            ("cmd2", "Command fine-tune round 2", "from cmd, go ∈ {0, 0.5, 0.75, 1}")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.1))
    for ax, (run, title, sub) in zip(axs, runs):
        L = pd.read_csv(W.RUNS / run / "log.csv")
        ax.plot(L.gen, L["mean"], color=MUTED, lw=1.2, label="population mean")
        ax.plot(L.gen, L.best, color=BLUE, lw=1.2, alpha=0.55, label="generation best")
        ax.step(L.gen, L.best_so_far, where="post", color=BLUE, lw=2.2, label="kept (re-checked) best")
        ax.set_title(title, fontsize=9.5, pad=15)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
        ax.set_xlabel("generation (22 candidates each)")
        h = L.seconds.iloc[-1] / 60
        ax.text(0.98, 0.96, f"{h:.0f} min", transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8)
    axs[0].set_ylabel("cost (lower is better)")
    axs[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.88))
    fig.suptitle("CMA-ES training: cost per generation (each panel has its own cost definition and scale)",
                 x=0.01, ha="left", fontsize=10, color=INK2, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_training_cma")


def fig_paths(P):
    import matplotlib.pyplot as plt
    labs = list(POLICIES)
    fig, axs = plt.subplots(1, len(labs), figsize=(3.0 * len(labs), 3.4), sharex=True, sharey=True)
    for ax, lab in zip(axs, labs):
        D = P[P.controller == lab]
        A = D
        D = D[D.seed == 0]
        for _, r in D.iterrows():
            tr = r.trail
            ax.plot(tr[:, 0], tr[:, 1], color=BLUE, lw=1.1, alpha=0.85)
            ax.add_patch(plt.Circle((r.tx, r.ty), W.ARRIVE_CM, fill=False, ec=MUTED, lw=0.8, ls=":"))
            mk = dict(marker="o", ms=5, mfc=GREEN, mec=SURF) if r.reached else dict(marker="o", ms=5, mfc=SURF, mec=INK2)
            ax.plot(r.tx, r.ty, **mk)
            if r.fell:
                ax.plot(tr[-1, 0], tr[-1, 1], marker="x", color=RED, ms=7, mew=2)
        ax.plot(0, 0, marker="s", color=INK, ms=4)
        ax.set_aspect("equal")
        ax.set_title(lab, fontsize=9.5)
        ax.text(0.02, 0.02, f"shown: {int(D.reached.sum())}/{len(D)} reached, {int(D.fell.sum())} falls\n"
                f"all {len(A)} runs: {int(A.reached.sum())} reached, {int(A.fell.sum())} falls",
                transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
        ax.set_xlabel("x (cm)")
    axs[0].set_ylabel("y (cm)")
    from matplotlib.lines import Line2D
    hand = [Line2D([], [], color=BLUE, lw=1.2, label="path"),
            Line2D([], [], ls="", marker="o", mfc=GREEN, mec=SURF, ms=6, label="target reached"),
            Line2D([], [], ls="", marker="o", mfc=SURF, mec=INK2, ms=6, label="target missed"),
            Line2D([], [], ls="", marker="x", color=RED, mew=2, ms=7, label="fell")]
    fig.legend(handles=hand, loc="upper right", ncol=4, bbox_to_anchor=(1.0, 1.06))
    fig.suptitle(f"Walking to {N_TARGETS} unseen targets 1.5-2 cm away ({EP_MS / 1000:.0f} s each, start at the square; seed 0 shown)",
                 x=0.01, ha="left", y=1.06, fontsize=10)
    fig.tight_layout()
    save(fig, "fig3_paths")


def summary(P):
    g = P.groupby("controller", sort=False)
    S = pd.DataFrame({"targets reached": g.reached.sum().astype(int), "episodes": g.size(),
                      "falls": g.fell.sum().astype(int),
                      "time to target (s, median)": g.apply(lambda d: np.median(d.t[d.reached]) / 1000
                                                            if d.reached.any() else np.nan),
                      "joint error (rad)": g.kin_err.mean(), "tilt (deg)": g.tilt_deg.mean(),
                      "speed (cm/s)": g.speed.mean()}).reindex(list(POLICIES))
    S.to_csv(FIG / "results_summary.csv", float_format="%.3f")
    print(S.to_string(float_format=lambda v: f"{v:.2f}"))
    return S


def fig_stage(S):
    import matplotlib.pyplot as plt
    labs = ["Teacher\n(no network)", "Imitation", "CMA-ES\nround 1", "CMA-ES\nround 2", "Command\nfine-tune"]
    cols = [REAL, MUTED, MUTED, BLUE, MUTED]
    n_ep = N_TARGETS * N_SEEDS
    mets = [("targets reached", "Targets reached (of %d runs)" % n_ep, "{:.0f}"),
            ("falls", "Falls (of %d runs)" % n_ep, "{:.0f}"),
            ("joint error (rad)", "Joint error to real fly (rad, RMS)", "{:.2f}"),
            ("tilt (deg)", "Body tilt (deg, RMS)", "{:.1f}")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.0))
    x = np.arange(len(labs))
    for ax, (col, title, fmt) in zip(axs, mets):
        v = S[col].to_numpy(float)
        ax.bar(x, np.nan_to_num(v), color=cols, width=0.62, edgecolor=SURF, linewidth=2)
        for xi, vi in zip(x, v):
            ax.text(xi, (0 if np.isnan(vi) else vi), "n/a" if np.isnan(vi) else fmt.format(vi), ha="center",
                    va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x, labs, fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    fig.suptitle("Controller comparison on the same unseen targets (blue: the default trained network)",
                 x=0.01, ha="left", y=1.04, fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_stage_comparison")


def _rec(rec, run, k):
    return rec[f"{run}/{k}"]


def fig_joints(rec):
    import matplotlib.pyplot as plt
    leg = "T1_left"
    keys = list(_rec(rec, "cma_robust", "keys"))
    t0, t1 = 600, 915                                   # ~3 strides after start-up
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 6.4), sharex=True)
    for ax, j in zip(axs, W.JOINTS if hasattr(W, "JOINTS") else ["coxa", "femur", "tibia", "tarsus"]):
        k = keys.index(f"{leg}|{j}")
        s = _rec(rec, "cma_robust", "sign")[k]
        tt = np.arange(t0, t1)
        ref = np.degrees(s * _rec(rec, "cma_robust", "qref")[t0:t1, k])
        ax.plot(tt, ref, color=REAL, ls="--", lw=1.6, label="real fly (motion-capture model)")
        ax.plot(tt, np.degrees(s * _rec(rec, "teacher", "q")[t0:t1, k]), color=ORANGE, lw=1.4, alpha=0.9,
                label="teacher (no network)")
        ax.plot(tt, np.degrees(s * _rec(rec, "cma_robust", "q")[t0:t1, k]), color=BLUE, lw=2, label="trained network")
        ax.set_ylabel(f"{j} (deg)")
    axs[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02))
    axs[-1].set_xlabel("time (ms)")
    fig.suptitle(f"Joint angles of the {leg.replace('_', ' ')} leg over 3 strides (anatomical sign: + = promote / extend / levate)",
                 x=0.01, ha="left", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, "fig5_joint_tracking")

    # per leg x joint RMS error
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.0))
    legs = ["T1_left", "T2_left", "T3_left", "T1_right", "T2_right", "T3_right"]
    js = ["coxa", "femur", "tibia", "tarsus"]
    vmax = 0
    mats = {}
    for run in ("teacher", "cma_robust"):
        q, qr = _rec(rec, run, "q")[200:], _rec(rec, run, "qref")[200:]
        e = np.degrees(np.sqrt(np.mean((q - qr) ** 2, 0)))
        M = np.array([[e[keys.index(f"{l}|{j}")] for j in js] for l in legs])
        mats[run] = M
        vmax = max(vmax, M.max())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fc", "#9cc1ee", BLUE, "#123a6b"])
    for ax, (run, title) in zip(axs, [("teacher", "Teacher (no network)"), ("cma_robust", "Trained network (round 2)")]):
        M = mats[run]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color=SURF if M[i, j] > 0.55 * vmax else INK)
        ax.set_xticks(range(4), js)
        ax.set_yticks(range(6), [l.replace("_", " ") for l in legs])
        ax.grid(False)
        ax.set_title(title, fontsize=9.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.02)
    cb.set_label("RMS error to real fly (deg)")
    cb.outline.set_visible(False)
    fig.suptitle("Where the walk differs from the real fly: RMS joint-angle error, straight walking",
                 x=0.01, ha="left", y=1.03, fontsize=10)
    save(fig, "fig6_joint_error_map")


def fig_gait(rec):
    import matplotlib.pyplot as plt
    t0, t1 = 600, 1020
    legs = ["T1_left", "T2_left", "T3_left", "T1_right", "T2_right", "T3_right"]
    gl = list(W.GAIT_LEGS)
    phi = _rec(rec, "cma_robust", "phi")[t0:t1]
    real = np.array([W.GAIT.in_stance(p)[[gl.index(l) for l in legs]] for p in phi]) > 0.5   # P(stance)

    def stance(run):
        x = _rec(rec, run, "claw")[:, [W.LEGS.index(l) for l in legs]]
        k = np.ones(9) / 9
        xs = np.array([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])]).T
        return (np.gradient(xs, axis=0) < 0)[t0:t1]            # claw moving backwards = stance

    rows = [("Real fly (gait model)", real), ("Teacher (no network)", stance("teacher")),
            ("Trained network", stance("cma_robust"))]
    fig, axs = plt.subplots(len(rows), 1, figsize=(7.2, 5.2), sharex=True)
    tt = np.arange(t0, t1)
    for ax, (title, S) in zip(axs, rows):
        for i in range(S.shape[1]):
            on = S[:, i].astype(int)
            edges = np.flatnonzero(np.diff(np.r_[0, on, 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                ax.add_patch(plt.Rectangle((tt[a], i - 0.36), b - a, 0.72, color=INK2 if title.startswith("Real") else BLUE,
                                           lw=0))
        ax.set_ylim(-0.6, S.shape[1] - 0.4)
        ax.set_xlim(t0, t1)
        ax.invert_yaxis()
        ax.set_yticks(range(len(legs)), [l.replace("_", " ") for l in legs], fontsize=7.5)
        ax.grid(False)
        ax.set_title(title, fontsize=9.5)
    axs[-1].set_xlabel("time (ms)")
    fig.suptitle("Gait diagram: bars = stance (claw moving backwards); tripods T1L-T2R-T3L and T1R-T2L-T3R",
                 x=0.01, ha="left", y=1.0, fontsize=10)
    fig.tight_layout()
    save(fig, "fig7_gait_diagram")


def fig_neural(rec):
    import matplotlib.pyplot as plt
    c = W.Circuit()
    mot = c.motor
    order = np.lexsort((mot.direction.to_numpy(), mot.joint.map({j: i for i, j in enumerate(JOINT_COL)}).to_numpy(),
                        mot.leg.map({l: i for i, l in enumerate(W.LEGS)}).to_numpy()))
    ypos = np.empty(len(order), int)
    ypos[order] = np.arange(len(order))
    t0, t1 = 800, 1100
    F = _rec(rec, "cma_robust", "fired")[t0:t1]
    fig, (a1, *a2) = plt.subplots(4, 1, figsize=(7.2, 7.4), sharex=True, gridspec_kw={"height_ratios": [3, .7, .7, .7]})
    for j, col in JOINT_COL.items():
        m = (mot.joint == j).to_numpy()
        ti, ni = np.nonzero(F[:, m])
        a1.scatter(t0 + ti, ypos[np.flatnonzero(m)[ni]], s=1.2, color=col, lw=0, label=j)
    for i, leg in enumerate(W.LEGS):
        ys = ypos[(mot.leg == leg).to_numpy()]
        a1.axhline(ys.max() + 0.5, color=GRID, lw=0.8)
        a1.text(t1 + 3, ys.mean(), leg.replace("_", " "), va="center", fontsize=7.5, color=INK2)
    a1.set_ylim(len(order), -1)
    a1.set_ylabel("motor neuron (by leg, joint)")
    a1.set_yticks([])
    a1.grid(False)
    a1.legend(markerscale=6, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0))
    a1.set_title("Motor-neuron spikes of the trained network, straight walking (302 neurons)", pad=22)
    L = _rec(rec, "cma_robust", "layers")[t0:t1]
    tt = np.arange(t0, t1)
    for i, (ax, lab) in enumerate(zip(a2, ["descending (81)", "premotor (2,205)", "motor (302)"])):
        ax.plot(tt, L[:, i], color=BLUE, lw=1.6)
        ax.set_ylabel(lab, fontsize=7.5)
        ax.margins(y=0.25)
    a2[0].set_title("Mean rate relu(x) of each layer (own scale per row)", fontsize=9.5)
    a2[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    save(fig, "fig8_neural_activity")


def fig_cmd(C):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.6))
    T = C[C.go == 1.0]
    for run, lab, col in (("cma_robust", "before fine-tune (CMA-ES round 2)", MUTED), ("cmd2", "after fine-tune", BLUE)):
        g = T[T.run == run].groupby("turn")
        m, s = g.yaw_deg_s.mean(), g.yaw_deg_s.std()
        a1.fill_between(m.index, m - s, m + s, color=col, alpha=0.15, lw=0)
        a1.plot(m.index, m, color=col, marker="o", ms=4, label=lab)
        falls = g.fell.sum()
        for tv, f in falls.items():
            if f:
                a1.text(tv, m[tv] + s[tv] + 6, f"{int(f)}✕", ha="center", fontsize=7, color=RED)
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set(xlabel="turn command (+ = right)", ylabel="yaw rate (deg/s, + = left)")
    a1.set_title("Open-loop turning (go 1)", pad=14)
    a1.text(0, 1.0, "4 s, 4 seeds; band = ±1 SD; red = falls", transform=a1.transAxes, fontsize=7.5, color=INK2, va="bottom")
    a1.legend(loc="lower left", fontsize=7.5)
    G = pd.concat([C[(C.turn == 0) & (C.go < 1)], T[T.turn == 0]])
    w = 0.36
    gos = sorted(G.go.unique())
    for k, (run, lab, col) in enumerate((("cma_robust", "before", MUTED), ("cmd2", "after", BLUE))):
        g = G[G.run == run].groupby("go").fwd
        m, s = g.mean().reindex(gos), g.std().reindex(gos)
        x = np.arange(len(gos)) + (k - 0.5) * w
        a2.bar(x, m, width=w, color=col, edgecolor=SURF, linewidth=2, label=lab)
        a2.errorbar(x, m, yerr=s, fmt="none", ecolor=INK2, lw=1, capsize=2)
    a2.set_xticks(np.arange(len(gos)), [f"{g:g}" for g in gos])
    a2.set(xlabel="go command", ylabel="forward speed (cm/s)")
    a2.set_title("Speed vs go (turn 0)")
    a2.grid(axis="x", visible=False)
    a2.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig9_command_following")


def fig_film():
    import matplotlib.pyplot as plt
    import mujoco
    c = W.Circuit()
    sim = W.Sim(theta_of("cma_robust"), c)
    sim.body.rng = np.random.default_rng(7)
    sim.reset()
    sim.targets, sim.turn_override, sim.go = [], 0.0, True
    r = mujoco.Renderer(sim.m, 360, 480)
    cam = mujoco.MjvCamera()
    cam.type, cam.trackbodyid = mujoco.mjtCamera.mjCAMERA_TRACKING, sim.body.thorax
    cam.distance, cam.elevation, cam.azimuth = 0.62, -12, 150
    T0, n = 700, 6
    times = [T0 + round(i * W.PERIOD_MS / n) for i in range(n)]
    frames = []
    while sim.t <= times[-1]:
        sim.step()
        if int(sim.t) in times:
            r.update_scene(sim.d, cam)
            frames.append((int(sim.t), r.render().copy()))
    fig, axs = plt.subplots(1, n, figsize=(2.3 * n, 2.0))
    for ax, (t, im) in zip(axs, frames):
        ax.imshow(im[40:320, 40:440])
        ax.set_axis_off()
        ax.set_title(f"{t - times[0]} ms", fontsize=8.5, color=INK2, loc="center", fontweight="normal")
    fig.suptitle(f"Trained network walking: one stride ({W.PERIOD_MS:.0f} ms)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    save(fig, "fig10_filmstrip")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    style()
    want = lambda k: only is None or k in only  # noqa: E731
    if want("training"):
        fig_training()
    if want("film"):
        fig_film()
    if only is None or only & {"paths", "stage", "joints", "gait", "neural", "cmd"}:
        P, C, rec = simulate(a.redo, a.workers)
        if want("paths"):
            fig_paths(P)
        S = summary(P)
        if want("stage"):
            fig_stage(S)
        if want("joints"):
            fig_joints(rec)
        if want("gait"):
            fig_gait(rec)
        if want("neural"):
            fig_neural(rec)
        if want("cmd"):
            fig_cmd(C)


if __name__ == "__main__":
    main()
