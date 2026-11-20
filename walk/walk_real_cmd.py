#!/usr/bin/env python3
"""Fine-tune the walk_real network to FOLLOW (go, turn) COMMANDS open-loop (CMA-ES, as fly_arm).

Why: walk_real was only ever steered closed-loop, with the turn command saturating at +/-1. Driven
open-loop (fly-taste-learning T0b), it veers left at turn 0 (+76 deg/s), and capsizes at intermediate
turns (100% at turn -0.5). A downstream brain needs a walker that follows any command in its range.

Same network, same learned parameters (E, P, gains, g), same body; walk_real.py is unchanged.
Starts from runs_real/cma_robust. Each candidate walks command schedules: segments of 0.6-1.0 s
with turn ~ U[-1, 1] (some exactly 0 or +/-1), go = 1 (sometimes 0). Cost per episode:
    yaw tracking   mean over segments of ((yaw - yaw*) / YAW_REF)^2, yaw* = -turn * YAW_REF
                   (yaw measured over each segment after SETTLE_MS; + = left, turn + = right)
    speed          go = 1: max(0, V_MIN - forward) / V_MIN;  go = 0: |forward| / V_MIN
    falls          FALL_PENALTY x (fraction of the episode left)
    realism        K_REALISM x joint-angle error to the real fly (rad), as walk_real
The best candidate of a generation is re-checked on fixed validation schedules before it is kept.

    python walk_real_cmd.py train --gens 80            -> runs_real/cmd/policy.npz
    python walk_real_cmd.py train --start runs_real/cmd/policy.npz --gens 60 --episodes 3 \
           --go-levels 0.5,0.75,1 --out cmd2                    (round 2: intermediate go levels)
    python walk_real_cmd.py eval runs_real/cmd/policy.npz
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import walk_real as W

YAW_REF = np.radians(90.0)        # rad/s at |turn| = 1 (what the closed-loop policy reached at +/-1)
V_MIN = 0.8                       # cm/s expected forward speed when walking
SETTLE_MS = 300.0
FALL_PENALTY = 6.0
K_REALISM = 4.0
EP_MS = 3000.0
_W = {}
GO_LEVELS = (1.0,)                # round 1: go in {0, 1}; round 2 (--go-levels): also intermediate


class GoSim(W.Sim):
    """walk_real.Sim with a float go level (the original passes 1.0 / 0.0). Same as
    fly-taste-learning's CommandSim; identical to the original for go in {0, 1}."""
    go_level = 1.0

    def inputs(self):
        b, d = self.body, self.d
        roll, pitch = b.tilt(d)
        go = float(self.go_level) if self.go else 0.0
        return W.features(self.t, go, b.turn, roll, pitch), b.proprio(d), go


def schedule(rng, ms=EP_MS):
    """[(t_start, go, turn)]: 0.6-1.0 s segments."""
    out, t = [], 0.0
    while t < ms:
        r = rng.random()
        turn = 0.0 if r < 0.2 else float(rng.choice([-1.0, 1.0])) if r < 0.35 else float(rng.uniform(-1, 1))
        go = 0.0 if rng.random() < 0.1 else float(rng.choice(GO_LEVELS))
        out.append((t, go, turn))
        t += rng.uniform(600, 1000)
    return out


def episode(sim, sched, ms=EP_MS, heading=0.0):
    sim.reset(heading_deg0=heading)
    sim.targets = []
    b, d = sim.body, sim.d
    k, rows, kin = 0, [], []
    seg_log = {i: [] for i in range(len(sched))}
    while sim.t < ms:
        while k < len(sched) and sim.t >= sched[k][0]:
            sim.go, sim.turn_override = sched[k][1] > 0, sched[k][2]
            sim.go_level = sched[k][1]
            k += 1
        sim.step()
        if int(sim.t) % 10 == 0:
            if sim.t - sched[k - 1][0] >= SETTLE_MS:
                seg_log[k - 1].append((sim.t, *sim.pos(), b.heading(d)))
            if sim.t > 200:
                kin.append(b.kin_err(d))
        if sim.fallen():
            break
    fell = sim.fallen()
    for i, (t0, go, turn) in enumerate(sched):
        L = np.array(seg_log[i])
        if len(L) < 10:
            continue
        dt = (L[-1, 0] - L[0, 0]) / 1000
        h = np.unwrap(L[:, 3])
        yaw = (h[-1] - h[0]) / dt
        dx, dy = L[-1, 1] - L[0, 1], L[-1, 2] - L[0, 2]
        hm = np.angle(np.mean(np.exp(1j * h)))
        fwd = (dx * np.cos(hm) + dy * np.sin(hm)) / dt
        rows.append(dict(go=go, turn=turn, yaw=yaw, target_yaw=-turn * YAW_REF * go, fwd=fwd))
    return rows, fell, sim.t, float(np.mean(kin)) if kin else 0.0


def cost(rows, fell, t, kin, ms=EP_MS):
    c = 0.0
    if rows:
        yaw_err = np.mean([((r["yaw"] - r["target_yaw"]) / YAW_REF) ** 2 for r in rows])
        spd = np.mean([max(0.0, r["go"] * V_MIN - r["fwd"]) / V_MIN if r["go"] else abs(r["fwd"]) / V_MIN
                       for r in rows])
        c += yaw_err + spd
    else:
        c += 3.0
    return c + FALL_PENALTY * (1.0 - t / ms) * fell + 2.0 * fell + K_REALISM * kin


def _init(theta0):
    _W["c"] = W.Circuit()
    _W["sim"] = GoSim(theta0, _W["c"], seed=os.getpid())


def _fitness(args_):
    theta, scheds = args_
    sim, c = _W["sim"], _W["c"]
    sim.brain = W.Brain(c, theta)
    tot = 0.0
    for sc, hd in scheds:
        tot += cost(*episode(sim, sc, heading=hd))
    return tot / len(scheds)


def val_schedules(n=8, seed=999):
    rng = np.random.default_rng(seed)
    out = [([(0.0, 1.0, tv)], float(rng.uniform(-180, 180))) for tv in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    out += [([(0.0, g, 0.0)], float(rng.uniform(-180, 180))) for g in GO_LEVELS if g < 1.0]
    out += [(schedule(rng), float(rng.uniform(-180, 180))) for _ in range(n - 5)]
    return out


def train(a):
    global GO_LEVELS
    if a.go_levels:
        GO_LEVELS = tuple(float(g) for g in a.go_levels.split(","))   # set before the pool forks
    import cma
    from multiprocessing import get_context
    theta0 = np.load(a.start)["theta"]
    out = W.RUNS / a.out
    out.mkdir(parents=True, exist_ok=True)
    es = cma.CMAEvolutionStrategy(theta0, a.sigma, {"popsize": a.pop, "seed": a.seed + 1, "verbose": -9,
                                                    "CMA_diagonal": True})
    VAL = val_schedules()
    rng = np.random.default_rng(a.seed)
    with get_context("fork").Pool(a.workers, _init, (theta0,)) as pool, open(out / "log.csv", "w", newline="") as fh:
        f0 = float(np.mean(pool.map(_fitness, [(theta0, [v]) for v in VAL])))
        best = (f0, theta0)
        print(f"CMA-ES (command following) on {len(theta0)} parameters, population {a.pop}, start {a.start}: "
              f"validation cost {f0:.3f}", flush=True)
        log = csv.writer(fh)
        log.writerow(["gen", "best", "mean", "best_so_far", "seconds"])
        t0 = time.time()
        for gen in range(a.gens):
            sc = [(schedule(rng), float(rng.uniform(-180, 180))) for _ in range(a.episodes)]
            X = es.ask()
            F = pool.map(_fitness, [(x, sc) for x in X])
            es.tell(X, F)
            i = int(np.argmin(F))
            if gen % a.check_every == 0 or gen == a.gens - 1:
                fv = float(np.mean(pool.map(_fitness, [(X[i], [v]) for v in VAL])))
                if fv < best[0]:
                    best = (fv, np.array(X[i]))
                    np.savez(out / "policy.npz", theta=best[1], fitness=best[0], stage="cmd", gen=gen)
            log.writerow([gen, min(F), float(np.mean(F)), best[0], round(time.time() - t0)])
            fh.flush()
            print(f"gen {gen:3d}  best {min(F):.3f}  mean {np.mean(F):.3f}  (kept, validation {best[0]:.3f})  "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"-> {out}/policy.npz")


def _eval_job(args_):
    f, turn, seed = args_
    sim = GoSim(np.load(f)["theta"], seed=seed)
    rng = np.random.default_rng(seed)
    rows, fell, t, kin = episode(sim, [(0.0, 1.0, turn)], 4000, heading=float(rng.uniform(-180, 180)))
    r = rows[0] if rows else dict(yaw=np.nan, fwd=np.nan)
    return dict(turn=turn, seed=seed, yaw_deg_s=np.degrees(r["yaw"]), fwd=r["fwd"], fell=fell, kin=kin)


def evaluate(a):
    from multiprocessing import Pool
    jobs = [(a.file, tv, s) for tv in np.round(np.arange(-1, 1.01, 0.25), 2) for s in range(a.n)]
    with Pool(a.workers) as p:
        df = pd.DataFrame(p.map(_eval_job, jobs))
    print(df.groupby("turn").agg(yaw=("yaw_deg_s", "mean"), yaw_sd=("yaw_deg_s", "std"), fwd=("fwd", "mean"),
                                 falls=("fell", "sum"), kin=("kin", "mean")).to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"falls {df.fell.sum()}/{len(df)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--start", default=str(W.RUNS / "cma_robust" / "policy.npz"))
    tr.add_argument("--gens", type=int, default=80)
    tr.add_argument("--pop", type=int, default=22)
    tr.add_argument("--sigma", type=float, default=0.02)
    tr.add_argument("--episodes", type=int, default=2)
    tr.add_argument("--check-every", type=int, default=1)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--workers", type=int, default=22)
    tr.add_argument("--out", default="cmd")
    tr.add_argument("--go-levels", default="", help="e.g. 0.5,0.75,1 (round 2)")
    ev = sub.add_parser("eval")
    ev.add_argument("file")
    ev.add_argument("--n", type=int, default=8)
    ev.add_argument("--workers", type=int, default=22)
    a = ap.parse_args()
    {"train": train, "eval": evaluate}[a.cmd](a)
