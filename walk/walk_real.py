#!/usr/bin/env python3
"""A trained connectome network walks the fly with a REAL fly's gait (flybody-style imitation).

Same as walk_trained.py (fixed-weight connectome circuit, learned input / output maps, option-5
muscles, imitation then CMA-ES), but the gait the network learns is no longer the hand-made
tripod: it is real Drosophila walking, from flybody's motion-capture dataset (real_gait.py fits
it as joint angle vs stride phase and yaw rate). flybody trains its fly to imitate the same data.

    TEACHER  tracks the real joint angles q_ref(phase, yaw) with muscle drive
             u = K (q_ref - q) + KD (dq_ref - dq) (+ posture), and grips with the claws in
             stance (co-contraction of tarsal levator + depressor). Stride 9.5 Hz.
    EXTRA    coxa abduction / twist and femur twist (18 joints) have no motor neurons in the
             connectome table; they follow the real fly's angles with flybody's position servos,
             from the same stride clock. The 24 coxa / femur / tibia / tarsus joints are
             driven only by motor neuron spikes (option 5).
    REALISM  every episode also measures the joint-angle error to the real fly (kin_err, rad),
             and CMA-ES is penalised for it.

    python real_gait.py                                 the real gait model -> real_gait.npz
    python walk_real.py teacher                         the real-gait teacher in this body
    python walk_real.py imitate --rounds 0              stage 1 -> runs_real/imitate/
    python walk_real.py train --gens 60                 stage 2 -> runs_real/cma/
    python walk_real.py eval runs_real/cma/policy.npz
    python walk_real_gui.py                             the GUI

The original description of walk_trained.py follows.

A trained connectome network walks the fly (as the fly_arm project trains the arm).

    BRAIN  the six-leg walking circuit (walk_circuit/, built by walk_circuit_build.py):
           81 descending (command) neurons -> 2,205 premotor interneurons -> 302 leg motor
           neurons, plus 67 leg proprioceptors. Graded rate model, weights FIXED from the
           connectome (W = sign x synapses / receiver's total input), as in fly_arm.
    IN     task  -> command neurons     : learned matrix E   (stride clock, turn, go, body tilt)
           legs  -> proprioceptors      : learned matrix P   (each leg's joint angles / velocities
                                                              go only to that leg's sensors)
    OUT    each motor neuron fires at R_MAX x clip(gain_pool x activity) Hz; every spike is a
           twitch of its muscle and torque = Tmax x (extensor - flexor activation)
           (option 5, as walk_torque.py / walk_connectome.py)
    LEARN  1. imitation: E, P, gains, g are fitted (gradient, BPTT through the network) so the
              motor pools reproduce the drive of the controller that already walks (tripod gait
              + stretch / posture reflexes, walk_connectome.py), on states the fly visits;
              DAgger rounds then let the network walk and the old controller label the states.
           2. CMA-ES on the same parameters in MuJoCo: walk to targets without falling (fly_arm).
           The connectome weights themselves never change.

Results (runs/; unseen targets 1.5-2 cm away, 6 s, 12 runs unless noted):
    old hand-written controller (teacher)      2/6 reached, 0 falls
    network, imitation only                    0/6 reached, 2 falls (copies 62 % of the drive, R^2)
    network, + CMA-ES round 1 (runs/cma)       8/12 reached, 2 falls
    network, + round 2 (runs/cma_robust)       11/12 reached, 0 falls, typically in 1.5-3 s
    straight 3 cm runs (4 headings x 2 seeds)  8/8 reached in 3.7 s, max 3.7 mm off the line,
                                               path straightness 98 % (per stride)
    with Poisson motor spikes                  falls (trained on regular spikes; a CMA-ES round
                                               with Poisson spikes converged to standing still)
Design notes: units with fly_arm's 10 ms time constant are too slow for the walking
reflexes (see NEURON_TAU_MS); DAgger rounds made imitation worse (post-fall states), so
CMA-ES starts from the round-0 imitation fit.

    python walk_trained.py teacher                      the old controller in this body (sanity check)
    python walk_circuit_build.py                        the circuit -> walk_circuit/
    python walk_trained.py imitate --rounds 0           stage 1 -> runs/imitate/policy.npz
    python walk_trained.py train --gens 80              stage 2 -> runs/cma/policy.npz
    python walk_trained.py train --start runs/cma/policy.npz --gens 50 --targets 4 --ms 4000 \
           --sigma 0.02 --out cma_robust                 round 2 -> runs/cma_robust/policy.npz
    python walk_trained.py eval runs/cma_robust/policy.npz   headless test walks
    python walk_trained_gui.py                          the GUI
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import fly_body as WT  # noqa: E402  (the option-5 body: build, FLOOR_Z)
from fly_body import ADH_REF, LEGS, TAU_ADH, TRIPOD_A, heading_deg, probe_signs  # noqa: E402
from fly_body import build_model as build_single_fly  # noqa: E402
from real_gait import LEGS6 as GAIT_LEGS, RealGait  # noqa: E402

CIRCUIT = HERE / "walk_circuit"
RUNS = HERE / "runs_real"
PREFIX = "F_"
FLOOR_Z = WT.FLOOR_Z
STEP_MS = 1.0

# --- body (option 5) ---------------------------------------------------------------
R_MAX = 300.0             # Hz per motor neuron at full drive
TAU_RISE, TAU_DECAY = 3.0, 20.0
MUSCLE_SCALE = 1.0        # x the torque the servo makes at full drive
# --- the controller the network imitates (walk_connectome.Body.drive, re-tuned) -----
# walk_gait.py's tripod x 2 (stroke), kd 0.03: walks ~1.2-1.4 cm / 3 s (vs 0.2-0.3), never
# fell with drive noise 0.1 (3 seeds), reaches 3/4 targets 1.5-2 cm away in 8 s. Weaker
# reflexes (kp 1) walk faster but tip over at noise 0.07, i.e. under imitation error.
TEACHER = dict(kp=3.0, kd=0.03, k_tilt=2.0)
GAIT = RealGait()
PERIOD_MS = GAIT.period_ms  # real stride period (9.5 Hz -> ~105 ms)
# real-gait tracking (sweep: K 2-8, KD 0.02-0.05, grip 0/0.5 all walk; K 4, KD 0.05, grip 0.5:
# 1.58 cm/s (real flies 1.73), tilt 4 deg, no falls with drive noise 0.1)
K_TRACK, KD_TRACK, K_GRIP = 4.0, 0.05, 0.5
K_PASSIVE, GRIP_NO_LEVATOR = 2.0, 0.3   # see Body.__init__; sweep: K_PASSIVE 1-4 x grip 0.1-0.3 all walk
# 1.4-1.5 cm/s; without the spring the mid/hind tarsi fold to their limit (joint error 0.28 -> 0.07 rad)
YAW_MAX = 6.0             # rad/s of the real-gait turning model at |turn| = 1 (sweep 2-8: 26-160 deg/s;
                          # 6 -> ~115 deg/s, radius 0.55 cm, no falls; beyond most of the data)
V_TURN_DROP = 1.6         # cm/s slower (real gait's speed term) at |turn| = 1
EXTRA_JOINTS = ["coxa_abduct", "coxa_twist", "femur_twist"]
SWING = {"coxa": 1.0, "femur": -1.0, "tibia": -1.0, "tarsus": 1.0}
STANCE = {"coxa": -1.0, "femur": 0.6, "tibia": 0.4, "tarsus": -0.6}
K_HEAD, STEER_SIGN, TURN_MAX = 1.5, -1.0, 1.0   # turn > 0 -> right; |turn| <= 1 (tuned on targets)
ARRIVE_CM = 0.4
# --- network ------------------------------------------------------------------------
QD_SCALE = 20.0           # rad/s -> proprioceptor input units
N_PROP = 9                # per leg: 4 joint angles, 4 velocities, 1 bias
X_MIN, X_MAX = -1.0, 5.0  # activity clip, as fly_arm
# Time constant of the graded (population-rate) units. fly_arm used 10 ms (slow reaching);
# walking needs faster loops: with the teacher's reflex + posture feedback passed through
# three such stages it still walks at 2 ms (3/4 targets, 0 falls) but falls at 3 ms (4/4)
# and at 10 ms (4/4). Population rates of spiking neurons can follow inputs this fast.
NEURON_TAU_MS = 2.0


def leg_commands(t_ms, go, turn):
    """Tripod gait (walk_gait.leg_commands with this module's amplitudes / period):
    {(leg, joint): u}. Turning scales the coxa stroke by (1 -/+ turn) on the left / right."""
    if not go:
        return {(leg, j): 0.0 for leg in LEGS for j in SWING}
    first_half = (t_ms / PERIOD_MS) % 1.0 < 0.5
    out = {}
    for leg in LEGS:
        target = STANCE if first_half == (leg in TRIPOD_A) else SWING
        side = 1.0 - turn if leg.endswith("left") else 1.0 + turn
        for j, u in target.items():
            out[(leg, j)] = float(np.clip(u * side, -1, 1)) if j == "coxa" else u
    return out


# =============================================================================
# CIRCUIT + POLICY PARAMETERS
# =============================================================================
class Circuit:
    def __init__(self, folder=CIRCUIT):
        self.nodes = pd.read_csv(folder / "neurons.csv")
        z = np.load(folder / "weights.npz")
        self.n = len(self.nodes)
        self.pre, self.post, self.w = z["pre"], z["post"], z["w"].astype(np.float32)
        self.W = sparse.csr_matrix((self.w, (self.post, self.pre)), shape=(self.n, self.n))
        self.alpha = np.full(self.n, STEP_MS / NEURON_TAU_MS, np.float32)
        role = self.nodes.role.to_numpy()
        self.cmd = np.flatnonzero(role == "command")
        self.sens = np.flatnonzero(role == "sensory")
        self.sens_leg = np.array([LEGS.index(l) for l in self.nodes.leg[self.sens]])
        mn = self.nodes[role == "motor"]
        self.mn = mn.index.to_numpy()
        self.motor = pd.DataFrame({"leg": mn.leg.to_numpy(), "joint": mn.joint.to_numpy(),
                                   "direction": mn.direction.to_numpy().astype(int)})
        self.pools = sorted(set(zip(self.motor.leg, self.motor.joint, self.motor.direction)))
        self.mn_pool = np.array([self.pools.index(k) for k in zip(self.motor.leg, self.motor.joint,
                                                                   self.motor.direction)])
        self.keys = sorted(set(zip(self.motor.leg, self.motor.joint)))     # 24 (leg, joint)

    def pool_mean(self):
        """(n_pools, n_mn) matrix: pool average of motor neuron values."""
        M = np.zeros((len(self.pools), len(self.mn)), np.float32)
        M[self.mn_pool, np.arange(len(self.mn))] = 1
        return M / M.sum(1, keepdims=True)


def stride_phase(t_ms):
    """Stride phase in [-pi, pi) of the stride clock (the real gait's phase)."""
    return (2 * np.pi * t_ms / PERIOD_MS + np.pi) % (2 * np.pi) - np.pi


def features(t_ms, go, turn, roll, pitch):
    """Task vector for the command neurons: stride clock (1st + 3rd harmonic), its
    modulation by the turn command, go, and the body tilt (as from the halteres / neck)."""
    ph = stride_phase(t_ms)
    h = np.array([np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph), np.sin(3 * ph), np.cos(3 * ph)])
    return np.r_[go * h, go * turn * h, go * abs(turn) * h, go, go * turn, roll, pitch, 1.0].astype(np.float32)


N_TASK = len(features(0, 1, 0, 0, 0))


def sizes(c):
    return {"E": (len(c.cmd), N_TASK), "P": (len(c.sens), N_PROP), "log_gain": (len(c.pools),), "g": (1,)}


def dim(c):
    return sum(int(np.prod(s)) for s in sizes(c).values())


def unpack(theta, c):
    out, i = {}, 0
    for k, shp in sizes(c).items():
        n = int(np.prod(shp))
        out[k] = np.asarray(theta[i:i + n], np.float32).reshape(shp)
        i += n
    return out


def pack(p, c):
    return np.concatenate([np.asarray(p[k], np.float64).ravel() for k in sizes(c)])


def init_theta(c, seed=0):
    rng = np.random.default_rng(seed)
    p = {"E": rng.normal(0, 0.3, sizes(c)["E"]), "P": np.zeros(sizes(c)["P"]),
         "log_gain": np.full(len(c.pools), np.log(20.0)), "g": np.zeros(1)}
    return pack(p, c)


def recurrent_gain(g):
    return float(np.clip(1.0 + g, 0.0, 2.0))


# =============================================================================
# BRAIN (numpy, 1 ms steps)
# =============================================================================
class Brain:
    def __init__(self, circuit, theta):
        self.c = circuit
        self.set_theta(theta)
        self.reset()

    def set_theta(self, theta):
        p = unpack(theta, self.c)
        self.E, self.P = p["E"], p["P"]
        self.gain = np.exp(p["log_gain"])[self.c.mn_pool].astype(np.float32)
        self.Wg = (self.c.W * recurrent_gain(p["g"][0])).astype(np.float32).tocsr()

    def reset(self):
        self.x = np.zeros(self.c.n, np.float32)
        self.inp = np.zeros(self.c.n, np.float32)

    def step(self, task, prop):
        """task: (N_TASK,), prop: (6, N_PROP). Returns motor neuron drive in [0, 1]."""
        c = self.c
        self.inp[c.cmd] = self.E @ task
        self.inp[c.sens] = np.einsum("sk,sk->s", self.P, prop[c.sens_leg])
        r = np.maximum(self.x, 0)
        self.x += c.alpha * (self.Wg @ r + self.inp - self.x)
        np.clip(self.x, X_MIN, X_MAX, out=self.x)
        return np.clip(self.gain * np.maximum(self.x[c.mn], 0), 0.0, 1.0)


# =============================================================================
# BODY: option-5 muscles, proprioception, the old controller (teacher), steering
# =============================================================================
class Body:
    def __init__(self, m, circuit, tmax, signs, poisson=False, seed=0):
        mot = circuit.motor
        self.c = circuit
        self.keys = circuit.keys
        self.sign = np.array([signs[k] for k in self.keys], float)
        self.aid = np.array([m.actuator(f"{PREFIX}{j}_{l}").id for l, j in self.keys])
        self.qadr = np.array([m.joint(f"{PREFIX}{j}_{l}").qposadr[0] for l, j in self.keys])
        self.vadr = np.array([m.joint(f"{PREFIX}{j}_{l}").dofadr[0] for l, j in self.keys])
        self.lo, self.hi = m.actuator_ctrlrange[self.aid].T.copy()
        self.tmax = np.array([tmax[k] for k in self.keys]) * MUSCLE_SCALE
        self.key_of_mn = np.array([self.keys.index(k) for k in zip(mot.leg, mot.joint)])
        self.dir_col = (mot.direction.to_numpy() < 0).astype(int)
        self.n = np.zeros((len(self.keys), 2))
        np.add.at(self.n, (self.key_of_mn, self.dir_col), 1)
        self.n = np.maximum(self.n, 1)
        self.leg_of_key = np.array([LEGS.index(l) for l, _ in self.keys])
        self.leg_y = np.array([1.0 if l.endswith("left") else -1.0 for l, _ in self.keys])
        self.leg_x = np.array([{"T1": 1.0, "T2": 0.0, "T3": -1.0}[l[:2]] for l, _ in self.keys])
        self.support = np.array([j in ("femur", "tibia") for _, j in self.keys], float)
        self.claw_mn = (mot.joint.to_numpy() == "tarsus") & (mot.direction.to_numpy() < 0)
        self.claw_leg = np.array([LEGS.index(l) for l in mot.leg])
        self.n_claw = np.maximum(np.bincount(self.claw_leg[self.claw_mn], minlength=6), 1)
        self.adh = np.array([m.actuator(f"{PREFIX}adhere_claw_{l}").id for l in LEGS])
        self.thorax = m.body(PREFIX + "thorax").id
        # real gait: reference index of each driven joint, the extra (servo) joints, stance legs
        self.ref_key = np.array([GAIT.idx[f"{j}_{l}"] for l, j in self.keys])
        ex = [(l, j) for l in LEGS for j in EXTRA_JOINTS]
        self.ex_names = [f"{j}_{l}" for l, j in ex]
        self.ex_aid = np.array([m.actuator(f"{PREFIX}{j}_{l}").id for l, j in ex])
        self.ex_qadr = np.array([m.joint(f"{PREFIX}{j}_{l}").qposadr[0] for l, j in ex])
        self.ref_ex = np.array([GAIT.idx[n] for n in self.ex_names])
        gl = list(GAIT_LEGS)
        self.stance_leg = np.array([gl.index(l) for l in LEGS])
        # the connectome table has no tarsal levator motor neurons on the middle and hind legs:
        # there, a passive elastic return (K_PASSIVE x Tmax per rad) opposes the depressors, and
        # the stance grip drive is GRIP_NO_LEVATOR instead of a levator + depressor co-contraction
        has_lev = {l for l, j, dd in circuit.pools if j == "tarsus" and dd == 1}
        self.k_pass = np.array([K_PASSIVE * self.tmax[k] if (j == "tarsus" and l not in has_lev) else 0.0
                                for k, (l, j) in enumerate(self.keys)])
        self.pool_tarsus = np.array([(j == "tarsus") * (K_GRIP if l in has_lev else GRIP_NO_LEVATOR)
                                     for l, j, _ in circuit.pools], float)
        self.pool_leg = np.array([LEGS.index(l) for l, _, _ in circuit.pools])
        self.q_ref = GAIT.a[0].copy()          # current reference pose (all leg joints)
        # pool p = (leg, joint, d): which key, which direction
        self.pool_key = np.array([self.keys.index((l, j)) for l, j, _ in circuit.pools])
        self.pool_dir = np.array([d for _, _, d in circuit.pools], float)
        self.poisson = poisson
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.s = np.zeros((len(self.keys), 2))
        self.F = np.zeros((len(self.keys), 2))
        self.fast = np.zeros(6)
        self.ph = self.rng.uniform(0, 1, len(self.key_of_mn))
        self.turn = 0.0
        self.q_ref, self.dq_ref, self.phi = GAIT.a[0].copy(), np.zeros(len(GAIT.names)), None

    # --- sensing -------------------------------------------------------------------
    def tilt(self, d):
        up = d.xmat[self.thorax].reshape(3, 3).T @ np.array([0, 0, 1.0])
        return float(np.arctan2(-up[1], up[2])), float(np.arctan2(-up[0], up[2]))

    def heading(self, d):
        return np.radians(heading_deg(d, self.thorax))

    def proprio(self, d):
        """(6, N_PROP): per leg the 4 joint angles and velocities (anatomical sign) + bias."""
        q = self.sign * d.qpos[self.qadr]
        qd = self.sign * d.qvel[self.vadr] / QD_SCALE
        out = np.zeros((6, N_PROP), np.float32)
        for k, leg in enumerate(self.leg_of_key):
            j = k % 4
            out[leg, j], out[leg, 4 + j] = q[k], qd[k]
        out[:, 8] = 1.0
        return out

    def steer(self, d, target):
        if target is None:
            self.turn = 0.0
            return 0.0
        p = d.xpos[self.thorax]
        want = np.arctan2(target[1] - p[1], target[0] - p[0])
        err = (want - self.heading(d) + np.pi) % (2 * np.pi) - np.pi
        self.turn = float(np.clip(STEER_SIGN * K_HEAD * err, -TURN_MAX, TURN_MAX))
        return self.turn

    # --- the controller the network imitates ------------------------------------------
    def reference(self, t_ms, go, turn):
        """Real fly joint angles (and rates, rad/s) at this stride phase and turn; standing
        still (go = 0) is the real gait's mean pose."""
        if not go:
            return GAIT.a[0], np.zeros(len(GAIT.names)), None
        phi = stride_phase(t_ms)
        yaw = -turn * YAW_MAX                     # turn > 0 -> right -> negative yaw rate
        w = 2 * np.pi / PERIOD_MS * 1000.0
        dv = -abs(turn) * V_TURN_DROP             # real flies slow down in sharp turns
        return GAIT.q(phi, yaw, dv), GAIT.dq(phi, yaw, dv) * w, phi

    def set_extra(self, d, t_ms, go, turn):
        """Joints without motor neurons follow the real fly with their position servos."""
        self.q_ref, self.dq_ref, self.phi = self.reference(t_ms, go, turn)
        d.ctrl[self.ex_aid] = self.q_ref[self.ref_ex]

    def teacher_pools(self, d, t_ms, go, turn):
        """Pool drive in [0, 1] (44,): track the real fly's joint angles, keep the body level,
        grip with the claws in stance."""
        qr, dqr, phi = self.q_ref, self.dq_ref, self.phi
        q, qd = d.qpos[self.qadr], d.qvel[self.vadr]
        u = self.sign * (K_TRACK * (qr[self.ref_key] - q) + KD_TRACK * (dqr[self.ref_key] - qd))
        roll, pitch = self.tilt(d)
        u = u + self.support * TEACHER["k_tilt"] * (roll * self.leg_y + pitch * self.leg_x)
        u = np.clip(u, -1, 1)
        p = np.maximum(self.pool_dir * u[self.pool_key], 0.0)
        stance = GAIT.in_stance(phi)[self.stance_leg] if phi is not None else np.ones(6)
        p = p + self.pool_tarsus * stance[self.pool_leg]
        return np.clip(p, 0.0, 1.0)

    def kin_err(self, d):
        """RMS joint-angle difference to the real fly (all 42 walking joints), rad."""
        e1 = d.qpos[self.qadr] - self.q_ref[self.ref_key]
        e2 = d.qpos[self.ex_qadr] - self.q_ref[self.ref_ex]
        return float(np.sqrt((np.sum(e1 ** 2) + np.sum(e2 ** 2)) / (len(e1) + len(e2))))

    # --- muscles -------------------------------------------------------------------------
    def apply(self, d, drive):
        """drive: (n_mn,) in [0, 1]. Motor neurons spike at R_MAX x drive; each spike is a
        twitch; torque = sign x Tmax x (a+ - a-); claw grip from the claw motor neurons."""
        rate = drive * R_MAX * STEP_MS / 1000.0
        if self.poisson:
            fired = self.rng.random(len(rate)) < rate
        else:
            self.ph += rate
            fired = self.ph >= 1.0
            self.ph[fired] -= 1.0
        counts = np.zeros_like(self.n)
        np.add.at(counts, (self.key_of_mn[fired], self.dir_col[fired]), 1)
        self.s = self.s * np.exp(-STEP_MS / TAU_DECAY) + counts / self.n
        self.F += STEP_MS / TAU_RISE * (self.s - self.F)
        a = self.F / (R_MAX * TAU_DECAY / 1000.0)
        d.ctrl[self.aid] = np.clip(self.sign * self.tmax * (a[:, 0] - a[:, 1]) - self.k_pass * d.qpos[self.qadr],
                                   self.lo, self.hi)
        self.fast *= np.exp(-STEP_MS / TAU_ADH)
        np.add.at(self.fast, self.claw_leg[fired & self.claw_mn], 1.0)
        grip = self.fast / (self.n_claw * TAU_ADH / 1000.0) / R_MAX
        d.ctrl[self.adh] = np.clip(grip / abs(ADH_REF) * 0.6, 0.0, 1.0)
        return fired


# =============================================================================
# SIM: fly + controller (teacher or trained network), 1 ms steps
# =============================================================================
class Sim:
    def __init__(self, theta=None, circuit=None, poisson=False, seed=0, verbose=False):
        t0 = time.time()
        self.c = circuit or Circuit()
        m1, d1 = build_single_fly()
        signs = probe_signs(m1, d1, LEGS)
        self.m, self.d, tmax = WT.build(1.0, flies=[(PREFIX, 0.0, "fly", "torque", [1, 1, 1, 1])])
        self.body = Body(self.m, self.c, tmax, signs, poisson, seed)
        self.brain = Brain(self.c, theta) if theta is not None else None
        self.nsub = int(round(STEP_MS / 1000 / self.m.opt.timestep))
        self.targets, self.loop_targets = [], False
        self.go = True
        self.turn_override = None         # fixed turn command (data collection)
        self.teacher_noise = 0.0
        self.rng = np.random.default_rng(seed)
        self.reset()
        if verbose:
            print(f"[sim] circuit {self.c.n} neurons, {self.c.W.nnz} connections; "
                  f"ready in {time.time() - t0:.1f} s", flush=True)

    def reset(self, heading_deg0=0.0, xy=(0.0, 0.0)):
        mujoco.mj_resetData(self.m, self.d)
        adr = self.m.joint(PREFIX + "free").qposadr[0]
        self.d.qpos[adr:adr + 2] = xy
        q = np.zeros(4)
        mujoco.mju_axisAngle2Quat(q, np.array([0, 0, 1.0]), np.radians(heading_deg0))
        self.d.qpos[adr + 3:adr + 7] = q
        wings = [i for i in range(self.m.njnt) if "wing" in self.m.joint(i).name]
        for i in wings:                       # start with the wings folded (their spring rest pose)
            a = self.m.jnt_qposadr[i]
            self.d.qpos[a] = self.m.qpos_spring[a]
        mujoco.mj_forward(self.m, self.d)
        self.body.reset()
        if self.brain is not None:
            self.brain.reset()
        self.ou = np.zeros(len(self.c.pools))
        self.t = 0.0
        self.trail = [self.pos().copy()]
        self.arrivals = []
        self.fired = np.zeros(len(self.c.mn), bool)
        self.mn_hist = []

    def pos(self):
        return self.d.xpos[self.body.thorax][:2].copy()

    def height(self):
        return self.d.xpos[self.body.thorax][2] - FLOOR_Z

    def update_turn(self):
        b, d = self.body, self.d
        if self.turn_override is not None:
            b.turn = self.turn_override
            return
        tgt = self.targets[0] if self.targets else None
        b.steer(d, tgt)
        if tgt is not None and np.hypot(*(self.pos() - tgt)) < ARRIVE_CM:
            self.arrivals.append((tuple(tgt), self.t / 1000))
            done = self.targets.pop(0)
            if self.loop_targets:
                self.targets.append(done)

    def inputs(self):
        """What the network sees this ms: task vector and proprioception."""
        b, d = self.body, self.d
        roll, pitch = b.tilt(d)
        go = 1.0 if (self.go and (self.targets or self.turn_override is not None)) else 0.0
        return features(self.t, go, b.turn, roll, pitch), b.proprio(d), go

    def step(self, record=None):
        b, d = self.body, self.d
        k = int(round(self.t / STEP_MS))
        if k % int(PERIOD_MS / STEP_MS) == 0:
            self.update_turn()
        task, prop, go = self.inputs()
        b.set_extra(d, self.t, go, b.turn)
        if record is not None or self.brain is None:
            target = b.teacher_pools(d, self.t, go, b.turn)
        if self.brain is None:                       # the old controller drives the muscles
            if self.teacher_noise > 0:
                self.ou += -self.ou / 20.0 + self.teacher_noise * np.sqrt(2 / 20.0) * self.rng.standard_normal(len(self.ou))
                target_act = np.clip(target + self.ou, 0, 1)
            else:
                target_act = target
            drive = target_act[self.c.mn_pool]
        else:
            drive = self.brain.step(task, prop)
        if record is not None:
            record.append((task, prop, target))
        self.fired = b.apply(d, drive)
        mujoco.mj_step(self.m, d, nstep=self.nsub)
        self.t += STEP_MS
        if k % 20 == 0:
            self.trail.append(self.pos().copy())

    def fallen(self):
        r, p = self.body.tilt(self.d)
        return self.height() < 0.07 or abs(r) > 1.0 or abs(p) > 1.0

    def state(self):
        d, b = self.d, self.body
        r, p = b.tilt(d)
        x, y = self.pos()
        return dict(t=self.t, x=x, y=y, z=self.height(), heading=heading_deg(d, b.thorax),
                    roll=np.degrees(r), pitch=np.degrees(p), turn=b.turn)


# =============================================================================
# episodes, data collection, evaluation
# =============================================================================
def course_episode(sim, target, ms, record=None, stop_on_fall=True):
    """Walk from the origin toward `target` for `ms`. Returns progress (cm) and a log."""
    sim.reset()
    sim.targets = [np.asarray(target, float)]
    start = np.hypot(*target)
    fell, tilt, kin = False, [], []
    while sim.t < ms:
        sim.step(record)
        if int(sim.t) % 10 == 0:
            r, p = sim.body.tilt(sim.d)
            tilt.append(r * r + p * p)
            if sim.t > 200:
                kin.append(sim.body.kin_err(sim.d))
        if sim.fallen():
            fell = True
            if stop_on_fall:
                break
        if not sim.targets:
            break
    dist = np.hypot(*(sim.pos() - target)) if sim.targets else 0.0
    return dict(progress=start - dist, reached=not sim.targets, fell=fell, t=sim.t,
                tilt_deg=float(np.degrees(np.sqrt(np.mean(tilt)))) if tilt else 0.0,
                kin_err=float(np.mean(kin)) if kin else 0.0, speed=(start - dist) / max(sim.t / 1000, 1e-3))


def random_target(rng, r=(1.0, 2.0)):
    a = rng.uniform(-np.pi, np.pi)
    return np.array([np.cos(a), np.sin(a)]) * rng.uniform(*r)


def collect(sim, n_ep, ms, seed, noise=0.0):
    """Roll out (teacher if sim.brain is None, else the network) and record, per ms, the
    network inputs and the teacher's pool drive for that state (DAgger labels)."""
    rng = np.random.default_rng(seed)
    T, Pp, Y, falls = [], [], [], 0
    sim.teacher_noise = noise
    for ep in range(n_ep):
        rec = []
        mode = rng.random()
        if mode < 0.6:          # walk toward a random target (steering)
            sim.turn_override = None
            course_episode(sim, random_target(rng), ms, rec)
        else:                   # fixed turn command, or standing still
            sim.reset(heading_deg0=rng.uniform(-180, 180))
            sim.turn_override = float(rng.uniform(-TURN_MAX, TURN_MAX)) if mode < 0.9 else 0.0
            sim.go = mode < 0.9
            while sim.t < ms and not sim.fallen():
                sim.step(rec)
            sim.go, sim.turn_override = True, None
        falls += sim.fallen()
        if len(rec) > 200:
            T.append(np.array([r[0] for r in rec])); Pp.append(np.array([r[1] for r in rec]))
            Y.append(np.array([r[2] for r in rec], np.float32))
    sim.teacher_noise = 0.0
    return T, Pp, Y, falls


# =============================================================================
# STAGE 1: imitation (torch, BPTT through the fixed-weight connectome network)
# =============================================================================
def imitate_fit(c, data, theta, iters, lr=1e-2, chunk=300, burn=50, batch=96, device=None, log=None):
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    W = torch.sparse_coo_tensor(np.vstack([c.post, c.pre]), c.w, (c.n, c.n)).coalesce().to(dev)
    alpha = torch.tensor(c.alpha, device=dev)
    cmd, sens, mn = (torch.tensor(a, device=dev) for a in (c.cmd, c.sens, c.mn))
    sens_leg = torch.tensor(c.sens_leg, device=dev)
    pool = torch.tensor(c.mn_pool, device=dev)
    M = torch.tensor(c.pool_mean(), device=dev)
    p0 = unpack(theta, c)
    prm = {k: torch.tensor(v, device=dev, requires_grad=True) for k, v in p0.items()}
    opt = torch.optim.Adam(prm.values(), lr=lr)
    # chunks of `chunk` ms from all episodes
    Tk, Pk, Yk = data
    starts = [(e, s) for e in range(len(Tk)) for s in range(0, len(Tk[e]) - chunk, chunk // 3)]
    rng = np.random.default_rng(0)
    for it in range(iters):
        sel = [starts[i] for i in rng.choice(len(starts), min(batch, len(starts)), replace=False)]
        task = torch.tensor(np.stack([Tk[e][s:s + chunk] for e, s in sel], 1), device=dev)   # (L,B,N_TASK)
        prop = torch.tensor(np.stack([Pk[e][s:s + chunk] for e, s in sel], 1), device=dev)   # (L,B,6,9)
        y = torch.tensor(np.stack([Yk[e][s:s + chunk] for e, s in sel], 1), device=dev)      # (L,B,44)
        B = len(sel)
        g = torch.clamp(1.0 + prm["g"], 0.0, 2.0)
        gain = torch.exp(prm["log_gain"])[pool]
        x = torch.zeros(B, c.n, device=dev)
        loss = 0.0
        cmd_in = task @ prm["E"].T                                         # (L,B,ncmd)
        sens_in = (prop[:, :, sens_leg, :] * prm["P"]).sum(-1)              # (L,B,nsens)
        for k in range(chunk):
            inp = torch.zeros(B, c.n, device=dev)
            inp[:, cmd] = cmd_in[k]
            inp[:, sens] = sens_in[k]
            r = torch.relu(x)
            x = x + alpha * (g * torch.sparse.mm(W, r.T).T + inp - x)
            x = torch.clamp(x, X_MIN, X_MAX)
            if k >= burn:
                out = torch.clamp(gain * torch.relu(x[:, mn]), 0, 1) @ M.T
                loss = loss + ((out - y[k]) ** 2).mean()
        loss = loss / (chunk - burn)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 25 == 0 or it == iters - 1:
            msg = f"  iter {it:4d}  imitation loss {loss.item():.4f}"
            print(msg, flush=True)
            if log:
                log.write(msg + "\n"); log.flush()
    return pack({k: v.detach().cpu().numpy() for k, v in prm.items()}, c)


def evaluate(theta, c, n=4, ms=4000, seed=100, sim=None):
    sim = sim or Sim(theta, c)
    if theta is not None:
        sim.brain = Brain(c, theta)
    rng = np.random.default_rng(seed)
    res = [course_episode(sim, random_target(rng, (1.5, 2.0)), ms) for _ in range(n)]
    return pd.DataFrame(res)


def imitate(args):
    c = Circuit()
    out = RUNS / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")
    sim = Sim(None, c)
    print(f"circuit: {c.n} neurons ({len(c.cmd)} command, {len(c.sens)} sensory, {len(c.mn)} motor), "
          f"{c.W.nnz} connections; {dim(c)} learned parameters", flush=True)
    t0 = time.time()
    cache = out / "teacher_data.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        T, Pp, Y = list(z["T"]), list(z["P"]), list(z["Y"])
        print(f"teacher data from {cache}: {sum(map(len, T)) / 1000:.0f} s", flush=True)
    else:
        T, Pp, Y, falls = collect(sim, args.episodes, args.ms, seed=1, noise=0.1)
        np.savez(cache, T=np.array(T, dtype=object), P=np.array(Pp, dtype=object), Y=np.array(Y, dtype=object))
        print(f"teacher data: {sum(map(len, T)) / 1000:.0f} s, {falls} falls ({time.time() - t0:.0f} s)", flush=True)
    data = (T, Pp, Y)
    theta = init_theta(c, args.seed)
    for rnd in range(args.rounds + 1):
        theta = imitate_fit(c, data, theta, args.iters if rnd == 0 else args.iters // 2, lr=args.lr, log=log)
        np.savez(out / "policy.npz", theta=theta, stage="imitate", round=rnd)
        ev = evaluate(theta, c, n=4, ms=3000)
        msg = (f"round {rnd}: network walks: progress {ev.progress.mean():.2f} cm, reached {ev.reached.sum()}/4, "
               f"falls {ev.fell.sum()}/4, tilt {ev.tilt_deg.mean():.1f} deg ({time.time() - t0:.0f} s)")
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
        if rnd == args.rounds:
            break
        # DAgger: the network walks, the teacher labels the states it reaches
        sim.brain = Brain(c, theta)
        T2, P2, Y2, f2 = collect(sim, args.episodes // 2, args.ms, seed=10 + rnd)
        sim.brain = None
        T += T2; Pp += P2; Y += Y2
        print(f"  DAgger: +{sum(map(len, T2)) / 1000:.0f} s of network-driven states ({f2} falls)", flush=True)
    print(f"-> {out}/policy.npz")


# =============================================================================
# STAGE 2: CMA-ES in MuJoCo (as fly_arm)
# =============================================================================
TRAIN_TARGETS = [np.array([2.0 * np.cos(a), 2.0 * np.sin(a)]) for a in np.radians([0, 60, 120, 180, -60, -120])]
_W = {}
FALL_PENALTY = 6.0
K_REALISM = 4.0          # cost per rad of RMS joint-angle error to the real fly


def _init(theta0, poisson=False):
    _W["c"] = Circuit()
    _W["sim"] = Sim(theta0, _W["c"], poisson=poisson, seed=os.getpid())


def _fitness(args_):
    theta, targets, ms = args_
    sim, c = _W["sim"], _W["c"]
    sim.brain = Brain(c, theta)
    cost = 0.0
    for tgt in targets:
        r = course_episode(sim, tgt, ms)
        # progress toward the target, a large penalty for falling, a small one for tilt
        cost += -r["progress"] + FALL_PENALTY * (1.5 - 0.5 * r["t"] / ms) * r["fell"] + 0.02 * r["tilt_deg"] \
            - 1.0 * r["reached"] * (1 - r["t"] / ms) + K_REALISM * r["kin_err"]
    return cost / len(targets)


def train(args):
    import cma
    from multiprocessing import get_context
    global K_REALISM
    if args.realism is not None:
        K_REALISM = args.realism          # set before the worker pool forks
    c = Circuit()
    z = np.load(args.start)
    theta0 = z["theta"]
    out = RUNS / args.out
    out.mkdir(parents=True, exist_ok=True)
    es = cma.CMAEvolutionStrategy(theta0, args.sigma, {"popsize": args.pop, "seed": args.seed + 1,
                                                       "verbose": -9, "CMA_diagonal": True})
    print(f"CMA-ES on {len(theta0)} parameters (diagonal), population {args.pop}, start {args.start}"
          f"{', Poisson motor spikes' if args.poisson else ''}", flush=True)
    best = (_fitness_local(theta0, c, args), theta0)
    print(f"start fitness {best[0]:.3f}", flush=True)
    rng = np.random.default_rng(args.seed)
    with get_context("fork").Pool(args.workers, _init, (theta0, args.poisson)) as pool, open(out / "log.csv", "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["gen", "best", "mean", "best_so_far", "seconds"])
        t0 = time.time()
        for gen in range(args.gens):
            idx = rng.choice(len(TRAIN_TARGETS), args.targets, replace=False)
            tg = [TRAIN_TARGETS[i] for i in idx]
            X = es.ask()
            F = pool.map(_fitness, [(x, tg, args.ms) for x in X])
            es.tell(X, F)
            i = int(np.argmin(F))
            # re-check the generation's best on all training targets before keeping it
            if F[i] < best[0]:
                full = pool.map(_fitness, [(X[i], TRAIN_TARGETS[j:j + 1], args.ms) for j in range(len(TRAIN_TARGETS))])
                f_all = float(np.mean(full))
                if f_all < best[0]:
                    best = (f_all, np.array(X[i]))
                    np.savez(out / "policy.npz", theta=best[1], fitness=best[0], stage="cma", gen=gen)
            log.writerow([gen, min(F), float(np.mean(F)), best[0], round(time.time() - t0)])
            fh.flush()
            print(f"gen {gen:3d}  best {min(F):.3f}  mean {np.mean(F):.3f}  (kept {best[0]:.3f})  "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"-> {out}/policy.npz")


def _fitness_local(theta, c, args):
    _W["c"], _W["sim"] = c, Sim(theta, c, poisson=args.poisson)
    return float(np.mean([_fitness((theta, [t], args.ms)) for t in TRAIN_TARGETS]))


# =============================================================================
def teacher(args):
    c = Circuit()
    sim = Sim(None, c, verbose=True)
    t0 = time.time()
    ev = evaluate(None, c, n=args.n, ms=args.ms, sim=sim)
    wall = time.time() - t0
    print(ev.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"physics + controller: {ev.t.sum() / 1000 / wall:.2f} x real time")


def eval_cmd(args):
    c = Circuit()
    theta = np.load(args.file)["theta"]
    sim = Sim(theta, c, poisson=args.poisson)
    t0 = time.time()
    ev = evaluate(theta, c, n=args.n, ms=args.ms, sim=sim, seed=args.seed)
    wall = time.time() - t0
    print(ev.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"reached {ev.reached.sum()}/{len(ev)}, falls {ev.fell.sum()}, "
          f"{ev.t.sum() / 1000 / wall:.2f} x real time")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teacher")
    t.add_argument("--n", type=int, default=4)
    t.add_argument("--ms", type=float, default=4000)
    im = sub.add_parser("imitate")
    im.add_argument("--episodes", type=int, default=40)
    im.add_argument("--ms", type=float, default=3000)
    im.add_argument("--iters", type=int, default=1500)
    im.add_argument("--lr", type=float, default=1e-2)
    im.add_argument("--rounds", type=int, default=3)
    im.add_argument("--seed", type=int, default=0)
    im.add_argument("--out", default="imitate")
    tr = sub.add_parser("train")
    tr.add_argument("--start", default=str(RUNS / "imitate" / "policy.npz"))
    tr.add_argument("--gens", type=int, default=60)
    tr.add_argument("--pop", type=int, default=22)
    tr.add_argument("--sigma", type=float, default=0.03)
    tr.add_argument("--targets", type=int, default=2)
    tr.add_argument("--ms", type=float, default=3000)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    tr.add_argument("--out", default="cma")
    tr.add_argument("--poisson", action="store_true", help="train with Poisson (noisy) motor spikes")
    tr.add_argument("--realism", type=float, default=None, help="cost per rad of joint-angle error to the real fly")
    ev = sub.add_parser("eval")
    ev.add_argument("file")
    ev.add_argument("--n", type=int, default=6)
    ev.add_argument("--ms", type=float, default=5000)
    ev.add_argument("--seed", type=int, default=100)
    ev.add_argument("--poisson", action="store_true")
    a = ap.parse_args()
    {"teacher": teacher, "imitate": imitate, "train": train, "eval": eval_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
