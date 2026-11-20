#!/usr/bin/env python3
"""A gait model of real Drosophila walking, fitted to flybody's motion-capture dataset.

flybody (Vaxenburg et al. 2024) makes its fly walk by imitating motion-capture of real
walking flies, expressed as joint angles of the same MuJoCo fly model. This script turns that
data into a compact, phase-based gait model that the trained network can imitate:

  1. stride phase  : for each leg, the fore-aft position of the claw in the body frame is
                     band-passed (4-25 Hz) and its Hilbert phase taken; the six leg phases are
                     combined into one tripod phase phi (tripod B shifted by pi). Frames where
                     the legs are not coordinated (|mean vector| < 0.6) are dropped.
  2. joint angles  : every leg joint q_j is fitted as a Fourier series in phi (harmonics 0..3)
                     whose coefficients depend linearly on the yaw rate and forward speed:
                         q_j = sum_h [a_jh + b_jh * yaw + c_jh * (speed - v0)] * basis_h(phi)
  3. stride rate   : d(phi)/dt as a linear function of speed; stance = claw moving backwards.

    python real_gait.py                 fit -> real_gait.npz (+ a figure in runs/real_gait/)
    python real_gait.py --data FILE     another flybody walking .hdf5 (e.g. the full 6.4 GB one)
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from paths import MOCAP_FILE  # noqa: E402

DEFAULT_DATA = MOCAP_FILE
OUT = HERE / "real_gait.npz"
LEGS6 = ["T1_left", "T2_left", "T3_left", "T1_right", "T2_right", "T3_right"]
TRIPOD_A = {"T1_left", "T2_right", "T3_left"}
LEG_JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia", "tarsus"]
N_HARM = 3


def basis(phi):
    """(..., 2*N_HARM+1): 1, cos(h phi), sin(h phi)."""
    phi = np.asarray(phi, float)[..., None]
    h = np.arange(1, N_HARM + 1)
    return np.concatenate([np.ones_like(phi), np.cos(h * phi), np.sin(h * phi)], -1)


def _quat_yaw(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def fit(path=DEFAULT_DATA, out=OUT):
    import h5py
    from scipy.signal import butter, filtfilt, hilbert
    f = h5py.File(path, "r")
    dt = float(f["timestep_seconds"][()])
    jn = [s.decode() for s in f["id2name"]["joints"]]
    sites = [s.decode() for s in f["id2name"]["sites"]]
    site_of = [sites.index(f"claw_{leg}") for leg in LEGS6]
    names = [f"{j}_{leg}" for leg in LEGS6 for j in LEG_JOINTS]
    cols = [jn.index(n) for n in names]
    bb, ba = butter(2, [4, 25], btype="band", fs=1 / dt)
    PHI, Q, YAW, V, DPHI, STANCE = [], [], [], [], [], []
    for key in f["trajectories"]:
        g = f["trajectories"][key]
        q, rq, rv, r2s = g["qpos"][()], g["root_qpos"][()], g["root_qvel"][()], g["root2site"][()]
        n = len(q)
        if n < 120:
            continue
        cx = r2s[:, site_of, 0]                                  # claw fore-aft, body frame
        z = hilbert(filtfilt(bb, ba, cx - cx.mean(0), axis=0), axis=0)
        ph = np.angle(z)
        shift = np.array([0.0 if leg in TRIPOD_A else np.pi for leg in LEGS6])
        mv = np.exp(1j * (ph + shift)).mean(1)
        phi = np.angle(mv)
        yaw = _quat_yaw(rq[:, 3:7])
        heading = np.unwrap(yaw)
        yaw_rate = np.gradient(heading, dt)
        speed = rv[:, 0] * np.cos(heading) + rv[:, 1] * np.sin(heading)    # forward speed
        dphi = np.gradient(np.unwrap(phi), dt)
        swing_vel = np.gradient(cx, dt, axis=0)                  # claw moving forward = swing
        keep = (np.abs(mv) > 0.6) & (dphi > 0)
        keep[:30] = keep[-30:] = False
        PHI.append(phi[keep]); Q.append(q[keep][:, cols]); YAW.append(yaw_rate[keep])
        V.append(speed[keep]); DPHI.append(dphi[keep]); STANCE.append(swing_vel[keep] < 0)
    phi, Qm, yaw, v, dphi, stance = map(np.concatenate, (PHI, Q, YAW, V, DPHI, STANCE))
    v0 = float(np.median(v))
    B = basis(phi)
    X = np.concatenate([B, B * yaw[:, None], B * (v - v0)[:, None]], 1)
    coef, *_ = np.linalg.lstsq(X, Qm, rcond=None)
    pred = X @ coef
    r2 = 1 - ((Qm - pred) ** 2).sum(0) / ((Qm - Qm.mean(0)) ** 2).sum(0)
    # stride frequency vs speed, and the stance fraction / stance phase window per leg
    A = np.c_[np.ones_like(v), v - v0]
    fcoef, *_ = np.linalg.lstsq(A, dphi / (2 * np.pi), rcond=None)
    bins = np.linspace(-np.pi, np.pi, 37)
    idx = np.clip(np.digitize(phi, bins) - 1, 0, 35)
    stance_p = np.array([[stance[idx == b, k].mean() for b in range(36)] for k in range(6)])
    nb = B.shape[1]
    np.savez(out, names=np.array(names), a=coef[:nb], b=coef[nb:2 * nb], c=coef[2 * nb:], r2=r2,
             v0=v0, f0=fcoef[0], f_per_v=fcoef[1], stance=stance_p, legs=np.array(LEGS6),
             yaw_sd=float(yaw.std()), seconds=len(phi) * dt, source=str(path))
    print(f"{len(phi) * dt:.1f} s of coordinated walking, speed median {v0:.2f} cm/s, "
          f"stride {fcoef[0]:.1f} Hz (+{fcoef[1]:.1f} Hz per cm/s), yaw rate sd {yaw.std():.2f} rad/s")
    for leg in LEGS6:
        print(f"  {leg:9s} " + "  ".join(f"{j}:{r2[names.index(f'{j}_{leg}')]:.2f}" for j in LEG_JOINTS))
    print(f"-> {out}  (R^2 of each joint above)")
    return out


class RealGait:
    """q_ref(phi, yaw rate) for all leg joints, from real_gait.npz."""

    def __init__(self, path=OUT):
        z = np.load(path)
        self.names = list(z["names"])
        self.a, self.b, self.c = z["a"], z["b"], z["c"]
        self.v0, self.f0, self.f_per_v = float(z["v0"]), float(z["f0"]), float(z["f_per_v"])
        self.stance = z["stance"]
        self.period_ms = 1000.0 / self.f0
        self.idx = {n: i for i, n in enumerate(self.names)}

    def q(self, phi, yaw=0.0, dv=0.0):
        B = basis(phi)
        return B @ self.a + yaw * (B @ self.b) + dv * (B @ self.c)

    def dq(self, phi, yaw=0.0, dv=0.0):
        """dq/dphi (rad per rad of phase)."""
        h = np.arange(1, N_HARM + 1)
        phi = float(phi)
        dB = np.r_[0.0, -h * np.sin(h * phi), h * np.cos(h * phi)]
        return dB @ self.a + yaw * (dB @ self.b) + dv * (dB @ self.c)

    def in_stance(self, phi):
        b = int((phi + np.pi) / (2 * np.pi) * 36) % 36
        return self.stance[:, b]


def figure(path=OUT, folder=HERE / "runs" / "real_gait"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder.mkdir(parents=True, exist_ok=True)
    g = RealGait(path)
    ph = np.linspace(-np.pi, np.pi, 200)
    fig, axs = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, leg in zip(axs.T.ravel(), LEGS6):
        for j in LEG_JOINTS:
            ax.plot(ph, g.q(ph)[:, g.idx[f"{j}_{leg}"]], label=j)
        ax.set_title(leg)
        ax.set_ylabel("rad")
    axs[0, 0].legend(fontsize=7)
    axs[-1, 0].set_xlabel("stride phase")
    axs[-1, 1].set_xlabel("stride phase")
    fig.suptitle(f"real fly gait (flybody mocap): {g.f0:.1f} Hz strides at {g.v0:.2f} cm/s")
    fig.tight_layout()
    fig.savefig(folder / "real_gait.png", dpi=110)
    print(f"-> {folder / 'real_gait.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    a = ap.parse_args()
    fit(Path(a.data))
    figure()
