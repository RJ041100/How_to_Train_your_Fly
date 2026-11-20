#!/usr/bin/env python3
"""Six-leg walking circuit from the male CNS connectome (same format and rules as the
fly_arm front-leg circuit, leg_circuit_L/, and the quadruped circuit):

    command (descending neurons) -> premotor interneurons -> leg motor neurons -> joints
                                          ^
                           leg proprioceptors (joint angle / load) <- MuJoCo sensors

Only the motor pathways of walking are kept: the motor neurons of the 24 leg joints the
fly model drives (coxa, femur, tibia, tarsus x 6 legs, muscles as in fly_motor.py), the
interneurons that make >= 1 % of some motor neuron's input, the descending neurons with
the largest reach onto each leg (direct + one hop) plus the known walking/steering types,
and each leg's proprioceptors that synapse into the circuit.

    python walk_circuit_build.py            -> walk_circuit/ (neurons.csv, synapses.csv, weights.npz, README.md)

Weights are graded as in fly_arm: W[post, pre] = sign x synapses / (post's total input).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fly_body import LEG_SEGMENT, MN_JOINT, SIDE  # noqa: E402
from paths import DATA_DIR  # noqa: E402

OUT = HERE / "walk_circuit"
PREMOTOR_MIN_SHARE = 0.01        # of some motor neuron's total input
N_COMMAND_PER_LEG = 20           # descending neurons with the largest reach onto each leg
N_COMMAND_TOP = 40              # ... plus the top ones onto all legs together
EXTRA_DN = ["DNa02", "DNg13", "DNb02", "DNa01", "DNg100", "MDN", "oDN1", "DNg97", "DNb05", "DNb06"]
LEG_NERVE = {"T1": "ProLN", "T2": "MesoLN", "T3": "MetaLN"}
PROPRIO_SUBCLASSES = {"chordotonal organ", "campaniform sensilla", "hair plate"}
PROPRIO_TYPES = ("SNpp", "SApp")
SENSORS_PER_LEG = 12             # strongest proprioceptors into the circuit, per leg
SENSORY_MIN_SYN = 5
TAU_MS = {"command": 10.0, "premotor": 10.0, "motor": 10.0, "sensory": 5.0}


def main():
    cols = ["Body_id", "status", "superclass", "subclass", "type", "somaSide", "rootSide",
            "entryNerve", "nt_sign", "consensus_nt", "in_weight"]
    ann = pd.read_feather(DATA_DIR / "neurons.feather", columns=cols)
    ann = ann[ann.status == "Traced"].set_index("Body_id")
    w = pd.read_feather(DATA_DIR / "connectivity.feather", columns=["Body_id_pre", "Body_id_post", "weight", "sign"])
    w = w[(w.weight >= 5) & (w.Body_id_pre != w.Body_id_post)
          & w.Body_id_pre.isin(ann.index) & w.Body_id_post.isin(ann.index)]
    tot = w.groupby("Body_id_post").weight.sum()
    sc, typ = ann.superclass, ann.type
    side = ann.somaSide.where(ann.somaSide.isin(["L", "R"]), ann.rootSide)

    # ---- motor neurons of the 24 driven leg joints ---------------------------------
    m = ann[(sc == "vnc_motor") & ann.subclass.isin(LEG_SEGMENT) & ann.somaSide.isin(SIDE)
            & typ.isin(MN_JOINT)]
    mn_leg = (m.subclass.map(LEG_SEGMENT) + "_" + m.somaSide.map(SIDE))
    leg_of = mn_leg.to_dict()
    mn = set(m.index)
    print(f"motor neurons: {len(mn)} ({mn_leg.value_counts().sort_index().to_dict()})")

    # ---- premotor -------------------------------------------------------------------
    e = w[w.Body_id_post.isin(mn)].copy()
    e["share"] = e.weight / e.Body_id_post.map(tot)
    e = e[e.Body_id_pre.map(sc).isin(["vnc_intrinsic", "ascending_neuron"])]
    best = e.groupby("Body_id_pre").share.max()
    premotor = set(best.index[best >= PREMOTOR_MIN_SHARE])
    to_leg = (e[e.Body_id_pre.isin(premotor)].assign(leg=lambda x: x.Body_id_post.map(leg_of))
              .groupby(["Body_id_pre", "leg"]).weight.sum().unstack(fill_value=0))
    frac = to_leg.max(axis=1) / to_leg.sum(axis=1)
    for i in premotor:
        leg_of[i] = to_leg.idxmax(axis=1)[i] if frac[i] >= 0.8 else "shared"
    print(f"premotor: {len(premotor)} (one leg: {(frac >= 0.8).sum()}, several: {(frac < 0.8).sum()})")

    # ---- command: descending neurons by reach onto each leg ---------------------------
    dns = set(ann.index[sc == "descending_neuron"])
    pm_out = (w[w.Body_id_pre.isin(premotor) & w.Body_id_post.isin(mn)]
              .assign(leg=lambda x: x.Body_id_post.map(leg_of))
              .groupby(["Body_id_pre", "leg"]).weight.sum().unstack(fill_value=0))
    direct = (w[w.Body_id_pre.isin(dns) & w.Body_id_post.isin(mn)]
              .assign(leg=lambda x: x.Body_id_post.map(leg_of))
              .groupby(["Body_id_pre", "leg"]).weight.sum().unstack(fill_value=0))
    d2 = w[w.Body_id_pre.isin(dns) & w.Body_id_post.isin(premotor)].copy()
    d2["share"] = d2.weight / d2.Body_id_post.map(tot)
    two = pd.DataFrame(0.0, index=sorted(set(d2.Body_id_pre)), columns=pm_out.columns)
    for leg in pm_out.columns:
        s = d2.share * d2.Body_id_post.map(pm_out[leg]).fillna(0)
        two[leg] = s.groupby(d2.Body_id_pre).sum()
    reach = two.add(direct.reindex(columns=two.columns), fill_value=0).fillna(0)
    command = set(reach.sum(axis=1).nlargest(N_COMMAND_TOP).index)
    for leg in reach.columns:
        command |= set(reach[leg].nlargest(N_COMMAND_PER_LEG).index)
    found = []
    for name in EXTRA_DN:
        ids = set(ann.index[(sc == "descending_neuron") & (typ == name)])
        if ids:
            command |= ids
            found.append(f"{name}({len(ids)})")
    print(f"command: {len(command)} descending neurons; walking types found: {' '.join(found)}")
    for i in command:
        leg_of[i] = "command"

    # ---- proprioceptors of each leg ---------------------------------------------------
    sensory = set()
    for leg in sorted(set(mn_leg)):
        seg, s = leg.split("_")
        s = "L" if s == "left" else "R"
        ls = ann[sc.isin(["vnc_sensory", "sensory_ascending"])
                 & ann.entryNerve.astype(str).str.contains(LEG_NERVE[seg]) & (side == s)]
        prop = ls.index[ls.subclass.isin(PROPRIO_SUBCLASSES) | typ.reindex(ls.index).astype(str).str.startswith(PROPRIO_TYPES)]
        into = w[w.Body_id_pre.isin(prop) & w.Body_id_post.isin(premotor | mn)]
        strength = into.groupby("Body_id_pre").weight.sum()
        got = set(strength[strength >= SENSORY_MIN_SYN].nlargest(SENSORS_PER_LEG).index)
        for i in got:
            leg_of[i] = leg
        sensory |= got
        print(f"  sensors {leg}: {len(prop)} proprioceptors on the leg, {len(strength)} into the circuit, kept {len(got)}")

    # ---- assemble -------------------------------------------------------------------
    role = {}
    for ids, r in ((sensory, "sensory"), (command, "command"), (premotor, "premotor"), (mn, "motor")):
        for i in ids:
            role[i] = r
    ids = sorted(role)
    nodes = pd.DataFrame({"index": np.arange(len(ids)), "root_id": ids})
    nodes["type"] = typ.reindex(ids).values
    nodes["role"] = nodes.root_id.map(role)
    nodes["leg"] = nodes.root_id.map(leg_of)
    nodes["side"] = side.reindex(ids).values
    nodes["super_class"] = sc.reindex(ids).values
    is_mn = nodes.role == "motor"
    nodes["joint"] = np.where(is_mn, nodes.type.map(lambda t: MN_JOINT.get(t, ("", 0))[0]), "")
    nodes["direction"] = np.where(is_mn, nodes.type.map(lambda t: MN_JOINT.get(t, ("", 0))[1]), 0)
    nodes["top_nt"] = ann.consensus_nt.reindex(ids).values
    nodes["sign"] = ann.nt_sign.reindex(ids).fillna(0).astype(int).values
    nodes["tau_ms"] = nodes.role.map(TAU_MS)
    nodes["total_input_synapses"] = tot.reindex(ids).fillna(0).astype(int).values
    idx = dict(zip(nodes.root_id, nodes["index"]))

    ed = w[w.Body_id_pre.isin(idx) & w.Body_id_post.isin(idx) & w.sign.isin([1, -1])].copy()
    ed = ed[ed.Body_id_post.map(role) != "sensory"]
    ed["pre_index"] = ed.Body_id_pre.map(idx)
    ed["post_index"] = ed.Body_id_post.map(idx)
    ed["pre_role"] = ed.Body_id_pre.map(role)
    ed["post_role"] = ed.Body_id_post.map(role)
    ed["post_total_input"] = ed.Body_id_post.map(tot)
    ed["weight_graded"] = ed.sign * ed.weight / ed.post_total_input
    ed = ed.rename(columns={"Body_id_pre": "pre_id", "Body_id_post": "post_id", "weight": "synapses"})

    OUT.mkdir(exist_ok=True)
    nodes.to_csv(OUT / "neurons.csv", index=False)
    ed.to_csv(OUT / "synapses.csv", index=False)
    np.savez_compressed(OUT / "weights.npz", post=ed.post_index.values, pre=ed.pre_index.values,
                        w=ed.weight_graded.values.astype(np.float32), synapses=ed.synapses.values,
                        neuron_ids=nodes.root_id.values)
    reach.assign(total=reach.sum(axis=1), type=typ.reindex(reach.index).values) \
         .sort_values("total", ascending=False).head(100).to_csv(OUT / "command_candidates.csv")
    roles = nodes.groupby(["role", "leg"]).size().unstack(fill_value=0)
    print(f"\nCIRCUIT: {len(nodes)} neurons, {len(ed)} connections\n{roles.to_string()}")
    (OUT / "README.md").write_text(f"""# Six-leg walking circuit (male CNS), built by walk_circuit_build.py

{len(nodes)} neurons, {len(ed)} chemical connections (>= 5 synapses).
command (descending) -> premotor -> motor (24 leg joints of the flybody fly); leg sensors -> premotor/motor.

Roles: {nodes.role.value_counts().to_dict()}

Rules: motor neurons of the coxa / femur / tibia / tarsus muscles of all six legs (fly_motor.MN_JOINT);
premotor = VNC intrinsic or ascending neurons with >= {PREMOTOR_MIN_SHARE:.0%} of some motor neuron's input;
command = top {N_COMMAND_PER_LEG} descending neurons per leg + top {N_COMMAND_TOP} overall by reach (direct + one hop),
plus walking types found among {', '.join(EXTRA_DN)}; sensors = the {SENSORS_PER_LEG} leg proprioceptors
(chordotonal, campaniform, hair plate, SNpp/SApp) per leg with the most synapses into the circuit.

neurons.csv : index, root_id, type, role, leg, side, super_class, joint + direction (motor), top_nt, sign, tau_ms,
              total_input_synapses
weights.npz : post, pre, w = sign x synapses / post's total input (graded, as fly_arm), synapses, neuron_ids
""")
    print(f"-> {OUT}/")


if __name__ == "__main__":
    main()
