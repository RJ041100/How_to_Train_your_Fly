#!/usr/bin/env python3
"""Make the two tables walk_circuit_build.py reads, from three MaleCNS v1.0 flat files.

    data/malecns/body-annotations-male-cns-v1.0-minconf-0.5.feather
    data/malecns/body-neurotransmitters-male-cns-v1.0.feather
    data/malecns/connectome-weights-male-cns-v1.0-minconf-0.5.feather
 -> data/processed/neurons.feather, connectivity.feather

Same rules as the original full pipeline (build_combined_data.py), limited to the columns the walking
project needs: only annotated bodies are kept; the transmitter used for signs is the consensus call,
falling back to the per-body prediction; ACh +1, GABA / glutamate / histamine -1, monoamines 0.

    python prepare_connectome.py
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from paths import DATA_DIR, MALECNS_DIR

SUFFIX = "male-cns-v1.0"
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "histamine": -1,
           "dopamine": 0, "octopamine": 0, "serotonin": 0}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ann = pd.read_feather(MALECNS_DIR / f"body-annotations-{SUFFIX}-minconf-0.5.feather")
    ann = ann.rename(columns={"bodyId": "Body_id"}).drop(columns=["somaLocation", "tosomaLocation"])
    ids = np.sort(ann.Body_id.to_numpy())
    nt = pd.read_feather(MALECNS_DIR / f"body-neurotransmitters-{SUFFIX}.feather")
    nt = nt.drop(columns="cell_type").rename(columns={"body": "Body_id"})
    nt = nt[nt.Body_id.isin(ids)]
    nt["nt_for_model"] = nt.consensus_nt.where(nt.consensus_nt != "unclear", nt.predicted_nt)
    nt["nt_sign"] = nt.nt_for_model.map(NT_SIGN)
    neurons = ann.merge(nt, on="Body_id", how="left")
    print(f"neurons: {len(neurons):,}")

    t = feather.read_table(MALECNS_DIR / f"connectome-weights-{SUFFIX}-minconf-0.5.feather")
    pre, post, w = (t.column(c).to_numpy() for c in ("body_pre", "body_post", "weight"))
    keep = np.isin(pre, ids) & np.isin(post, ids)
    con = pd.DataFrame({"Body_id_pre": pre[keep], "Body_id_post": post[keep], "weight": w[keep]})
    sign = neurons.set_index("Body_id").nt_sign
    con["sign"] = sign.reindex(con.Body_id_pre).to_numpy()
    con = con.sort_values(["Body_id_pre", "Body_id_post"]).reset_index(drop=True)
    print(f"connections: {len(con):,}")

    inp = con.groupby("Body_id_post").weight.sum()
    neurons["in_weight"] = neurons.Body_id.map(inp).fillna(0).astype(np.int64)
    feather.write_feather(pa.Table.from_pandas(neurons, preserve_index=False), DATA_DIR / "neurons.feather")
    feather.write_feather(pa.Table.from_pandas(con, preserve_index=False), DATA_DIR / "connectivity.feather")
    print(f"-> {DATA_DIR}")


if __name__ == "__main__":
    main()
