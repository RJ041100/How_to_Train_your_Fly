# Data (not in git)

Only needed to rebuild the circuit or the gait model; the derived files are committed.

## MaleCNS v1.0 → `data/malecns/` (or set `MALECNS_DIR`)

Three files of the MaleCNS v1.0 flat-connectome release (Janelia FlyEM):

```
body-annotations-male-cns-v1.0-minconf-0.5.feather       ~14 MB
body-neurotransmitters-male-cns-v1.0.feather             ~43 MB
connectome-weights-male-cns-v1.0-minconf-0.5.feather     ~1.05 GB
```

sha256 of the copies used:

```
2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2  body-annotations-male-cns-v1.0-minconf-0.5.feather
95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621  body-neurotransmitters-male-cns-v1.0.feather
e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1  connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

`python walk/prepare_connectome.py` writes `data/processed/` (or `FLY_DATA_DIR`).

## flybody walking motion capture → `data/flybody/` (or set `FLYBODY_WALKING_DATA`)

`datasets_walking-imitation.zip` (3.0 GB) from figshare, doi:10.25378/janelia.25309105
(https://ndownloader.figshare.com/files/51196868; figshare may require a browser download). Only the small file
inside is used:

```
walking-dataset-small_female-only_snippets-100_min-len-0.5s_trk-files-0-9.hdf5   (~95 MB, 100 clips, 73 s)
```
