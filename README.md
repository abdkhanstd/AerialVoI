# AerialVoI: value of aerial traffic information for dedicated-lane decisions

Code and run-level results for a study of which layers of an aerial (UAV or helicopter) traffic survey are needed to
decide whether, and where, to open a dedicated lane (DL) for connected automated vehicles (CAVs) in mixed traffic.

Two aerially recorded sites are rebuilt in SUMO 1.24: an urban arterial junction (site A) and a freeway weaving section
(site B). Planner models that use progressively more of the survey (imagery only, observed volumes, movement counts
and signal plan, full trajectories) each choose a DL layout on their own selection seeds. Their choices are then
scored in four plausible calibrated reference models on separate evaluation seeds. Every layout is simulated for
every scenario and seed, so every planner, rule and oracle is a lookup in the same grid (common random numbers).

## Data (public, not redistributed here except the derived site A tracks)

| Site | Data set | Source | Licence |
|---|---|---|---|
| A, urban arterial junction, Dongguan | CityArtery (UAV, 1 Hz) | Mendeley Data, doi:10.17632/ftzzmh2cmt.1 | CC BY 4.0 |
| B, freeway weaving, Chicago | TGSIM I-90/I-94 stationary, runs 2/4/6 (helicopter, 10 Hz) | data.transportation.gov dataset `9uas-hf8b` | US DOT public data |

`data/siteA_tracks.parquet` is derived from CityArtery (CC BY 4.0, Wang et al.) and included so that the site A
scripts run without the raw files. The site B tracks (about 220 MB) are rebuilt from the raw TGSIM file.
To rebuild everything from the raw data, place the files as

```
data_raw/cityartery/190b112d__ssh_data_2024060*_*.csv   (the five processed multi-intersection files)
data_raw/tgsim_i9094s/I90_94_stationary_final.csv
data_raw/tgsim_i9094s/I-90-stationary-Run_{2,4,6}-geometry-with-ramps.csv
data_raw/tgsim_i9094s/ann/Annotation on Regions/*.jpg
```

with `data_raw/` next to this repository folder, and convert the TGSIM CSV once to parquet with
`python explore/tgsim_profile.py` (run from the folder that contains `data_raw/`).

## Setup

```
pip install -r requirements.txt
```

`libsumo` is used when available (a grid run takes 50 to 70 s of one CPU core), otherwise TraCI.

## Layout

```
geotz/      simulation, planners, calibration and analysis code
data/       derived aerial layers (station axes, cells, demand, calibration targets, lane-change events)
results/    run-level grid tables and every analysis output
runs/       calibration grid results (runs/calibA, runs/calibB); simulation outputs are written here too
explore/    conversion of the raw TGSIM file
```

## Pipeline

```
python geotz/extract_a.py              # site A station axes, lane widths, landmarks, cells, tracks
python geotz/extract_b.py              # site B station axes, cells, tracks
python geotz/events.py                 # lane-change events
python geotz/data_stats.py             # descriptive statistics of the two sites
python geotz/calib_a.py 40             # calibration grids (results included in runs/)
python geotz/calib_b.py 40
python geotz/calib_report.py           # observed vs simulated targets  -> results/calibration.json
python geotz/calib_holdout.py          # leave-one-recording-out check  -> results/calib_holdout.csv
GEOTZ_LEVEL=B python geotz/fidelity.py # fidelity of each model level at observed demand (any level below)

# simulation grids (one per planner or reference model, see the table below), e.g.
GEOTZ_LEVEL=G0 GEOTZ_GRID_SUFFIX=G0 GEOTZ_SEEDS=0,1,2,3 python geotz/grid.py A 40
python geotz/collect.py A G0 runs/grid_AG0/res          # -> results/grid_A_G0.csv

python geotz/voi.py                    # regrets, paired tests, Holm, CIs, TOST -> results/voi_summary.json
GEOTZ_METRIC=delay python geotz/voi.py # same with delay as the only cost       -> results/voi_summary_delay.json
python geotz/direction.py              # per-direction analysis at site A       -> results/direction.json
python geotz/mechanism.py              # WB DL penalty, seed-half steps          -> results/mechanism.json
python geotz/cheap_queue.py            # downstream queue from video without calibration -> results/cheap_queue.json
python geotz/hv_penalty.py             # human-driver delay under the best DL    -> results/hv_penalty.json
python geotz/allred_check.py           # check without the 1 s all-red interval  -> results/allred_check.json
python geotz/check_teleports.py 40     # teleports in the most congested scenario
python geotz/figs_data.py              # data and model figures   (written to figs/)
python geotz/figs_voi.py               # result figures           (written to figs/)
```

`grid.py` is resumable (one JSON file per run, finished runs are skipped) and can be sharded across machines with a
third argument `k/K`. `collect.py` merges result folders from several machines.

## Model levels and grids

`GEOTZ_LEVEL` selects the information the site model uses (`site_a.py`, `site_b.py`). Selection seeds are 0-3 and
8-11 (grids with suffix `s`), evaluation seeds 4-7. `GEOTZ_NOALLRED=1` removes the 1 s all-red phase that
netconvert inserts at the site A junction in some layouts (grids with suffix `n`).

| Level | Information | Grids (site A) | Grids (site B) |
|---|---|---|---|
| G0 | imagery only, demand from lane-count capacity at v/c 0.75, generic 120 s signal | G0, G0s, G0n | G0 |
| G0V | imagery + observed total volume, generic 120 s signal | G0V, G0Vs | |
| G0H, G0F | imagery, capacity default at v/c 0.9 / 1.0, generic signal | G0H(s), G0F(s) | |
| G | imagery + observed total volume, observed cycle with equal split | G3, G3s, G3n | G3 |
| C | + movement counts and observed signal plan | C3, C3s, C3n | C3 |
| B | + trajectory layer (downstream queue and calibrated behaviour); reference B | v3, v3s, v3n | v3 |
| B2 | second-best calibration; reference B' | B23, B23n | B23 |
| R3, R4 | references selected in the leave-one-recording-out calibration | R3, R4, R3n, R4n | R3, R4 |
| AB1 | B without the downstream queue | AB1 | AB1 |
| AB1g, AB1e | AB1 with only the calibrated time gap / only random entry lanes | AB1g, AB1e | |
| AB2 | B without calibrated behaviour | AB2, AB2s | AB2 |
| G0Vq, G0Hq | G0V / G0H + the calibrated downstream queue | G0Vq(s), G0Hq(s) | |
| G0Vc, G0Vk | G0V + downstream signal with green from counts / from edge density and speed | G0Vc(s), G0Vk(s) | |
| G0Vg, G0Vo | G0V + counts green in the generic 120 s cycle, at offset 0 / 60 s | G0Vg(s), G0Vo(s) | |
| G0Vx | G0V + downstream signal with the generic green of the planner's own signal | G0Vx(s) | |

Each site A grid covers 20 layouts plus no DL, the six CAV penetrations 0.1 to 0.6, the demand factors and platoon
sizes of the design, for its seeds. The reference B grids also hold the extension to penetrations 0.7 and 0.8.
The 52 grids in `results/` hold 128,328 runs.

## Results

`results/grid_<site>_<grid>.csv` has one row per run and decision unit (site A: EB and WB; site B: the corridor)
with delay, time exposed TTC of human-driven followers, the cost `J`, throughput, DL use, platoon statistics and
per-class delays. `results/voi_decisions*.csv` lists every choice of every planner and rule with its regret in
every reference, and the JSON files hold the statistics computed from them.

## Licence

MIT, see `LICENSE`.
