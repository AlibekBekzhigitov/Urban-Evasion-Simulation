#!/bin/sh
# Regenerates every number and figure used in the QASH v2 stage-1 article.
# Output macro files: qash_v2_numbers*.tex ; figure: qash_v2_riskmaps.pdf
set -e
OUT=${1:-out}
python3 qash_v2_encounter.py --out-dir "$OUT"                                    # base case, dt = 0.1
python3 qash_v2_encounter.py --out-dir "$OUT" --no-figure --tag spec  --v-catch 4.0 --t-chase 10
python3 qash_v2_encounter.py --out-dir "$OUT" --no-figure --tag fast  --tau-lo 0.5 --tau-hi 1.5
python3 qash_v2_encounter.py --out-dir "$OUT" --no-figure --tag cheap --c-wait 0.005
python3 qash_v2_encounter.py --out-dir "$OUT" --no-figure --tag coarse --dt 0.5   # convergence check
python3 qash_v2_encounter.py --out-dir "$OUT" --no-figure --tag mid    --dt 0.25
python3 qash_v2_encounter.py --out-dir "$OUT" --no-figure --tag finest --dt 0.05
