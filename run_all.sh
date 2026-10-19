#!/bin/bash
# Full experiment: ~15 min on 4 CPU cores. Results -> results/*.json, results/*.log
set -e
cd "$(dirname "$0")"
python3 tools/build_wordnet_dag.py
mkdir -p results
for frac in 0.0 0.5; do
  python3 train_eval.py --model popularity --frac $frac --out results > results/popularity_frac$frac.log 2>&1
  OMP_NUM_THREADS=1 python3 train_eval.py --model poincare --lr 30 --hard_neg --frac $frac --epochs 30 --out results > results/poincare_frac$frac.log 2>&1 &
  OMP_NUM_THREADS=1 python3 train_eval.py --model order --hard_neg --frac $frac --epochs 30 --out results > results/order_frac$frac.log 2>&1 &
  OMP_NUM_THREADS=1 python3 train_eval.py --model box --hard_neg --frac $frac --epochs 30 --out results > results/box_frac$frac.log 2>&1 &
  wait
done
python3 tools/summarize.py
