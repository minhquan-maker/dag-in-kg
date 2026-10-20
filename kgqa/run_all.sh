#!/bin/bash
# Full pipeline on CPU. Needs ~4 cores: the four models train in parallel, one thread each.
set -e
cd "$(dirname "$0")"
# steps are matched to each model's speed on CPU so all four finish in about the same time
STEPS_GQE=${STEPS_GQE:-50000}; STEPS_Q2B=${STEPS_Q2B:-30000}; STEPS_BETAE=${STEPS_BETAE:-20000}
mkdir -p data/FB15k-237 results
base=https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237
for s in train valid test; do
  [ -f data/FB15k-237/$s.txt ] || curl -sSfL -o data/FB15k-237/$s.txt $base/$s.txt
done
[ -f data/queries/stats.txt ] || python3 gen_queries.py
python3 train.py --model gqe   --steps $STEPS_GQE --gamma 24 --eval_every 5000 > results/gqe.log 2>&1 &
python3 train.py --model q2b   --steps $STEPS_Q2B --gamma 24 > results/q2b.log 2>&1 &
python3 train.py --model betae --steps $STEPS_BETAE --gamma 24 > results/betae.log 2>&1 &
python3 train.py --model betae --neg --steps $STEPS_BETAE --gamma 24 > results/betae_neg.log 2>&1 &
wait
python3 summarize.py results | tee results/summary.md
