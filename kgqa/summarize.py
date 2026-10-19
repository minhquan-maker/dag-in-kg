"""Turn results/*.json into markdown tables."""
import json, os, sys
import dagq

res_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
runs = {}
for f in sorted(os.listdir(res_dir)):
    if f.endswith(".json"):
        r = json.load(open(os.path.join(res_dir, f)))
        runs[r["tag"]] = r
order = [t for t in ("gqe", "q2b", "betae", "betae_neg") if t in runs]
NAMES = {"gqe": "GQE", "q2b": "Query2Box", "betae": "BetaE", "betae_neg": "BetaE (+neg)"}

print("### MRR (%) on hard test answers\n")
print("| Model | " + " | ".join(dagq.EVAL_POS) + " | avg |")
print("|---|" + "---|" * (len(dagq.EVAL_POS) + 1))
for t in order:
    r = runs[t]
    print(f"| {NAMES[t]} | " + " | ".join(f"{r['test'][s]['mrr']:.1f}" for s in dagq.EVAL_POS) + f" | **{r['test_avg_pos']:.1f}** |")
neg = [t for t in order if "test_avg_neg" in runs[t]]
if neg:
    print("\n### Negation structures, MRR (%)\n")
    print("| Model | " + " | ".join(dagq.EVAL_NEG) + " | avg |")
    print("|---|" + "---|" * (len(dagq.EVAL_NEG) + 1))
    for t in neg:
        r = runs[t]
        print(f"| {NAMES[t]} | " + " | ".join(f"{r['test'][s]['mrr']:.1f}" for s in dagq.EVAL_NEG) + f" | **{r['test_avg_neg']:.1f}** |")
print("\n### Hits@k (%), average over the 9 positive structures\n")
print("| Model | Hits@1 | Hits@3 | Hits@10 | train time (s) |")
print("|---|---|---|---|---|")
for t in order:
    r = runs[t]
    m = lambda k: sum(r["test"][s][k] for s in dagq.EVAL_POS) / len(dagq.EVAL_POS)
    print(f"| {NAMES[t]} | {m('h1'):.1f} | {m('h3'):.1f} | {m('h10'):.1f} | {r['train_seconds']:.0f} |")
