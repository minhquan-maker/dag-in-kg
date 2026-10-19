"""Collect results/*.json into a markdown table."""
import glob, json
rows = []
for f in sorted(glob.glob("results/*.json")):
    r = json.load(open(f)); a = r["args"]
    rows.append((a["frac"], a["model"], r["test"]))
order = {"popularity": 0, "poincare": 1, "order": 2, "box": 3}
print("| train non-basic | model | easy F1 | easy MRR | hard F1 | hard MRR |\n|---|---|---|---|---|---|")
for frac, m, t in sorted(rows, key=lambda x: (x[0], order[x[1]])):
    print(f"| {frac:.0%} | {m} | {t['easy']['f1']:.3f} | {t['easy']['mrr']:.3f} | {t['hard']['f1']:.3f} | {t['hard']['mrr']:.3f} |")
