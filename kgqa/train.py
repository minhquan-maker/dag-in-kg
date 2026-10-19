"""Train one model on generated queries and evaluate it on hard answers only (filtered)."""
import argparse, json, math, os, time
import numpy as np, torch
import torch.nn.functional as F
import dagq
from models import MODELS


class Split:
    """All queries of one structure: anchors, relations and CSR answer lists."""
    def __init__(self, path):
        z = np.load(path)
        self.anchors = torch.from_numpy(z["anchors"].astype(np.int64))
        self.rels = torch.from_numpy(z["rels"].astype(np.int64))
        self.z = {k: z[k] for k in z.files if k.endswith(("ptr", "val"))}
        self.n = len(self.anchors)


def embed(model, name, anchors, rels):
    return [dagq.run_program(model, prog, anchors, rels) for prog in dagq.T[name]]


def train_batch(model, name, sp, B, k, rng):
    i = rng.integers(0, sp.n, B)
    ptr, val = sp.z["ptr"], sp.z["val"]
    pos = val[ptr[i] + (rng.random(B) * (ptr[i + 1] - ptr[i])).astype(np.int64)]
    q = embed(model, name, sp.anchors[i], sp.rels[i])[0]
    neg = torch.from_numpy(rng.integers(0, model.E, k))
    g = model.gamma
    pos_loss = -F.logsigmoid(g - model.dist_pos(q, torch.from_numpy(pos.astype(np.int64)))).mean()
    neg_loss = -F.logsigmoid(model.dist_set(q, neg) - g).mean()
    return (pos_loss + neg_loss) / 2


@torch.no_grad()
def evaluate(model, splits, names, limit=None, chunk=16):
    model.eval()
    res = {}
    for name in names:
        sp = splits[name]
        n = sp.n if limit is None else min(limit, sp.n)
        rr, h1, h3, h10 = [], [], [], []
        for s in range(0, n, chunk):
            idx = torch.arange(s, min(s + chunk, n))
            d = torch.stack([model.dist_set(q, torch.arange(model.E)) for q in
                             embed(model, name, sp.anchors[idx], sp.rels[idx])]).min(0).values
            for b, i in enumerate(idx.tolist()):
                z = sp.z
                easy = z["easy_val"][z["easy_ptr"][i]:z["easy_ptr"][i + 1]]
                hard = z["hard_val"][z["hard_ptr"][i]:z["hard_ptr"][i + 1]]
                sc = -d[b]
                mask = torch.zeros(model.E, dtype=torch.bool)
                mask[torch.from_numpy(np.concatenate([easy, hard]).astype(np.int64))] = True
                rest = torch.sort(sc[~mask]).values              # non-answers, ascending
                hs = sc[torch.from_numpy(hard.astype(np.int64))]
                rank = 1 + len(rest) - torch.searchsorted(rest, hs, right=True)
                rank = rank.float()
                rr.append((1 / rank).mean().item()); h1.append((rank <= 1).float().mean().item())
                h3.append((rank <= 3).float().mean().item()); h10.append((rank <= 10).float().mean().item())
        res[name] = {"mrr": 100 * np.mean(rr), "h1": 100 * np.mean(h1), "h3": 100 * np.mean(h3),
                     "h10": 100 * np.mean(h10), "n": n}
    model.train()
    return res


def avg(res, names):
    return float(np.mean([res[n]["mrr"] for n in names]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--neg", action="store_true", help="also train and test on negation structures")
    ap.add_argument("--queries", default="data/queries")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=24.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--negatives", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--valid_limit", type=int, default=100)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tag = a.tag or a.model + ("_neg" if a.neg else "")
    os.makedirs(a.out, exist_ok=True)

    E, R = map(int, open(os.path.join(a.queries, "stats.txt")).read().split())
    tr_names = dagq.TRAIN_POS + (dagq.TRAIN_NEG if a.neg else [])
    ev_names = dagq.EVAL_POS + (dagq.EVAL_NEG if a.neg else [])
    load = lambda sp, ns: {n: Split(os.path.join(a.queries, f"{sp}-{n}.npz")) for n in ns}
    train, valid, test = load("train", tr_names), load("valid", ev_names), load("test", ev_names)

    model = MODELS[a.model](E, R, a.dim, a.gamma)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lr_at = lambda s: a.lr * min(1, (s + 1) / a.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    log, t0, run = [], time.time(), 0.0
    print(f"[{tag}] {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, structs {tr_names}", flush=True)
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        name = tr_names[rng.integers(len(tr_names))]
        loss = train_batch(model, name, train[name], a.batch, a.negatives, rng)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0:
            print(f"[{tag}] step {step:6d} loss {run:.4f} lr {lr_at(step):.2e} {time.time()-t0:6.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate(model, valid, ev_names, a.valid_limit)
            log.append({"step": step, "valid_avg_mrr": avg(v, dagq.EVAL_POS), "loss": run})
            print(f"[{tag}] step {step} valid(first {a.valid_limit}/struct) avg MRR {log[-1]['valid_avg_mrr']:.2f}", flush=True)
    t = evaluate(model, test, ev_names)
    out = {"tag": tag, "args": vars(a), "train_seconds": time.time() - t0, "test": t, "curve": log,
           "test_avg_pos": avg(t, dagq.EVAL_POS)}
    if a.neg:
        out["test_avg_neg"] = avg(t, dagq.EVAL_NEG)
    json.dump(out, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)
    print(f"[{tag}] TEST avg MRR (9 pos structs) {out['test_avg_pos']:.2f}" +
          (f"  neg {out['test_avg_neg']:.2f}" if a.neg else ""), flush=True)


if __name__ == "__main__":
    main()
