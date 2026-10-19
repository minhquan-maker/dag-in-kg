"""Train a DAG embedding on WordNet nouns and evaluate is-a link prediction."""
import argparse, json, os, time
import numpy as np, torch
import dagkg, models


def sample_neg(rng, k, n, bsz, pool=None):
    """k random replacement parents per positive; with `pool`, half of them come from the
    parents of train edges (hard negatives), the rest are uniform."""
    neg = rng.randint(0, n, size=(bsz, k))
    if pool is not None:
        h = k // 2
        neg[:, :h] = pool[rng.randint(0, len(pool), size=(bsz, h))]
    return torch.from_numpy(neg)


@torch.no_grad()
def evaluate(model, n, anc, thr, pos, rng, pool=None, n_cand=100):
    """Classification F1 (balanced negatives, threshold chosen on val) and MRR of the
    true ancestor against `n_cand` random non-ancestors. Replacement nodes are uniform
    ('easy'), or drawn from `pool` ('hard': pool = parents of train edges, so that
    popularity of the parent gives no hint)."""
    draw = (lambda *sh: rng.randint(0, n, sh)) if pool is None else \
        (lambda *sh: pool[rng.randint(0, len(pool), sh)])
    u, v = torch.from_numpy(pos[:, 0]), torch.from_numpy(pos[:, 1])
    sp = model.score(u, v)
    # negatives: corrupt the child or the parent (50/50) with a random non-related node
    un, nv = u.clone(), v.clone()
    for j in range(len(pos)):
        a, b = int(u[j]), int(v[j])
        for t in range(20):  # corrupt the child; if impossible (e.g. parent = root), the parent
            r = int(draw(1)[0])
            na, nb = (r, b) if (j % 2 and t < 10) else (a, r)
            if na != nb and nb not in anc[na]:
                break
        un[j], nv[j] = na, nb
    sn = model.score(un, nv)
    if thr is None:  # pick the threshold maximising F1 on this (validation) set
        cands = torch.cat([sp, sn]).sort().values
        best = (-1, 0)
        for t in cands[::max(1, len(cands) // 400)]:
            f = f1(sp, sn, t)
            best = max(best, (f, float(t)))
        thr = best[1]
    cand = torch.from_numpy(draw(len(pos), n_cand))
    sc = model.score(u.unsqueeze(1).expand_as(cand), cand)
    rank = 1 + (sc > sp.unsqueeze(1)).sum(1).float()
    return {"f1": f1(sp, sn, thr), "mrr": float((1 / rank).mean())}, thr


def f1(sp, sn, t):
    tp, fn, fp = (sp >= t).sum().item(), (sp < t).sum().item(), (sn >= t).sum().item()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(models.MODELS), default="poincare")
    ap.add_argument("--edges", default="data/wordnet_nouns_edges.tsv")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--frac", type=float, default=0.25, help="fraction of non-basic train edges")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bsz", type=int, default=512)
    ap.add_argument("--neg", type=int, default=20)
    ap.add_argument("--hard_neg", action="store_true", help="mix parent-distributed negatives into training")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.RandomState(a.seed)
    names, edges = dagkg.load_edges(a.edges)
    n = len(names)
    anc = dagkg.closure(n, edges)
    train, val, test = dagkg.split(n, edges, anc, a.frac, a.seed)
    print(f"nodes={n} edges={len(edges)} closure={sum(map(len, anc))} "
          f"train={len(train)} val={len(val)} test={len(test)}", flush=True)
    model = models.MODELS[a.model](n, a.dim) if a.model != "popularity" else models.Popularity(n, train)
    if a.model == "popularity":
        a.epochs = 0
    lr = a.lr or {"poincare": 1.0, "order": 1e-2, "box": 1e-2, "popularity": 0}[a.model]
    opt = None if (model.riemannian or a.model == "popularity") else torch.optim.Adam(model.parameters(), lr=lr)
    hist, t0 = [], time.time()
    for ep in range(1, a.epochs + 1):
        perm = rng.permutation(len(train))
        tot = 0.0
        # Poincare: first epochs use a 10x smaller lr ("burn-in") as in the paper
        cur = lr / 10 if (model.riemannian and ep <= max(1, a.epochs // 10)) else lr
        for i in range(0, len(perm), a.bsz):
            b = torch.from_numpy(train[perm[i:i + a.bsz]])
            loss = model.loss(b[:, 0], b[:, 1], sample_neg(rng, a.neg, n, len(b), train[:, 1] if a.hard_neg else None))
            loss.backward()
            if model.riemannian:
                model.rsgd(cur)
            else:
                opt.step(); opt.zero_grad()
            tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == a.epochs:
            m, _ = evaluate(model, n, anc, None, val, np.random.RandomState(1), train[:, 1])
            print(f"ep {ep:3d} loss {tot/len(train):.4f} val[hard] F1 {m['f1']:.3f} MRR {m['mrr']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            hist.append((ep, tot / len(train), m))
    res = {}
    for mode, pool in (("easy", None), ("hard", train[:, 1])):
        _, thr = evaluate(model, n, anc, None, val, np.random.RandomState(1), pool)
        res[mode], _ = evaluate(model, n, anc, thr, test, np.random.RandomState(2), pool)
        print("TEST", mode, res[mode], flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"args": vars(a), "test": res, "hist": hist},
              open(f"{a.out}/{a.model}_frac{a.frac}{'_hardneg' if a.hard_neg else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
