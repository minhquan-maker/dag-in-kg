"""Sample logical queries by walking a witness entity backwards through a query DAG.

Pick a witness w, then for each projection op choose a random edge x --r--> (current target)
so the chosen anchors and relations are guaranteed to reach w. A negated branch gets an
independent random witness, so it excludes w only by chance and is checked afterwards.
For valid/test the first projection into w is forced to be a held-out edge, which makes hard
answers (reachable only with held-out edges) common instead of rare.
"""
import argparse, os, time
from multiprocessing import Pool
import numpy as np
from kg import load_fb15k237, Graph
import dagq

CAP_TRAIN, CAP_EVAL = 400, 100
G = {}  # filled in main(); inherited by forked workers


def sample_node(prog, i, target, st, rng, g):
    op = prog[i]
    if op[0] == "a":
        st["A"][op[1]] = target
        return True
    if op[0] == "p":
        slot = op[2]
        if slot in st["R"]:  # relation already fixed (shared slot in 'up')
            r = st["R"][slot]
            k = g.inv(r) * g.E + target
            lo, hi = g.ptr[k], g.ptr[k + 1]
            if hi == lo:
                return False
            x = int(g.nbr[rng.integers(lo, hi)])
        elif st["force"] is not None and st["force"][2] == target:
            x, r, _ = st["force"]
            st["force"] = None
        else:
            lo, hi = g.e_ptr[target], g.e_ptr[target + 1]
            if hi == lo:
                return False
            j = rng.integers(lo, hi)
            r, x = int(g.inv(g.e_rel[j])), int(g.e_nbr[j])
        st["R"][slot] = r
        return sample_node(prog, op[1], x, st, rng, g)
    if op[0] == "i":
        return all(sample_node(prog, j, target, st, rng, g) for j in op[1])
    w2 = int(g.has_edge[rng.integers(len(g.has_edge))])  # negated branch: unrelated witness
    return sample_node(prog, op[1], w2, st, rng, g)


def backtracks(name, rels, g):
    """True if a chain immediately undoes itself (r followed by its inverse)."""
    for prog in dagq.T[name]:
        for op in prog:
            if op[0] == "p" and prog[op[1]][0] == "p":
                if rels[op[2]] == g.inv(rels[prog[op[1]][2]]):
                    return True
    return False


def sample_query(name, g, rng, force=None):
    na, nr = dagq.n_slots(name)
    st = {"A": {}, "R": {}, "force": force}
    t = force[2] if force is not None else int(g.has_edge[rng.integers(len(g.has_edge))])
    progs = dagq.T[name]
    if not sample_node(progs[0], len(progs[0]) - 1, t, st, rng, g):
        return None
    for prog in progs[1:]:
        for _ in range(30):
            w2 = int(g.has_edge[rng.integers(len(g.has_edge))])
            if sample_node(prog, len(prog) - 1, w2, st, rng, g):
                break
        else:
            return None
    if len(st["A"]) != na or len(st["R"]) != nr:
        return None
    anchors = np.array([st["A"][k] for k in range(na)], dtype=np.int32)
    rels = np.array([st["R"][k] for k in range(nr)], dtype=np.int32)
    return None if backtracks(name, rels, g) else (anchors, rels)


def pack(rows):
    ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([len(r) for r in rows], out=ptr[1:])
    val = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    return ptr, val.astype(np.int32)


def make(task):
    split, name, n, seed = task
    rng = np.random.default_rng(seed)
    if split == "train":
        small, big, held, cap = None, G["tr"], None, CAP_TRAIN
    elif split == "valid":
        small, big, held, cap = G["tr"], G["va"], G["va_new"], CAP_EVAL
    else:
        small, big, held, cap = G["va"], G["te"], G["te_new"], CAP_EVAL
    seen, A, Rl, easy, hard, tries = set(), [], [], [], [], 0
    t0 = time.time()
    while len(A) < n and tries < n * 400:
        tries += 1
        force = None
        if held is not None:  # orient a random held-out edge, its tail is the witness
            h, r, t = held[rng.integers(len(held))]
            force = (int(h), int(r), int(t)) if rng.random() < .5 else (int(t), int(r) + big.R, int(h))
        q = sample_query(name, big, rng, force)
        if q is None:
            continue
        key = (tuple(q[0]), tuple(q[1]))
        if key in seen:
            continue
        ans = dagq.answers(name, *q, big)
        if split == "train":
            if not 1 <= len(ans) <= cap:
                continue
            e, h_ = ans, ans[:0]
        else:
            a_small = dagq.answers(name, *q, small)
            e, h_ = np.intersect1d(a_small, ans, assume_unique=True), np.setdiff1d(ans, a_small, assume_unique=True)
            if len(h_) < 1 or len(ans) > cap:
                continue
        seen.add(key)
        A.append(q[0]); Rl.append(q[1]); easy.append(e); hard.append(h_)
    out = {"anchors": np.stack(A), "rels": np.stack(Rl)}
    if split == "train":
        out["ptr"], out["val"] = pack(easy)
    else:
        out["easy_ptr"], out["easy_val"] = pack(easy)
        out["hard_ptr"], out["hard_val"] = pack(hard)
    print(f"{split:5s} {name:3s} {len(A):6d} queries  {tries:7d} tries  {time.time()-t0:5.0f}s", flush=True)
    return task, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/FB15k-237")
    ap.add_argument("--out", default="data/queries")
    ap.add_argument("--n_train", type=int, default=40000)
    ap.add_argument("--n_valid", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tr, va, te, E, R = load_fb15k237(a.data)
    print(f"entities {E}  relations {R} (x2 with inverses)  train/valid/test {len(tr)}/{len(va)}/{len(te)}")
    G["tr"] = Graph(tr, E, R)
    G["va"] = Graph(np.concatenate([tr, va]), E, R)
    G["te"] = Graph(np.concatenate([tr, va, te]), E, R)
    G["va_new"], G["te_new"] = va, te
    os.makedirs(a.out, exist_ok=True)
    tasks = [("train", s, a.n_train, a.seed + i) for i, s in enumerate(dagq.TRAIN_POS + dagq.TRAIN_NEG)]
    tasks += [(sp, s, n, a.seed + 100 * (1 + j) + i) for j, (sp, n) in enumerate((("valid", a.n_valid), ("test", a.n_test)))
              for i, s in enumerate(dagq.EVAL_POS + dagq.EVAL_NEG)]
    with Pool(a.procs) as p:
        for (split, name, _, _), out in p.imap_unordered(make, tasks):
            np.savez_compressed(os.path.join(a.out, f"{split}-{name}.npz"), **out)
    with open(os.path.join(a.out, "stats.txt"), "w") as f:
        f.write(f"{E} {R}\n")


if __name__ == "__main__":
    main()
