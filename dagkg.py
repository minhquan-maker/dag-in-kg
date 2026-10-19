"""DAG utilities: loading, Kahn topological sort, DP transitive closure, reduction."""
from collections import defaultdict, deque
import numpy as np


def load_edges(path):
    """Return (names, edges) with edges as (child, parent) int ids."""
    ids, names, edges = {}, [], []
    for line in open(path):
        c, p, cn, pn = line.rstrip("\n").split("\t")
        for s, n in ((c, cn), (p, pn)):
            if s not in ids:
                ids[s] = len(names)
                names.append(n)
        edges.append((ids[c], ids[p]))
    return names, edges


def topo_order(n, edges):
    """Kahn's algorithm, parents first. Raises if the graph has a cycle."""
    indeg = [0] * n
    kids = defaultdict(list)
    for c, p in edges:
        indeg[c] += 1  # edge p -> c in processing order
        kids[p].append(c)
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in kids[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != n:
        raise ValueError("graph has a cycle, not a DAG")
    return order


def closure(n, edges):
    """Ancestor sets by DP over topological order: anc[c] = U_p ({p} | anc[p])."""
    parents = defaultdict(list)
    for c, p in edges:
        parents[c].append(p)
    anc = [None] * n
    for u in topo_order(n, edges):
        s = set()
        for p in parents[u]:
            s.add(p)
            s |= anc[p]
        anc[u] = s
    return anc


def reduction(n, edges, anc):
    """Transitive reduction: drop (c,p) if p is an ancestor of another parent of c."""
    parents = defaultdict(set)
    for c, p in edges:
        parents[c].add(p)
    return [(c, p) for c, p in edges
            if not any(p in anc[q] for q in parents[c] if q != p)]


def split(n, edges, anc, frac_nonbasic, seed=0):
    """Protocol of Ganea et al. 2018: all basic (reduced) edges are train; of the
    remaining closure edges 10% test, 5% val, and `frac_nonbasic` of the rest train."""
    rng = np.random.RandomState(seed)
    basic = set(reduction(n, edges, anc))
    nonbasic = [(c, p) for c in range(n) for p in anc[c] if (c, p) not in basic]
    rng.shuffle(nonbasic)
    nt, nv = int(.10 * len(nonbasic)), int(.05 * len(nonbasic))
    test, val, rest = nonbasic[:nt], nonbasic[nt:nt + nv], nonbasic[nt + nv:]
    train = list(basic) + rest[:int(frac_nonbasic * len(rest))]
    return [np.array(x, dtype=np.int64).reshape(-1, 2) for x in (train, val, test)]
