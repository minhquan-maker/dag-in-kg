"""FB15k-237 loading and a CSR graph with vectorised set projection.

Relations are doubled: id r is the forward edge, id r + R its inverse. Three nested graphs
(train, train+valid, train+valid+test) give the easy / hard answer split used for evaluation.
"""
import os
import numpy as np


def read_triples(path):
    return [tuple(l.rstrip("\n").split("\t")) for l in open(path)]


def load_fb15k237(root):
    tr, va, te = (read_triples(os.path.join(root, f"{s}.txt")) for s in ("train", "valid", "test"))
    ent = {e: i for i, e in enumerate(sorted({x for h, _, t in tr for x in (h, t)}))}
    rel = {r: i for i, r in enumerate(sorted({r for _, r, _ in tr}))}

    def enc(ts):  # drop triples whose entity / relation never appears in train
        out = [(ent[h], rel[r], ent[t]) for h, r, t in ts if h in ent and t in ent and r in rel]
        return np.unique(np.array(out, dtype=np.int64), axis=0)

    return enc(tr), enc(va), enc(te), len(ent), len(rel)


class Graph:
    def __init__(self, triples, E, R):
        self.E, self.R = E, R
        h, r, t = triples.T
        heads = np.concatenate([h, t])
        rels = np.concatenate([r, r + R])
        tails = np.concatenate([t, h])
        # relation-major CSR: (rel, head) -> tails
        key = rels * E + heads
        order = np.argsort(key, kind="stable")
        self.nbr = tails[order].astype(np.int32)
        self.ptr = np.zeros(2 * R * E + 1, dtype=np.int64)
        np.cumsum(np.bincount(key, minlength=2 * R * E), out=self.ptr[1:])
        # entity-major CSR: head -> (rel, tail), used to pick a random incident edge
        o2 = np.argsort(heads, kind="stable")
        self.e_rel = rels[o2].astype(np.int32)
        self.e_nbr = tails[o2].astype(np.int32)
        self.e_ptr = np.zeros(E + 1, dtype=np.int64)
        np.cumsum(np.bincount(heads, minlength=E), out=self.e_ptr[1:])
        self.has_edge = np.flatnonzero(np.diff(self.e_ptr) > 0)

    def inv(self, r):
        return (r + self.R) % (2 * self.R)

    def step(self, S, r):
        """Projection of the entity set S through relation r."""
        keys = r * self.E + S.astype(np.int64)
        st = self.ptr[keys]
        ln = self.ptr[keys + 1] - st
        tot = int(ln.sum())
        if tot == 0:
            return np.empty(0, dtype=np.int32)
        offs = np.repeat(st - (np.cumsum(ln) - ln), ln) + np.arange(tot)
        return np.unique(self.nbr[offs])
