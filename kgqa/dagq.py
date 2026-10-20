"""Query structures written as small DAG programs.

A program is a list of ops in topological order; op i can only read ops j < i:
    ('a', k)         anchor entity number k
    ('p', j, k)      project op j through relation slot k
    ('i', (j, ...))  intersect
    ('n', j)         complement
The program's value is its last op. A union query has several programs (disjunctive normal
form): the answer is the union of their values, and a model scores an entity by its best branch.
"""
import numpy as np

T = {
    "1p": [[("a", 0), ("p", 0, 0)]],
    "2p": [[("a", 0), ("p", 0, 0), ("p", 1, 1)]],
    "3p": [[("a", 0), ("p", 0, 0), ("p", 1, 1), ("p", 2, 2)]],
    "2i": [[("a", 0), ("p", 0, 0), ("a", 1), ("p", 2, 1), ("i", (1, 3))]],
    "3i": [[("a", 0), ("p", 0, 0), ("a", 1), ("p", 2, 1), ("a", 2), ("p", 4, 2), ("i", (1, 3, 5))]],
    "ip": [[("a", 0), ("p", 0, 0), ("a", 1), ("p", 2, 1), ("i", (1, 3)), ("p", 4, 2)]],
    "pi": [[("a", 0), ("p", 0, 0), ("p", 1, 1), ("a", 1), ("p", 3, 2), ("i", (2, 4))]],
    "2u": [[("a", 0), ("p", 0, 0)], [("a", 1), ("p", 0, 1)]],
    "up": [[("a", 0), ("p", 0, 0), ("p", 1, 2)], [("a", 1), ("p", 0, 1), ("p", 1, 2)]],
    "2in": [[("a", 0), ("p", 0, 0), ("a", 1), ("p", 2, 1), ("n", 3), ("i", (1, 4))]],
    "3in": [[("a", 0), ("p", 0, 0), ("a", 1), ("p", 2, 1), ("a", 2), ("p", 4, 2), ("n", 5), ("i", (1, 3, 6))]],
    "inp": [[("a", 0), ("p", 0, 0), ("a", 1), ("p", 2, 1), ("n", 3), ("i", (1, 4)), ("p", 5, 2)]],
    "pin": [[("a", 0), ("p", 0, 0), ("p", 1, 1), ("a", 1), ("p", 3, 2), ("n", 4), ("i", (2, 5))]],
    "pni": [[("a", 0), ("p", 0, 0), ("p", 1, 1), ("n", 2), ("a", 1), ("p", 4, 2), ("i", (3, 5))]],
}
TRAIN_POS = ["1p", "2p", "3p", "2i", "3i"]
TRAIN_NEG = ["2in", "3in", "inp", "pin", "pni"]
EVAL_POS = TRAIN_POS + ["ip", "pi", "2u", "up"]
EVAL_NEG = TRAIN_NEG


def n_slots(name):
    ops = [o for prog in T[name] for o in prog]
    na = 1 + max(o[1] for o in ops if o[0] == "a")
    nr = 1 + max(o[2] for o in ops if o[0] == "p")
    return na, nr


def answers(name, anchors, rels, G):
    """Exact answer set on graph G (numpy sets), union over the branches."""
    out = None
    for prog in T[name]:
        v = []
        for op in prog:
            if op[0] == "a":
                v.append(np.array([anchors[op[1]]], dtype=np.int32))
            elif op[0] == "p":
                v.append(G.step(v[op[1]], int(rels[op[2]])))
            elif op[0] == "i":
                s = v[op[1][0]]
                for j in op[1][1:]:
                    s = np.intersect1d(s, v[j], assume_unique=True)
                v.append(s)
            else:
                v.append(np.setdiff1d(np.arange(G.E, dtype=np.int32), v[op[1]], assume_unique=True))
        out = v[-1] if out is None else np.union1d(out, v[-1])
    return out


def run_program(model, prog, anchors, rels):
    """Execute one branch on a batch: anchors [B, A], rels [B, R] -> query embedding [B, F]."""
    v = []
    for op in prog:
        if op[0] == "a":
            v.append(model.anchor(anchors[:, op[1]]))
        elif op[0] == "p":
            v.append(model.project(v[op[1]], rels[:, op[2]]))
        elif op[0] == "i":
            v.append(model.intersect([v[j] for j in op[1]]))
        else:
            v.append(model.negate(v[op[1]]))
    return v[-1]
