"""GQE, Query2Box and BetaE written against one interface the DAG executor drives:

    anchor(e) project(q, r) intersect([q..]) negate(q)       -> query embedding [B, F]
    dist_pos(q, e) -> [B]    dist_set(q, ids) -> [B, m]      (smaller = more likely an answer)

Training shares one set of negatives across the batch, which keeps BetaE's KL a few matmuls.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def mlp(i, h, o, layers=2):
    dims = [i] + [h] * (layers - 1) + [o]
    seq = []
    for a, b in zip(dims[:-1], dims[1:]):
        seq += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*seq[:-1])


class Base(nn.Module):
    def __init__(self, E, R, d, gamma):
        super().__init__()
        self.E, self.R, self.d, self.gamma = E, R, d, gamma
        self.rng = (gamma + 2.0) / d

    def uniform_(self, *shape):
        return nn.Parameter(torch.empty(*shape).uniform_(-self.rng, self.rng))

    def negate(self, q):
        raise NotImplementedError("this model has no negation")


class GQE(Base):
    """Query = one point. Projection translates, intersection is a learned attention average."""
    F = property(lambda s: s.d)

    def __init__(self, E, R, d=256, gamma=24.0):
        super().__init__(E, R, d, gamma)
        self.ent, self.rel = self.uniform_(E, d), self.uniform_(2 * R, d)
        self.att = mlp(d, d, d)

    def anchor(self, e):
        return self.ent[e]

    def project(self, q, r):
        return q + self.rel[r]

    def intersect(self, qs):
        x = torch.stack(qs)                               # [n, B, d]
        return (F.softmax(self.att(x), 0) * x).sum(0)

    def dist_pos(self, q, e):
        return (self.ent[e] - q).abs().sum(-1)

    def dist_set(self, q, ids):
        return torch.cdist(q, self.ent[ids], p=1)


class Query2Box(Base):
    """Query = box (centre, offset). Entities inside the box are answers."""
    F = property(lambda s: 2 * s.d)

    def __init__(self, E, R, d=256, gamma=24.0, alpha=0.02):
        super().__init__(E, R, d, gamma)
        self.alpha = alpha
        self.ent = self.uniform_(E, d)
        self.rel_c, self.rel_o = self.uniform_(2 * R, d), self.uniform_(2 * R, d)
        self.att = mlp(d, d, d)
        self.o1, self.o2 = nn.Linear(d, d), nn.Linear(d, d)  # DeepSets gate on the offsets

    def anchor(self, e):
        c = self.ent[e]
        return torch.cat([c, torch.zeros_like(c)], -1)

    def project(self, q, r):
        c, o = q.chunk(2, -1)
        return torch.cat([c + self.rel_c[r], o + F.relu(self.rel_o[r])], -1)

    def intersect(self, qs):
        c, o = torch.stack(qs).chunk(2, -1)               # [n, B, d] each
        c = (F.softmax(self.att(c), 0) * c).sum(0)
        o = o.min(0).values * torch.sigmoid(self.o2(F.relu(self.o1(o)).mean(0)))
        return torch.cat([c, o], -1)

    def _dist(self, delta, o):
        return F.relu(delta - o).sum(-1) + self.alpha * torch.minimum(delta, o).sum(-1)

    def dist_pos(self, q, e):
        c, o = q.chunk(2, -1)
        return self._dist((self.ent[e] - c).abs(), o)

    def dist_set(self, q, ids):
        c, o = q.chunk(2, -1)
        return self._dist((self.ent[ids][None] - c[:, None]).abs(), o[:, None])


class BetaE(Base):
    """Query = d independent Beta(alpha, beta). Intersection multiplies densities, negation
    inverts the parameters, distance is KL(entity || query)."""
    F = property(lambda s: 2 * s.d)

    def __init__(self, E, R, d=256, gamma=24.0, hidden=512):
        super().__init__(E, R, d, gamma)
        self.ent, self.rel = self.uniform_(E, 2 * d), self.uniform_(2 * R, d)
        self.proj = mlp(3 * d, hidden, 2 * d, layers=3)
        self.att = mlp(2 * d, 2 * d, d)

    @staticmethod
    def reg(x):
        return torch.clamp(x + 1.0, 0.05, 1e9)

    def anchor(self, e):
        return self.reg(self.ent[e])

    def project(self, q, r):
        return self.reg(self.proj(torch.cat([q, self.rel[r]], -1)))

    def intersect(self, qs):
        x = torch.stack(qs)                               # [n, B, 2d]
        w = F.softmax(self.att(x), 0)                     # [n, B, d], shared by alpha and beta
        a, b = x.chunk(2, -1)
        return torch.cat([(w * a).sum(0), (w * b).sum(0)], -1)

    def negate(self, q):
        return 1.0 / q

    @staticmethod
    def _lnB(a, b):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    def _ent_terms(self, ids):
        """Everything in KL(e || q) that depends on the entity only, plus the digammas."""
        a, b = self.reg(self.ent[ids]).chunk(2, -1)
        pa, pb, ps = torch.digamma(a), torch.digamma(b), torch.digamma(a + b)
        const = (-self._lnB(a, b) + a * pa + b * pb - (a + b) * ps).sum(-1)
        return const, ps - pa, ps - pb

    def dist_pos(self, q, e):
        const, ka, kb = self._ent_terms(e)
        a2, b2 = q.chunk(2, -1)
        return self._lnB(a2, b2).sum(-1) + const + (a2 * ka).sum(-1) + (b2 * kb).sum(-1)

    def dist_set(self, q, ids):
        const, ka, kb = self._ent_terms(ids)
        a2, b2 = q.chunk(2, -1)
        return self._lnB(a2, b2).sum(-1, keepdim=True) + const[None] + a2 @ ka.T + b2 @ kb.T


MODELS = {"gqe": GQE, "q2b": Query2Box, "betae": BetaE}
