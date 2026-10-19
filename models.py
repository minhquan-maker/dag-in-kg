"""Three DAG embeddings. score(u, v) is high when v is an ancestor of u (u is-a v)."""
import torch, torch.nn as nn, torch.nn.functional as F

EPS = 1e-5


class Poincare(nn.Module):
    """Nickel & Kiela 2017. Points in the unit ball, trained with Riemannian SGD."""
    riemannian = True

    def __init__(self, n, dim, alpha=0.1):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(n, dim).uniform_(-1e-3, 1e-3))
        self.alpha = alpha

    def dist(self, u, v):
        uu, vv = (u * u).sum(-1), (v * v).sum(-1)
        x = 1 + 2 * ((u - v) ** 2).sum(-1) / ((1 - uu).clamp_min(EPS) * (1 - vv).clamp_min(EPS))
        return torch.acosh(x.clamp_min(1 + EPS))

    def score(self, u, v):  # eval score: shorter distance, and v nearer the origin
        d = self.dist(self.emb[u], self.emb[v])
        nu, nv = self.emb[u].norm(dim=-1), self.emb[v].norm(dim=-1)
        return -(1 + self.alpha * (nv - nu)) * d

    def loss(self, u, v, neg):  # softmax over the true parent and sampled non-parents
        eu = self.emb[u].unsqueeze(1)
        cand = torch.cat([v.unsqueeze(1), neg], 1)
        d = self.dist(eu, self.emb[cand])
        return F.cross_entropy(-d, torch.zeros(len(u), dtype=torch.long))

    @torch.no_grad()
    def rsgd(self, lr):
        x = self.emb
        x -= lr * x.grad * ((1 - (x * x).sum(-1, keepdim=True)) ** 2 / 4)  # Riemannian gradient
        nrm = x.norm(dim=-1, keepdim=True)
        x.copy_(torch.where(nrm >= 1, x / nrm * (1 - 1e-3), x))  # project into the ball
        x.grad = None


class Order(nn.Module):
    """Vendrov et al. 2016. u is-a v  <=>  u >= v coordinatewise in the positive orthant."""
    riemannian = False

    def __init__(self, n, dim, margin=1.0):
        super().__init__()
        self.emb = nn.Parameter(torch.rand(n, dim))
        self.margin = margin

    def viol(self, u, v):
        return (F.relu(self.emb[v].abs() - self.emb[u].abs()) ** 2).sum(-1)

    def score(self, u, v):
        return -self.viol(u, v)

    def loss(self, u, v, neg):
        pos = self.viol(u, v)
        n = self.viol(u.unsqueeze(1).expand_as(neg), neg)
        return (pos.mean() + F.relu(self.margin - n).mean())


class Box(nn.Module):
    """Axis-aligned boxes (centre, offset). u is-a v  <=>  box(u) inside box(v)."""
    riemannian = False

    def __init__(self, n, dim, margin=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n, dim) * .1)
        self.o = nn.Parameter(torch.zeros(n, dim))
        self.margin = margin

    def viol(self, u, v):
        ou, ov = F.softplus(self.o[u]), F.softplus(self.o[v])
        return F.relu((self.c[u] - self.c[v]).abs() + ou - ov).sum(-1)

    def score(self, u, v):
        return -self.viol(u, v)

    def loss(self, u, v, neg):
        pos = self.viol(u, v)
        n = self.viol(u.unsqueeze(1).expand_as(neg), neg)
        return pos.mean() + F.relu(self.margin - n).mean()


class Popularity:
    """Untrained baseline: score(u, v) = log(1 + #train edges whose parent is v)."""
    riemannian = False

    def __init__(self, n, train):
        self.cnt = torch.log1p(torch.bincount(torch.from_numpy(train[:, 1]), minlength=n).float())

    def score(self, u, v):
        return self.cnt[v]


MODELS = {"poincare": Poincare, "order": Order, "box": Box, "popularity": None}
