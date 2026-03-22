"""
Step 126 — k-NN on XOR: solves at d=2 (99.9%) but fails with noise dims
Recovered from Leo session 0606b161 (line 9964).
All code was run inline (python -c) in the Leo session.

Context: k-NN era, Steps 97-165. Self-improving substrate investigation.
Session date: 2026-03-15.

Tests k-NN on XOR task (class = sign(x[0]) XOR sign(x[1])).
Result: 99.9% at d=2, 93.1% at d=5, 58.9% at d=50. Adding oracle quad feature x0*x1: +10pp.
Gap: substrate needs to discover WHICH features to compute, not just retrieve.
"""
# Recovered inline code below:
# Two commands: initial XOR test, then dimensionality sweep

# Command 1: Basic XOR test
python -c "
import torch, torch.nn.functional as F

device = 'cuda'
d = 100  # embedding dimension
n_train = 5000
n_test = 1000
k = 5

# Generate XOR-like task: class depends on sign of first two features
# Class = (x[0] > 0) XOR (x[1] > 0)
# k-NN should struggle because same-class regions are disjoint

def make_data(n):
    x = torch.randn(n, d, device=device)
    # Label based on XOR of signs of first two dims
    y = ((x[:, 0] > 0).long() ^ (x[:, 1] > 0).long())
    return F.normalize(x, dim=1), y

train_x, train_y = make_data(n_train)
test_x, test_y = make_data(n_test)

# k-NN classification
sims = test_x @ train_x.T
scores = torch.zeros(n_test, 2, device=device)
for c in range(2):
    m = train_y == c
    cs = sims[:, m]
    scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
preds = scores.argmax(dim=1)
acc_knn = (preds == test_y).float().mean().item() * 100

# 1-NN
preds_1nn = train_y[sims.argmax(dim=1)]
acc_1nn = (preds_1nn == test_y).float().mean().item() * 100

# Linear classifier (optimal for comparison)
# Logistic regression on the XOR features
from torch import nn, optim
model = nn.Linear(d, 2).to(device)
opt = optim.Adam(model.parameters(), lr=0.01)
for _ in range(1000):
    logits = model(train_x)
    loss = F.cross_entropy(logits, train_y)
    opt.zero_grad(); loss.backward(); opt.step()
acc_linear = (model(test_x).argmax(1) == test_y).float().mean().item() * 100

# Quadratic features (should solve XOR)
# Add x[0]*x[1] as a feature
train_xq = torch.cat([train_x, (train_x[:, 0] * train_x[:, 1]).unsqueeze(1)], dim=1)
test_xq = torch.cat([test_x, (test_x[:, 0] * test_x[:, 1]).unsqueeze(1)], dim=1)
train_xq = F.normalize(train_xq, dim=1)
test_xq = F.normalize(test_xq, dim=1)

sims_q = test_xq @ train_xq.T
scores_q = torch.zeros(n_test, 2, device=device)
for c in range(2):
    m = train_y == c
    cs = sims_q[:, m]
    scores_q[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
preds_q = scores_q.argmax(dim=1)
acc_knn_quad = (preds_q == test_y).float().mean().item() * 100

print(f'XOR task (d={d}):')
print(f'  Random baseline: 50.0%')
print(f'  1-NN:            {acc_1nn:.1f}%')
print(f'  top-5 k-NN:      {acc_knn:.1f}%')
print(f'  Linear:          {acc_linear:.1f}%')
print(f'  k-NN + quad feat:{acc_knn_quad:.1f}%')
print(f'')
print(f'k-NN beats random? {\"YES\" if acc_knn > 55 else \"NO\"}')
print(f'Quad feature helps? {\"YES\" if acc_knn_quad > acc_knn + 1 else \"NO\"}')
" 2>&1

# Command 2: Dimensionality sweep
python -c "
import torch, torch.nn.functional as F

device = 'cuda'
k = 5
n_train, n_test = 5000, 1000

for d in [2, 5, 10, 20, 50, 100]:
    x_tr = torch.randn(n_train, d, device=device)
    y_tr = ((x_tr[:, 0] > 0).long() ^ (x_tr[:, 1] > 0).long())
    x_te = torch.randn(n_test, d, device=device)
    y_te = ((x_te[:, 0] > 0).long() ^ (x_te[:, 1] > 0).long())
    
    x_tr_n = F.normalize(x_tr, dim=1)
    x_te_n = F.normalize(x_te, dim=1)
    
    sims = x_te_n @ x_tr_n.T
    scores = torch.zeros(n_test, 2, device=device)
    for c in range(2):
        m = y_tr == c; cs = sims[:, m]
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    acc = (scores.argmax(1) == y_te).float().mean().item() * 100
    
    # Also test with product feature appended
    x_tr_q = torch.cat([x_tr, (x_tr[:, 0] * x_tr[:, 1]).unsqueeze(1)], 1)
    x_te_q = torch.cat([x_te, (x_te[:, 0] * x_te[:, 1]).unsqueeze(1)], 1)
    x_tr_q = F.normalize(x_tr_q, dim=1)
    x_te_q = F.normalize(x_te_q, dim=1)
    sims_q = x_te_q @ x_tr_q.T
    scores_q = torch.zeros(n_test, 2, device=device)
    for c in range(2):
        m = y_tr == c; cs = sims_q[:, m]
        scores_q[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    acc_q = (scores_q.argmax(1) == y_te).float().mean().item() * 100
    
    print(f'd={d:3d}: kNN={acc:.1f}% kNN+quad={acc_q:.1f}% delta={acc_q-acc:+.1f}pp')
" 2>&1