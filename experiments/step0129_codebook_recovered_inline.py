"""
Step 129 — Random search for feature pairs: finds target at 500 candidates (0.3s)
Recovered from Leo session 0606b161 (line 9999).
All code was run inline (python -c) in the Leo session.

Context: k-NN era, Steps 97-165. Self-improving substrate investigation.
Session date: 2026-03-15.

Efficient feature search: instead of all d² pairs, sample random subsets.
Result: d=50 (1225 total pairs), 500 random draws found (0,1) at ~33.5% probability. 0.3s.
Limitation: search is EXTERNAL to substrate — not atomic.
"""
# Recovered inline code below:
python -c "
import torch, torch.nn.functional as F, time

device = 'cuda'
k, n_train, n_test, d = 5, 2000, 500, 50

x_tr = torch.randn(n_train, d, device=device)
y_tr = ((x_tr[:, 0] > 0).long() ^ (x_tr[:, 1] > 0).long())
x_te = torch.randn(n_test, d, device=device)
y_te = ((x_te[:, 0] > 0).long() ^ (x_te[:, 1] > 0).long())

def knn_acc(train_x, train_y, test_x, test_y, k=5):
    tr = F.normalize(train_x, dim=1)
    te = F.normalize(test_x, dim=1)
    sims = te @ tr.T
    scores = torch.zeros(te.shape[0], 2, device=device)
    for c in range(2):
        m = train_y == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == test_y).float().mean().item()

base_acc = knn_acc(x_tr, y_tr, x_te, y_te)
print(f'd={d}, base acc: {base_acc*100:.1f}%')

# RANDOM SEARCH: sample N random pairs, score each, keep the best
for n_candidates in [10, 50, 100, 500]:
    t0 = time.time()
    best_pair = None
    best_acc = base_acc
    found_target = False
    
    for _ in range(n_candidates):
        i, j = torch.randint(0, d, (2,)).tolist()
        if i == j: continue
        i, j = min(i,j), max(i,j)
        
        feat_tr = (x_tr[:, i] * x_tr[:, j]).unsqueeze(1)
        feat_te = (x_te[:, i] * x_te[:, j]).unsqueeze(1)
        aug_tr = torch.cat([x_tr, feat_tr], 1)
        aug_te = torch.cat([x_te, feat_te], 1)
        acc = knn_acc(aug_tr, y_tr, aug_te, y_te)
        
        if acc > best_acc:
            best_acc = acc
            best_pair = (i, j)
        if (i,j) == (0,1): found_target = True
    
    elapsed = time.time() - t0
    target_str = '*** FOUND (0,1) ***' if best_pair == (0,1) else f'got {best_pair}'
    print(f'n_cand={n_candidates:3d}: best_acc={best_acc*100:.1f}% {target_str} sampled_target={found_target} time={elapsed:.1f}s')

# Total pairs possible
n_total = d * (d-1) // 2
print(f'\\nTotal pairs for d={d}: {n_total}')
print(f'P(sample target in N draws) = 1 - (1 - 1/{n_total})^N')
for n in [10, 50, 100, 500]:
    p = 1 - (1 - 1/n_total)**n
    print(f'  N={n:3d}: P={p:.3f}')
" 2>&1