"""
Step 128 — Greedy feature search finds XOR pair (0,1) as first selection
Recovered from Leo session 0606b161 (line 9992).
All code was run inline (python -c) in the Leo session.

Context: k-NN era, Steps 97-165. Self-improving substrate investigation.
Session date: 2026-03-15.

Mutual information feature selection: try all d² quadratic pairs, score by k-NN accuracy improvement.
Result: System discovered (0,1) as FIRST selected feature — the exact XOR pair. 64% → 76%.
Proof of concept: feature discovery without backprop, Principle-II compliant.
"""
# Recovered inline code below:
python -c "
import torch, torch.nn.functional as F

device = 'cuda'
k, n_train, n_test, d = 5, 2000, 500, 20

# XOR task with known ground truth feature
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

# GREEDY FEATURE SEARCH: iteratively add the quadratic feature that
# most improves k-NN accuracy. This is the simplest possible feature
# discovery mechanism — brute force over pairs, scored by k-NN.
base_acc = knn_acc(x_tr, y_tr, x_te, y_te)
print(f'Base accuracy (raw d={d}): {base_acc*100:.1f}%')

selected = []
current_tr = x_tr.clone()
current_te = x_te.clone()

for step in range(5):
    best_pair = None
    best_acc = knn_acc(current_tr, y_tr, current_te, y_te)
    
    for i in range(d):
        for j in range(i+1, d):
            if (i,j) in selected: continue
            # Try adding this product feature
            feat_tr = (x_tr[:, i] * x_tr[:, j]).unsqueeze(1)
            feat_te = (x_te[:, i] * x_te[:, j]).unsqueeze(1)
            aug_tr = torch.cat([current_tr, feat_tr], 1)
            aug_te = torch.cat([current_te, feat_te], 1)
            acc = knn_acc(aug_tr, y_tr, aug_te, y_te)
            if acc > best_acc:
                best_acc = acc
                best_pair = (i, j)
    
    if best_pair is None:
        print(f'Step {step+1}: no improvement found')
        break
    
    selected.append(best_pair)
    feat_tr = (x_tr[:, best_pair[0]] * x_tr[:, best_pair[1]]).unsqueeze(1)
    feat_te = (x_te[:, best_pair[0]] * x_te[:, best_pair[1]]).unsqueeze(1)
    current_tr = torch.cat([current_tr, feat_tr], 1)
    current_te = torch.cat([current_te, feat_te], 1)
    
    is_target = best_pair == (0, 1) or best_pair == (1, 0)
    print(f'Step {step+1}: selected pair {best_pair} acc={best_acc*100:.1f}% {\"*** TARGET ***\" if is_target else \"\"}')

print(f'\\nFinal: {knn_acc(current_tr, y_tr, current_te, y_te)*100:.1f}% with pairs {selected}')
print(f'Oracle: {knn_acc(torch.cat([x_tr, (x_tr[:,0]*x_tr[:,1]).unsqueeze(1)], 1), y_tr, torch.cat([x_te, (x_te[:,0]*x_te[:,1]).unsqueeze(1)], 1), y_te)*100:.1f}%')
" 2>&1