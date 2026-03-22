# Step 134 — recovered from CC session (inline Bash, keyword match)
python -c "
import torch, torch.nn.functional as F, time

device = 'cuda'
d = 20
n_train, n_test = 2000, 500

x_tr = torch.randn(n_train, d, device=device)
y_tr = ((x_tr[:, 0] > 0).long() ^ (x_tr[:, 1] > 0).long())
x_te = torch.randn(n_test, d, device=device)
y_te = ((x_te[:, 0] > 0).long() ^ (x_te[:, 1] > 0).long())

k = 5

def knn_acc(V, labels, test_x, test_y, k=5):
    V_n = F.normalize(V, dim=1); te = F.normalize(test_x, dim=1)
    sims = te @ V_n.T
    scores = torch.zeros(te.shape[0], 2, device=device)
    for c in range(2):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == test_y).float().mean().item() * 100

def class_coherence(V, labels):
    V_n = F.normalize(V, dim=1)
    total = count = 0
    for c in range(labels.max().item() + 1):
        m = labels == c
        if m.sum() < 2: continue
        Vc = V_n[m]
        centroid = F.normalize(Vc.mean(0), dim=0)
        total += (Vc @ centroid).mean().item()
        count += 1
    return total / count if count > 0 else 0

# UNIFIED PROCESS WITH COHERENCE-GUIDED FEATURE DISCOVERY
# After storing all training data, discover features that improve coherence,
# augment ALL stored vectors with discovered features, then classify.

print(f'--- Baseline (raw d={d}) ---')
acc_base = knn_acc(x_tr, y_tr, x_te, y_te)
print(f'k-NN: {acc_base:.1f}%')

# Phase 1: Store all training data (always-spawn)
V = x_tr.clone()
labels = y_tr.clone()
raw_tr = x_tr.clone()  # keep raw for feature computation
raw_te = x_te.clone()

# Phase 2: Discover features (coherence-guided, greedy)
discovered_pairs = []
for step in range(3):
    base_coh = class_coherence(V, labels)
    best_pair = None
    best_delta = 0
    
    # Sample 100 random pairs (scalable)
    for _ in range(100):
        i, j = torch.randint(0, d, (2,)).tolist()
        if i == j: continue
        i, j = min(i,j), max(i,j)
        if (i,j) in discovered_pairs: continue
        
        feat = (raw_tr[:, i] * raw_tr[:, j]).unsqueeze(1)
        aug = torch.cat([V, feat], 1)
        delta = class_coherence(aug, labels) - base_coh
        if delta > best_delta:
            best_delta = delta
            best_pair = (i, j)
    
    if best_pair is None:
        print(f'Step {step+1}: no improvement found')
        break
    
    discovered_pairs.append(best_pair)
    # Augment stored vectors
    feat_tr = (raw_tr[:, best_pair[0]] * raw_tr[:, best_pair[1]]).unsqueeze(1)
    feat_te = (raw_te[:, best_pair[0]] * raw_te[:, best_pair[1]]).unsqueeze(1)
    V = torch.cat([V, feat_tr], 1)
    test_aug = torch.cat([x_te if step == 0 else test_aug, feat_te], 1) if step > 0 else torch.cat([x_te, feat_te], 1)
    
    acc = knn_acc(V, labels, test_aug if step > 0 else torch.cat([x_te, feat_te], 1), y_te)
    is_target = best_pair == (0,1)
    print(f'Step {step+1}: found {best_pair} coh_delta=+{best_delta:.6f} acc={acc:.1f}% {\"*** TARGET ***\" if is_target else \"\"}')

# Rebuild test with all discovered features
test_final = x_te.clone()
for (i,j) in discovered_pairs:
    feat = (raw_te[:, i] * raw_te[:, j]).unsqueeze(1)
    test_final = torch.cat([test_final, feat], 1)

acc_final = knn_acc(V, labels, test_final, y_te)
print(f'\\nFinal: {acc_final:.1f}% (was {acc_base:.1f}%, delta=+{acc_final-acc_base:.1f}pp)')
print(f'Discovered pairs: {discovered_pairs}')
" 2>&1