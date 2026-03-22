# Step 131 — recovered from CC session (inline Bash, keyword match)
python -c "
import torch, torch.nn.functional as F

device = 'cuda'

# Test at d=50 (1225 pairs — does target still rank #1?)
d = 50
n_train = 2000
x_tr = torch.randn(n_train, d, device=device)
y_tr = ((x_tr[:, 0] > 0).long() ^ (x_tr[:, 1] > 0).long())

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

base_coh = class_coherence(x_tr, y_tr)

# Score all pairs
scores = {}
for i in range(d):
    for j in range(i+1, d):
        feat = (x_tr[:, i] * x_tr[:, j]).unsqueeze(1)
        aug = torch.cat([x_tr, feat], 1)
        scores[(i,j)] = class_coherence(aug, y_tr) - base_coh

ranked = sorted(scores.items(), key=lambda x: -x[1])
print(f'd={d}, {len(ranked)} pairs')
print(f'Top 5:')
for (i,j), delta in ranked[:5]:
    t = '*** TARGET ***' if (i,j)==(0,1) else ''
    print(f'  ({i:2d},{j:2d}): +{delta:.6f} {t}')

for rank, ((i,j), delta) in enumerate(ranked):
    if (i,j) == (0,1):
        print(f'Target (0,1) rank: {rank+1} / {len(ranked)}')
        break

# Also test d=100
d2 = 100
x_tr2 = torch.randn(n_train, d2, device=device)
y_tr2 = ((x_tr2[:, 0] > 0).long() ^ (x_tr2[:, 1] > 0).long())
base2 = class_coherence(x_tr2, y_tr2)
scores2 = {}
for i in range(d2):
    for j in range(i+1, d2):
        feat = (x_tr2[:, i] * x_tr2[:, j]).unsqueeze(1)
        aug = torch.cat([x_tr2, feat], 1)
        scores2[(i,j)] = class_coherence(aug, y_tr2) - base2
ranked2 = sorted(scores2.items(), key=lambda x: -x[1])
for rank, ((i,j), delta) in enumerate(ranked2):
    if (i,j) == (0,1):
        print(f'd={d2}: Target (0,1) rank: {rank+1} / {len(ranked2)} (delta=+{delta:.6f})')
        break
print(f'd={d2} top-1: {ranked2[0]}')
" 2>&1