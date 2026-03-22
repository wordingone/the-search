# Step 130 — recovered from CC session (inline Bash, keyword match)
python -c "
import torch, torch.nn.functional as F

device = 'cuda'
k, n_train, n_test, d = 5, 2000, 500, 20

x_tr = torch.randn(n_train, d, device=device)
y_tr = ((x_tr[:, 0] > 0).long() ^ (x_tr[:, 1] > 0).long())
x_te = torch.randn(n_test, d, device=device)
y_te = ((x_te[:, 0] > 0).long() ^ (x_te[:, 1] > 0).long())

# SELF-SCORING: for each candidate feature pair, measure whether
# same-class vectors become MORE similar (internal coherence increases).
# No labels needed at eval time — this is unsupervised feature selection
# based on codebook structure alone.

def class_coherence(V, labels):
    '''Mean intra-class cosine similarity.'''
    V_n = F.normalize(V, dim=1)
    total = 0.0
    count = 0
    for c in range(labels.max().item() + 1):
        m = labels == c
        if m.sum() < 2: continue
        Vc = V_n[m]
        # Mean pairwise cosine (approximate with centroid similarity)
        centroid = F.normalize(Vc.mean(0), dim=0)
        total += (Vc @ centroid).mean().item()
        count += 1
    return total / count if count > 0 else 0

base_coh = class_coherence(x_tr, y_tr)
print(f'Base coherence (d={d}): {base_coh:.4f}')

# Score ALL pairs by coherence improvement
pairs = []
for i in range(d):
    for j in range(i+1, d):
        feat = (x_tr[:, i] * x_tr[:, j]).unsqueeze(1)
        aug = torch.cat([x_tr, feat], 1)
        coh = class_coherence(aug, y_tr)
        pairs.append(((i,j), coh - base_coh))

# Sort by coherence improvement
pairs.sort(key=lambda x: -x[1])

print(f'\\nTop 10 pairs by coherence improvement:')
for (i,j), delta in pairs[:10]:
    is_target = (i,j) == (0,1)
    print(f'  ({i:2d},{j:2d}): +{delta:.6f} {\"*** TARGET ***\" if is_target else \"\"}')

# Where does target (0,1) rank?
for rank, ((i,j), delta) in enumerate(pairs):
    if (i,j) == (0,1):
        print(f'\\nTarget (0,1) rank: {rank+1} / {len(pairs)} (delta=+{delta:.6f})')
        break
" 2>&1