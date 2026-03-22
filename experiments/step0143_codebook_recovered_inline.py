"""
Step 143 — Compositional feature discovery: sum then sum%2
Recovered from Leo session 0606b161 (line 10166).
All code was run inline (python -c) in the Leo session.

Context: k-NN era, Steps 97-165. Self-improving substrate investigation.
Session date: 2026-03-15.

Tests compositional coherence — discover 'sum' as Layer 1 feature, then 'sum%2' as Layer 2.
Result: raw bits 75.4%, +sum 80.5%. Layer 2 candidates: sum_sq > cos_sum_pi.
Coherence finds coarse features, not task-specific ones. Missing: task-aware signal.
"""
# Recovered inline code below:
python -c "
import torch, torch.nn.functional as F
device = 'cuda'

vocab = 2; context = 8; n_train = 1000

X_tr = torch.randint(0, 2, (n_train, context), device=device).float()
y_tr = (X_tr.sum(dim=1) % 2).long()
X_te = torch.zeros(256, context, device=device)
for i in range(256):
    for b in range(context):
        X_te[i, b] = (i >> b) & 1
y_te = (X_te.sum(dim=1) % 2).long()

def knn_acc(V, labels, test_x, test_y, k=5):
    V_n = F.normalize(V, dim=1); te = F.normalize(test_x, dim=1)
    sims = te @ V_n.T
    scores = torch.zeros(test_x.shape[0], 2, device=device)
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
        Vc = V_n[m]; centroid = F.normalize(Vc.mean(0), dim=0)
        total += (Vc @ centroid).mean().item(); count += 1
    return total / count if count > 0 else 0

print('=== Step 143: Compositional Feature Discovery ===')

V = X_tr.clone(); labels = y_tr.clone()
raw = X_tr.clone(); raw_te = X_te.clone()
test_current = X_te.clone()

acc = knn_acc(V, labels, test_current, y_te)
print(f'Layer 0 (raw bits): {acc:.1f}%')

# Layer 1: discover sum (best single feature from raw bits)
# Instead of trying all pairs, try AGGREGATION features: sum, mean, max, min, etc.
sum_feat_tr = raw.sum(1, keepdim=True)
sum_feat_te = raw_te.sum(1, keepdim=True)
V = torch.cat([V, sum_feat_tr], 1)
test_current = torch.cat([test_current, sum_feat_te], 1)
acc = knn_acc(V, labels, test_current, y_te)
print(f'Layer 1 (+sum): {acc:.1f}%')

# Layer 2: discover features OF the augmented representation
# Now the codebook has [bits, sum]. Try features of the new space.
# Key candidate: sum modulo 2 (parity) — can be expressed as:
# (sum - 2*floor(sum/2)) which is just the fractional part of sum/2

# Test candidates on the AUGMENTED space
aug_dim = V.shape[1]  # 8 + 1 = 9
base_coh = class_coherence(V, labels)

candidates = {}
# sum % 2
feat = (raw.sum(1) % 2).unsqueeze(1)
candidates['sum_mod2'] = class_coherence(torch.cat([V, feat], 1), labels) - base_coh

# sum % 3
feat = (raw.sum(1) % 3).unsqueeze(1)
candidates['sum_mod3'] = class_coherence(torch.cat([V, feat], 1), labels) - base_coh

# sum % 4  
feat = (raw.sum(1) % 4).unsqueeze(1)
candidates['sum_mod4'] = class_coherence(torch.cat([V, feat], 1), labels) - base_coh

# sum > 4 (threshold)
feat = (raw.sum(1) > 4).float().unsqueeze(1)
candidates['sum_gt4'] = class_coherence(torch.cat([V, feat], 1), labels) - base_coh

# sum * sum
feat = (raw.sum(1) ** 2).unsqueeze(1)
candidates['sum_sq'] = class_coherence(torch.cat([V, feat], 1), labels) - base_coh

# cos(sum * pi) — encodes parity as a continuous feature
feat = torch.cos(raw.sum(1) * 3.14159).unsqueeze(1)
candidates['cos_sum_pi'] = class_coherence(torch.cat([V, feat], 1), labels) - base_coh

# sin(sum * pi) — should be ~0 for all integers
feat = torch.sin(raw.sum(1) * 3.14159).unsqueeze(1)
candidates['sin_sum_pi'] = class_coherence(torch.cat([V, feat], 1), labels) - base_coh

ranked = sorted(candidates.items(), key=lambda x: -x[1])
print(f'\\nLayer 2 candidates (features of sum):')
for name, delta in ranked:
    print(f'  {name:15s}: +{delta:.6f}')

best_name, best_delta = ranked[0]
print(f'\\nBest: {best_name} (delta=+{best_delta:.6f})')

# Apply best feature
if best_name == 'sum_mod2':
    feat_tr = (raw.sum(1) % 2).unsqueeze(1)
    feat_te = (raw_te.sum(1) % 2).unsqueeze(1)
elif best_name == 'cos_sum_pi':
    feat_tr = torch.cos(raw.sum(1) * 3.14159).unsqueeze(1)
    feat_te = torch.cos(raw_te.sum(1) * 3.14159).unsqueeze(1)
else:
    feat_tr = feat_te = None

if feat_tr is not None:
    V2 = torch.cat([V, feat_tr], 1)
    test2 = torch.cat([test_current, feat_te], 1)
    acc2 = knn_acc(V2, labels, test2, y_te)
    print(f'Layer 2 (+{best_name}): {acc2:.1f}% (+{acc2-acc:.1f}pp from layer 1)')
    print(f'\\n=== TOTAL: {X_tr.shape[1]} raw → {acc2:.1f}% (was 76.6%) ===')
" 2>&1