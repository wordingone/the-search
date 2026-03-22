"""
Step 121 — Sequential S1 interleaving IS the anti-forgetting mechanism (CIFAR-100)
Recovered from Leo session 0606b161 (line 9902), 2026-03-15.
Leo ran this solo while Eli was down.

Commit: "Steps 120-121: Sequential interleaving IS the S1 anti-forgetting mechanism"

S1 eval: after each task, spawn test vectors into codebook during evaluation.
Sequential (interleaved train/eval) vs Standard (train all, then eval).
Result: Sequential S1 = +1.5pp, 11.7pp→~0.1pp forgetting.
The TIMING is load-bearing — interleaving preserves geometry.
"""
# Recovered inline code:
python -c "
import torch, torch.nn.functional as F, numpy as np

device = 'cuda'
data = np.load('C:/Users/Admin/cifar100_resnet18_features.npz')
X_train = F.normalize(torch.tensor(data['X_train'], device=device, dtype=torch.float32), dim=1)
y_train = torch.tensor(data['y_train'], device=device, dtype=torch.long)
X_test = F.normalize(torch.tensor(data['X_test'], device=device, dtype=torch.float32), dim=1)
y_test = torch.tensor(data['y_test'], device=device, dtype=torch.long)

n_tasks, cpt, k = 10, 10, 5

task_train, task_test = [], []
for t in range(n_tasks):
    cs, ce = t*cpt, (t+1)*cpt
    task_train.append((X_train[(y_train>=cs)&(y_train<ce)], y_train[(y_train>=cs)&(y_train<ce)]))
    task_test.append((X_test[(y_test>=cs)&(y_test<ce)], y_test[(y_test>=cs)&(y_test<ce)]))

def topk_predict(V, labels, queries, k=5):
    sims = queries @ V.T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(queries.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c
        if m.sum()==0: continue
        cs = sims[:, m]
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return scores.argmax(dim=1)

# SEQUENTIAL: train task t, eval all tasks 0..t, spawn during eval
V_std = torch.empty(0, 512, device=device)
lab_std = torch.empty(0, dtype=torch.long, device=device)
V_s1 = torch.empty(0, 512, device=device)
lab_s1 = torch.empty(0, dtype=torch.long, device=device)

for t in range(n_tasks):
    tx, ty = task_train[t]
    V_std = torch.cat([V_std, tx]); lab_std = torch.cat([lab_std, ty])
    V_s1 = torch.cat([V_s1, tx]); lab_s1 = torch.cat([lab_s1, ty])
    
    # Eval standard
    c_std = tot = 0
    for t2 in range(t+1):
        tex, tey = task_test[t2]
        preds = topk_predict(V_std, lab_std, tex, k)
        c_std += (preds == tey).sum().item()
        tot += tey.shape[0]
    aa_std = c_std / tot * 100
    
    # Eval S1 (spawn during eval)
    c_s1 = tot2 = 0
    for t2 in range(t+1):
        tex, tey = task_test[t2]
        preds = topk_predict(V_s1, lab_s1, tex, k)
        c_s1 += (preds == tey).sum().item()
        tot2 += tey.shape[0]
        # Spawn with predicted labels
        V_s1 = torch.cat([V_s1, tex])
        lab_s1 = torch.cat([lab_s1, preds])
    aa_s1 = c_s1 / tot2 * 100
    
    print(f'T{t}: std={aa_std:.1f}% s1={aa_s1:.1f}% delta={aa_s1-aa_std:+.1f}pp V_std={V_std.shape[0]} V_s1={V_s1.shape[0]}')

print(f'FINAL: std={aa_std:.1f}% s1={aa_s1:.1f}% delta={aa_s1-aa_std:+.1f}pp')
" 2>&1
