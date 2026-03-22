"""
Step 124 — S1 anti-forgetting: only real test samples help (noise/random useless)
Recovered from Leo session 0606b161 (line 9930).
All code was run inline (python -c) in the Leo session.

Context: k-NN era, Steps 97-165. Self-improving substrate investigation.
Session date: 2026-03-15.

Tests whether noisy copies or random vectors provide same anti-forgetting benefit as real eval samples.
Result: 38.2% AA without spawns → 39.7% with real eval spawns. Noisy copies 38.2%, random vectors 38.2%.
Confirms: S1 benefit is transductive memorization of exact test distribution.
"""
# Recovered inline code below:
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

# Four conditions: sequential eval with different types of spawning
conditions = {
    'no_spawn': lambda V, labs, tex, preds: (V, labs),
    's1_real': lambda V, labs, tex, preds: (torch.cat([V, tex]), torch.cat([labs, preds])),
    'noisy_copies': lambda V, labs, tex, preds: (
        torch.cat([V, F.normalize(tex + 0.1*torch.randn_like(tex), dim=1)]),
        torch.cat([labs, preds])
    ),
    'random_vecs': lambda V, labs, tex, preds: (
        torch.cat([V, F.normalize(torch.randn(tex.shape[0], 512, device=device), dim=1)]),
        torch.cat([labs, torch.randint(0, labs.max()+1, (tex.shape[0],), device=device)])
    ),
}

for name, spawn_fn in conditions.items():
    V = torch.empty(0, 512, device=device)
    labs = torch.empty(0, dtype=torch.long, device=device)
    
    for t in range(n_tasks):
        tx, ty = task_train[t]
        V = torch.cat([V, tx]); labs = torch.cat([labs, ty])
        
        c = tot = 0
        for t2 in range(t+1):
            tex, tey = task_test[t2]
            preds = topk_predict(V, labs, tex, k)
            c += (preds == tey).sum().item()
            tot += tey.shape[0]
            V, labs = spawn_fn(V, labs, tex, preds)
        aa = c / tot * 100
    
    print(f'{name:15s}: AA={aa:.1f}% CB={V.shape[0]}')
" 2>&1