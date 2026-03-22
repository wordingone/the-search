# Step 165 — recovered from CC session 0606b161 (inline Bash execution)
# Original command (2501 chars):
# cd B:/M/foldcore && python -c "
import torch, torch.nn.functional as F
device = 'cuda'
from torchvis...

cd B:/M/foldcore && python -c "
import torch, torch.nn.functional as F
device = 'cuda'
from torchvision import datasets

mnist = datasets.MNIST('C:/tmp/mnist', train=True, download=False)
mnist_test = datasets.MNIST('C:/tmp/mnist', train=False)

d = 784; n_train = 6000; n_test = 1000; k = 5

X_tr = F.normalize(mnist.data[:n_train].float().view(-1, 784).to(device), dim=1)
y_tr = mnist.targets[:n_train].to(device)
X_te = F.normalize(mnist_test.data[:n_test].float().view(-1, 784).to(device), dim=1)
y_te = mnist_test.targets[:n_test].to(device)

def knn_acc(V, labels, te, y_te, k=5):
    sims = te @ V.T
    n_cls = labels.max().item() + 1
    scores = torch.zeros(te.shape[0], n_cls, device=device)
    for c in range(n_cls):
        m = labels == c; cs = sims[:, m]
        if cs.shape[1] == 0: continue
        scores[:, c] = cs.topk(min(k, cs.shape[1]), dim=1).values.sum(dim=1)
    return (scores.argmax(1) == y_te).float().mean().item() * 100

# Iterative random feature addition
V_tr = X_tr.clone()
V_te = X_te.clone()
discovered_features = []

print('Step 165: Iterative random cosine feature addition on MNIST')
acc = knn_acc(V_tr, y_tr, V_te, y_te)
print(f'Round 0: {acc:.1f}% (base)')

for round in range(10):
    best_w = None; best_b = None; best_acc = knn_acc(F.normalize(V_tr,dim=1), y_tr, F.normalize(V_te,dim=1), y_te)
    
    for _ in range(100):  # 100 candidates per round
        w = torch.randn(d, device=device) * 0.1
        b = torch.rand(1, device=device) * 6.28
        feat_tr = torch.cos(X_tr @ w + b).unsqueeze(1)
        feat_te = torch.cos(X_te @ w + b).unsqueeze(1)
        aug_tr = F.normalize(torch.cat([V_tr, feat_tr], 1), dim=1)
        aug_te = F.normalize(torch.cat([V_te, feat_te], 1), dim=1)
        acc_trial = knn_acc(aug_tr, y_tr, aug_te, y_te)
        if acc_trial > best_acc:
            best_acc = acc_trial; best_w = w.clone(); best_b = b.clone()
    
    if best_w is None:
        print(f'Round {round+1}: no improvement')
        break
    
    feat_tr = torch.cos(X_tr @ best_w + best_b).unsqueeze(1)
    feat_te = torch.cos(X_te @ best_w + best_b).unsqueeze(1)
    V_tr = torch.cat([V_tr, feat_tr], 1)
    V_te = torch.cat([V_te, feat_te], 1)
    discovered_features.append((best_w, best_b))
    
    acc = knn_acc(F.normalize(V_tr,dim=1), y_tr, F.normalize(V_te,dim=1), y_te)
    print(f'Round {round+1}: {acc:.1f}% (d={V_tr.shape[1]})')

print(f'\\nFinal: {acc:.1f}% with {len(discovered_features)} features added (d={V_tr.shape[1]})')
" 2>&1