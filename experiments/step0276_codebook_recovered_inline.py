# Step 276 — recovered from CC session 0606b161 (inline Bash execution)
# Original command (1731 chars):
# cd B:/M/foldcore && python -c "
import torch, torch.nn.functional as F
device = 'cuda'

X_fa=[]; y_s...

cd B:/M/foldcore && python -c "
import torch, torch.nn.functional as F
device = 'cuda'

X_fa=[]; y_s=[]; y_c=[]
for a in range(2):
    for b in range(2):
        for cin in range(2):
            s=a+b+cin
            for _ in range(50):
                X_fa.append([float(a),float(b),float(cin)]); y_s.append(s%2); y_c.append(s//2)
X_fa=torch.tensor(X_fa,device=device); y_s=torch.tensor(y_s,device=device,dtype=torch.long); y_c=torch.tensor(y_c,device=device,dtype=torch.long)

def fa(a,b,cin):
    q=torch.tensor([float(a),float(b),float(cin)],device=device)
    sims=F.normalize(q.unsqueeze(0),dim=1)@F.normalize(X_fa,dim=1).T
    return y_s[sims[0].topk(5).indices].mode().values.item(), y_c[sims[0].topk(5).indices].mode().values.item()

def add_bin(a,b,nb=20):
    ab=[(a>>i)&1 for i in range(nb)];bb=[(b>>i)&1 for i in range(nb)];carry=0;r=[]
    for i in range(nb):s,carry=fa(ab[i],bb[i],carry);r.append(s)
    return sum(bit*(2**i) for i,bit in enumerate(r+[carry]))

def mul_bin(a,b,nb=10):
    result=0;bb=[(b>>i)&1 for i in range(nb)]
    for i in range(nb):
        if bb[i]==1:result=add_bin(result,a*(2**i),2*nb)
    return result

def power(base, exp):
    result = 1
    for _ in range(exp):
        result = mul_bin(result, base)
    return result

print('Step 276: Exponentiation via repeated multiplication')
tests = [(2,8), (3,4), (5,3), (7,2), (2,10), (3,5)]
correct = 0
for b, e in tests:
    pred = power(b, e)
    true = b ** e
    ok = pred == true
    correct += int(ok)
    print(f'  {b}^{e} = {pred} (true: {true}) {\"OK\" if ok else \"FAIL\"}')
print(f'\\n  {correct}/{len(tests)} correct')
print(f'  Composition: truth table -> add -> mul -> power')
print(f'  SIX levels from ONE truth table')
" 2>&1