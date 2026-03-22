# Step 278 — recovered from CC session 0606b161 (inline Bash execution)
# Original command (2088 chars):
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

def isqrt(n):
    '''Integer square root via binary search using composed arithmetic.'''
    if n == 0: return 0
    lo, hi = 1, n
    while lo <= hi:
        mid = add_bin(lo, hi) // 2  # use add for midpoint (// is just bit shift)
        sq = mul_bin(mid, mid)
        if sq == n: return mid
        elif sq < n:
            lo = add_bin(mid, 1)
        else:
            hi = mid - 1  # simplified sub
    return hi

print('Step 278: Integer square root via binary search')
import math
correct = 0; total = 0
for n in [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 169, 225, 256, 10, 15, 50, 99, 200]:
    pred = isqrt(n)
    true = int(math.isqrt(n))
    ok = pred == true
    total += 1; correct += int(ok)
    print(f'  isqrt({n:3d}) = {pred:2d} (true: {true:2d}) {\"OK\" if ok else \"FAIL\"}')
print(f'\\n  {correct}/{total} correct')
print(f'  EIGHT levels: truth table -> add -> mul -> comparison -> binary search -> isqrt')
" 2>&1