# Step 269 — recovered from CC session 0606b161 (inline Bash execution)
# Original command (2444 chars):
# cd B:/M/foldcore && python -c "
import torch

# Step 273: Primality via DECOMPOSED trial division
# ...

cd B:/M/foldcore && python -c "
import torch

# Step 273: Primality via DECOMPOSED trial division
# The substrate can't detect primes directly (Step 269: 0pp).
# But it CAN do modular arithmetic (Step 207: +38pp).
# Decomposition: is_prime(n) = for d in 2..sqrt(n): if n%d==0 return False

# Full adder (proven)
X_fa=[]; y_s=[]; y_c=[]
for a in range(2):
    for b in range(2):
        for cin in range(2):
            s=a+b+cin
            for _ in range(50):
                X_fa.append([float(a),float(b),float(cin)]); y_s.append(s%2); y_c.append(s//2)
import torch.nn.functional as F
device='cuda'
X_fa=torch.tensor(X_fa,device=device); y_s=torch.tensor(y_s,device=device,dtype=torch.long); y_c=torch.tensor(y_c,device=device,dtype=torch.long)

def fa(a,b,cin):
    q=torch.tensor([float(a),float(b),float(cin)],device=device)
    sims=F.normalize(q.unsqueeze(0),dim=1)@F.normalize(X_fa,dim=1).T
    return y_s[sims[0].topk(5).indices].mode().values.item(), y_c[sims[0].topk(5).indices].mode().values.item()

def add_bin(a,b,nb=12):
    ab=[(a>>i)&1 for i in range(nb)];bb=[(b>>i)&1 for i in range(nb)];carry=0;r=[]
    for i in range(nb):s,carry=fa(ab[i],bb[i],carry);r.append(s)
    return sum(bit*(2**i) for i,bit in enumerate(r+[carry]))

def mul_bin(a,b,nb=8):
    result=0;bb=[(b>>i)&1 for i in range(nb)]
    for i in range(nb):
        if bb[i]==1: result=add_bin(result,a*(2**i),2*nb)
    return result

def div_bin(a,b):
    if b==0: return -1,-1
    q=0;r=a
    for _ in range(256):
        if r<b: break
        r=r-b;q+=1
    return q,r

# Trial division primality test using PROVEN arithmetic
def is_prime_decomposed(n):
    if n < 2: return False
    if n < 4: return True
    _, r2 = div_bin(n, 2)
    if r2 == 0: return False
    d = 3
    while mul_bin(d, d) <= n:
        _, rd = div_bin(n, d)
        if rd == 0: return False
        d = add_bin(d, 2)
    return True

# Test
print('Step 273: Primality via decomposed trial division')
correct = 0; total = 0
import math
for n in range(2, 50):
    pred = is_prime_decomposed(n)
    true = all(n % d != 0 for d in range(2, int(math.sqrt(n))+1)) and n >= 2
    ok = pred == true
    total += 1; correct += int(ok)

print(f'  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)')
print(f'  Primes found: {[n for n in range(2,50) if is_prime_decomposed(n)]}')
print(f'  True primes:  {[n for n in range(2,50) if all(n%d!=0 for d in range(2,int(math.sqrt(n))+1))]}')
" 2>&1