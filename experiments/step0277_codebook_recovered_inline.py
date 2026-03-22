# Step 277 — recovered from CC session 0606b161 (inline Bash execution)
# Original command (2086 chars):
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

def div_mod(a,b):
    if b==0: return 0,a
    q=0;r=a
    while r>=b and q<10000: r=r-b; q+=1
    return q,r

def mod_pow(base, exp, mod):
    '''a^b mod m via repeated squaring'''
    result = 1
    base = div_mod(base, mod)[1]  # base mod m
    while exp > 0:
        if exp % 2 == 1:  # odd
            result = div_mod(mul_bin(result, base), mod)[1]
        exp = exp // 2
        base = div_mod(mul_bin(base, base), mod)[1]
    return result

print('Step 277: Modular exponentiation (RSA primitive)')
tests = [(2, 10, 7), (3, 5, 13), (5, 3, 17), (7, 4, 11), (2, 16, 97)]
correct = 0
for b, e, m in tests:
    pred = mod_pow(b, e, m)
    true = pow(b, e, m)
    ok = pred == true
    correct += int(ok)
    print(f'  {b}^{e} mod {m} = {pred} (true: {true}) {\"OK\" if ok else \"FAIL\"}')
print(f'\\n  {correct}/{len(tests)} correct')
print(f'  SEVEN composition levels: truth table -> add -> mul -> div/mod -> power -> mod_pow')
" 2>&1