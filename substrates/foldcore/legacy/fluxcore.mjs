/**
 * FluxCore — Final Corrected Algorithm + Four-Test Suite
 */

let _rng=42;
const rand =()=>{_rng=(Math.imul(1664525,_rng)+1013904223)>>>0;return(_rng>>>0)/4294967296;};
const randN=()=>{const u=rand()+1e-12,v=rand();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
const normalize=v=>{let s=0;for(let i=0;i<v.length;i++)s+=v[i]*v[i];const n=Math.sqrt(s)+1e-12;for(let i=0;i<v.length;i++)v[i]/=n;return v;};
const randUnit =d=>{const v=new Float64Array(d);for(let i=0;i<d;i++)v[i]=randN();return normalize(v);};
const dot      =(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;};
const l1       =(a,b)=>{let s=0;for(let i=0;i<a.length;i++)s+=Math.abs(a[i]-b[i]);return s/a.length;};
const clone    =v=>new Float64Array(v);
const avg      =a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:0;
const std      =a=>{const m=avg(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/(a.length||1));};
const tail     =(a,f)=>a.slice(Math.max(0,Math.floor(a.length*(1-f))));
const head     =(a,f)=>a.slice(0,Math.max(1,Math.floor(a.length*f)));

function makeOrtho(base,dim){
  const r=randUnit(dim),p=dot(r,base),b=new Float64Array(dim);
  for(let i=0;i<dim;i++) b[i]=r[i]-p*base[i];return normalize(b);
}
const makeStatic  =(base,n)=>()=>{const o=new Float64Array(base.length);for(let i=0;i<o.length;i++)o[i]=base[i]+n*randN();return normalize(o);};
const makeRotating=(A,P,d,n)=>t=>{const c=Math.cos(t*d),s=Math.sin(t*d),o=new Float64Array(A.length);for(let i=0;i<o.length;i++)o[i]=c*A[i]+s*P[i]+n*randN();return normalize(o);};
const makeNoise   =d=>()=>randUnit(d);

// ─── FLUXCORE FOLD ────────────────────────────────────────────────────────────
function fluxFold(self, memory, velocity, prevSelf, reality, baseLr, memLr, k, memW, velDecay, velGain){
  const dim=self.length;
  let gd=0;for(let i=0;i<dim;i++) gd+=Math.abs(self[i]-reality[i]);gd/=dim;
  const alr=baseLr*(1+k*gd);
  for(let idx=0;idx<dim;idx++){
    const s=self[idx],r=reality[idx],m=memory[idx];
    const d=Math.abs(s-r);
    const left=self[(idx+dim-1)%dim],right=self[(idx+1)%dim];
    const grad=(s-left)-(s-right);
    const u=s+alr*r+(alr*0.5)*d*grad+memW*m;
    self[idx]=u;
    memory[idx]=m+memLr*u;
  }
  normalize(self); normalize(memory);
  // velocity: unnormalized, keeps natural magnitude ≈ per-tick displacement
  for(let i=0;i<dim;i++) velocity[i]=velDecay*velocity[i]+velGain*(self[i]-prevSelf[i]);
  return gd;
}

function makeOutput(self, velocity, velScale, dim){
  const o=new Float64Array(dim);
  for(let i=0;i<dim;i++) o[i]=self[i]+velScale*velocity[i];
  return normalize(o);
}

// ════════════════════════════════════════════════════════════════════════════
// T1 — ANTICIPATION
// ════════════════════════════════════════════════════════════════════════════
function test1(dim,noise,delta,lr,warmup,measure){
  process.stdout.write(`\n─── T1: Anticipation  DIM=${dim} delta=${delta}rad/tick lr=${lr} ───\n`);
  const lag=(1-lr)/lr;
  const velScale=lag+1;
  const A=randUnit(dim),P=makeOrtho(A,dim),gen=makeRotating(A,P,delta,noise);
  const self=randUnit(dim),mem=randUnit(dim),vel=new Float64Array(dim);
  let prev=clone(self);
  for(let t=0;t<warmup;t++){
    const sn=clone(self);fluxFold(self,mem,vel,sn,gen(t),lr,0.005,0,0.05,0.95,0.05);prev=sn;
  }
  const gains=[];
  for(let t=warmup;t<warmup+measure;t++){
    const rNow=gen(t),rNext=gen(t+1);
    const sn=clone(self);
    fluxFold(self,mem,vel,sn,rNow,lr,0.005,0,0.05,0.95,0.05);
    const output=makeOutput(self,vel,velScale,dim);
    gains.push(l1(self,rNext)-l1(output,rNext));
    prev=sn;
  }
  const g=avg(tail(gains,0.3));
  const pass=g>0;
  process.stdout.write(`  lag=${lag.toFixed(1)} velScale=${velScale.toFixed(1)}\n`);
  process.stdout.write(`  avg gain (last 30%) = ${g.toFixed(8)}  → ${pass?'PASS ✓':'FAIL ✗'}\n`);
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// T2 — ACCELERATED REACQUISITION
// ════════════════════════════════════════════════════════════════════════════
function test2(dim,noise,lr,warmup,A1len,Blen,A2len,earlyW){
  process.stdout.write(`\n─── T2: Accelerated Reacquisition  DIM=${dim} noise=${noise} ───\n`);
  const bA=randUnit(dim),bB=makeOrtho(bA,dim);
  const gA=makeStatic(bA,noise),gB=makeStatic(bB,noise),gN=makeNoise(dim);
  const self=randUnit(dim),mem=randUnit(dim),vel=new Float64Array(dim);let prev=clone(self);
  for(let t=0;t<warmup;t++){const sn=clone(self);fluxFold(self,mem,vel,sn,gN(),lr,0.015,20,0.15,0.95,0.05);prev=sn;}
  const ea1=[],ea2=[];
  for(let t=0;t<A1len;t++){const r=gA(),sn=clone(self);fluxFold(self,mem,vel,sn,r,lr,0.015,20,0.15,0.95,0.05);prev=sn;if(t<earlyW)ea1.push(l1(self,r));}
  for(let t=0;t<Blen;t++){const sn=clone(self);fluxFold(self,mem,vel,sn,gB(),lr,0.015,20,0.15,0.95,0.05);prev=sn;}
  for(let t=0;t<A2len;t++){const r=gA(),sn=clone(self);fluxFold(self,mem,vel,sn,r,lr,0.015,20,0.15,0.95,0.05);prev=sn;if(t<earlyW)ea2.push(l1(self,r));}
  const m1=avg(ea1),m2=avg(ea2),pass=m2<m1;
  process.stdout.write(`  early A1 (${earlyW} ticks) = ${m1.toFixed(7)}\n`);
  process.stdout.write(`  early A2 (${earlyW} ticks) = ${m2.toFixed(7)}\n`);
  process.stdout.write(`  → ${pass?'PASS ✓':'FAIL ✗'}\n`);
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// T3 — DISTRIBUTIONAL SHIFT DISCRIMINATION
// ════════════════════════════════════════════════════════════════════════════
function test3(dim,noise,lr,warmup,stLen,abLen){
  process.stdout.write(`\n─── T3: Distributional Shift  DIM=${dim} noise=${noise} ───\n`);
  const A=randUnit(dim),B=makeOrtho(A,dim);
  const gA=makeStatic(A,noise),gB=makeStatic(B,noise);
  const self=randUnit(dim),mem=randUnit(dim),vel=new Float64Array(dim);let prev=clone(self);
  for(let t=0;t<warmup;t++){const sn=clone(self);fluxFold(self,mem,vel,sn,gA(),lr,0.01,20,0.1,0.95,0.05);prev=sn;}
  const sL=[],spL=[];
  for(let t=0;t<stLen;t++){const r=gA(),sn=clone(self);fluxFold(self,mem,vel,sn,r,lr,0.01,20,0.1,0.95,0.05);prev=sn;sL.push(l1(self,r));}
  for(let t=0;t<abLen;t++){const r=gB(),sn=clone(self);fluxFold(self,mem,vel,sn,r,lr,0.01,20,0.1,0.95,0.05);prev=sn;spL.push(l1(self,r));}
  const sm=avg(sL),ss=std(sL),spk=Math.max(...spL.slice(0,10));
  const sigma=(spk-sm)/(ss+1e-12),late=avg(tail(spL,0.4)),pass=sigma>=2&&late<spk;
  process.stdout.write(`  stable mean=${sm.toFixed(7)} std=${ss.toFixed(7)}\n`);
  process.stdout.write(`  spike=${spk.toFixed(7)} = ${sigma.toFixed(1)}σ  late=${late.toFixed(7)}\n`);
  process.stdout.write(`  → ${pass?'PASS ✓':'FAIL ✗'}\n`);
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
// T4 — ADAPTIVE LR VALUE
// ════════════════════════════════════════════════════════════════════════════
function test4(dim,noise,lr,warmup,stLen,abLen){
  process.stdout.write(`\n─── T4: Adaptive lr vs Fixed lr  DIM=${dim} ───\n`);
  const A=randUnit(dim),B=makeOrtho(A,dim);
  const gA=makeStatic(A,noise),gB=makeStatic(B,noise);
  const aS=randUnit(dim),aM=randUnit(dim),aV=new Float64Array(dim);
  const fS=clone(aS),fM=clone(aM),fV=new Float64Array(dim);
  for(let t=0;t<warmup+stLen;t++){
    const r=gA();
    const sa=clone(aS);fluxFold(aS,aM,aV,sa,r,lr,0.01,20,0.1,0.95,0.05);
    const sf=clone(fS);fluxFold(fS,fM,fV,sf,r,lr,0.01, 0,0.1,0.95,0.05);
  }
  const aL=[],fL=[];
  for(let t=0;t<abLen;t++){
    const r=gB();
    const sa=clone(aS);fluxFold(aS,aM,aV,sa,r,lr,0.01,20,0.1,0.95,0.05);aL.push(l1(aS,r));
    const sf=clone(fS);fluxFold(fS,fM,fV,sf,r,lr,0.01, 0,0.1,0.95,0.05);fL.push(l1(fS,r));
  }
  const q1a=avg(head(aL,0.25)),q1f=avg(head(fL,0.25));
  const cA=avg(aL),cF=avg(fL);
  const pass=q1a<q1f||cA<cF;
  process.stdout.write(`  Q1 cumL1: adaptive=${q1a.toFixed(7)}  fixed=${q1f.toFixed(7)}  Q1-wins=${q1a<q1f}\n`);
  process.stdout.write(`  Full cumL1: adaptive=${cA.toFixed(7)}  fixed=${cF.toFixed(7)}  full-wins=${cA<cF}\n`);
  process.stdout.write(`  → ${pass?'PASS ✓':'FAIL ✗'}\n`);
  return pass;
}

// ════════════════════════════════════════════════════════════════════════════
console.log('═'.repeat(68));
console.log('FluxCore — Final Verification Suite');
console.log('═'.repeat(68));

const results=[];
for(const [dim,noise,delta,lr,earlyW] of [
  [64,   0.0003, 0.012, 0.08, 50],
  [512,  0.0003, 0.005, 0.08, 50],
  [8192, 0.0002, 0.002, 0.08, 50],
]){
  console.log(`\n${'━'.repeat(68)}\nSCALE: DIM=${dim}\n${'━'.repeat(68)}`);
  const t1=test1(dim,noise,delta,lr,1500,3000);
  const t2=test2(dim,noise,lr,500,6000,2000,6000,earlyW);
  const t3=test3(dim,noise,lr,500,3000,600);
  const t4=test4(dim,noise,lr,500,3000,400);
  results.push({dim,t1,t2,t3,t4,passes:[t1,t2,t3,t4].filter(Boolean).length});
}

console.log(`\n${'═'.repeat(68)}`);
console.log('FINAL RESULTS');
console.log('═'.repeat(68));
for(const r of results){
  const f=[r.t1,r.t2,r.t3,r.t4].map((v,i)=>v?`T${i+1}✓`:`T${i+1}✗`).join('  ');
  console.log(`  DIM=${String(r.dim).padEnd(6)}  ${r.passes}/4  [${f}]`);
}
const all=results.every(r=>r.passes===4);
console.log(`\n  All 4/4 at all scales: ${all?'YES ✓':'PARTIAL'}`);
console.log('═'.repeat(68));
