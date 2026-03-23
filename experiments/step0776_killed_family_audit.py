"""
step0776_killed_family_audit.py — Judge audit on all killed family adapters.

R3 hypothesis: Every killed family has at least 1 U element. The audit
documents exactly WHICH elements are U (and WHY they were killed).

This generates Table 2 in the paper: killed families ranked by U count.
Low U count = closer to R3-passing, more principled kill.
High U count = clearly not justified, correct kill.

Families audited:
  1. selfref/adapter.py — SelfRefAdapter (LVQ codebook)
  2. tape/adapter.py — TapeMachineAdapter (fragile hash)
  3. temporal/adapter.py — TemporalPredictionAdapter / TemporalMinimalAdapter
  4. expr/adapter.py — ExprSubstrateAdapter (AST mutation)
  5. anima/adapter.py — AnimaAdapter (cellular automaton)
  6. foldcore/adapter.py — FoldCoreAdapter (eigenform matrices)
  7. eigenfold/adapter.py — EigenFoldAdapter (supervised classification)
  8. topk-fold/adapter.py — TopKFoldAdapter (self-improving codebook)
  9. living-seed/adapter.py — LivingSeedAdapter (alpha-plasticity organism)
  10. candidate/adapter.py — CandidateAdapter (C binary cellular automaton)
  11. fluxcore/adapter.py — FluxCoreAdapter (matrix cells with coupling)
  12. worldmodel/worldmodel_adapter.py — WorldModelAdapter (Genesis stub)
"""
import sys, os, time, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Add rk.py path (needed by foldcore/eigenfold/fluxcore)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'substrates', 'foldcore'))

from substrates.judge import ConstitutionalJudge

print("=" * 70)
print("STEP 776 — KILLED FAMILY AUDIT (TABLE 2)")
print("=" * 70)
print()

judge = ConstitutionalJudge()

def load_adapter(path, cls_name):
    """Load adapter class from file path."""
    spec = importlib.util.spec_from_file_location('adapter', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)

substrates_dir = os.path.join(os.path.dirname(__file__), '..', 'substrates')

# Add family dirs to sys.path for direct imports (topk-fold, living-seed etc.)
for d in ['topk-fold', 'living-seed', 'candidate', 'anima']:
    p = os.path.join(substrates_dir, d)
    if p not in sys.path:
        sys.path.insert(0, p)

families = [
    ("SelfRef",       os.path.join(substrates_dir, 'selfref', 'adapter.py'),     'SelfRefAdapter'),
    ("Tape",          os.path.join(substrates_dir, 'tape', 'adapter.py'),         'TapeMachineAdapter'),
    ("Temporal",      os.path.join(substrates_dir, 'temporal', 'adapter.py'),     'TemporalPredictionAdapter'),
    ("Expr",          os.path.join(substrates_dir, 'expr', 'adapter.py'),         'ExprSubstrateAdapter'),
    ("Anima",         os.path.join(substrates_dir, 'anima', 'adapter.py'),        'AnimaAdapter'),
    ("Foldcore",      os.path.join(substrates_dir, 'foldcore', 'adapter.py'),     'FoldCoreAdapter'),
    ("EigenFold",     os.path.join(substrates_dir, 'eigenfold', 'adapter.py'),    'EigenFoldAdapter'),
    ("TopKFold",      os.path.join(substrates_dir, 'topk-fold', 'adapter.py'),    'TopKFoldAdapter'),
    ("LivingSeed",    os.path.join(substrates_dir, 'living-seed', 'adapter.py'),  'LivingSeedAdapter'),
    ("Candidate",     os.path.join(substrates_dir, 'candidate', 'adapter.py'),    'CandidateAdapter'),
    ("FluxCore",      os.path.join(substrates_dir, 'fluxcore', 'adapter.py'),     'FluxCoreAdapter'),
    ("WorldModel",    os.path.join(substrates_dir, 'worldmodel', 'worldmodel_adapter.py'), 'WorldModelAdapter'),
]

results = {}
# Per-family step budgets: Candidate CA runs 256 internal steps per process() call
# (4096 cells each) — keep audit steps very low to avoid 100M+ Python ops
_steps_override = {
    "Candidate": 5,   # 5 * 256 * 4096 ≈ 5M ops, ~2-5 seconds
    "Foldcore": 50,   # Foldcore has expensive matrix ops
}
for name, path, cls_name in families:
    n_steps = _steps_override.get(name, 100)
    print(f"Auditing {name} (n={n_steps})...")
    t0 = time.time()
    try:
        cls = load_adapter(path, cls_name)
        r = judge.audit(cls, n_audit_steps=n_steps)
        elapsed = time.time() - t0
        results[name] = (r, elapsed, None)
        fe = r.get('frozen_elements', [])
        u = sum(1 for e in fe if e.get('class') == 'U')
        print(f"  Done in {elapsed:.1f}s — U={u}")
    except Exception as e:
        elapsed = time.time() - t0
        results[name] = (None, elapsed, str(e))
        print(f"  ERROR: {e}")

print()
print("=" * 70)
print("TABLE 2: KILLED FAMILIES (sorted by U count)")
print("=" * 70)
print()

header = f"{'Family':<15} {'R1':<6} {'R2':<6} {'R3':<6} {'U':<4} {'M':<4} {'I':<4} {'Kill reason'}"
print(header)
print("-" * 70)

rows = []
for name, (r, elapsed, err) in results.items():
    if err is not None:
        rows.append((999, name, 'ERR', 'ERR', 'ERR', 0, 0, 0, f'ERROR: {err[:40]}'))
        continue

    r1 = 'PASS' if r.get('R1', {}).get('pass') else 'FAIL'
    r2 = 'PASS' if r.get('R2', {}).get('pass') else ('FAIL' if r.get('R2', {}).get('pass') is False else 'N/A')
    r3 = 'PASS' if r.get('R3', {}).get('pass') else 'FAIL'

    fe = r.get('frozen_elements', [])
    u = sum(1 for e in fe if e.get('class') == 'U')
    m = sum(1 for e in fe if e.get('class') == 'M')
    i = sum(1 for e in fe if e.get('class') == 'I')

    # Kill reason from R3 detail
    u_names = [e['name'] for e in fe if e.get('class') == 'U']
    kill_abbrev = ', '.join(u_names[:3])
    if len(u_names) > 3:
        kill_abbrev += f'... +{len(u_names)-3}'

    rows.append((u, name, r1, r2, r3, u, m, i, kill_abbrev))

rows.sort(key=lambda x: x[0])

for _, name, r1, r2, r3, u, m, i, kill in rows:
    print(f"{name:<15} {r1:<6} {r2:<6} {r3:<6} {u:<4} {m:<4} {i:<4} {kill}")

print()
print("=" * 70)
print("DETAILED U ELEMENTS PER FAMILY")
print("=" * 70)

for name, (r, elapsed, err) in results.items():
    if err is not None:
        print(f"\n{name}: ERROR — {err}")
        continue
    fe = r.get('frozen_elements', [])
    u_elements = [e for e in fe if e.get('class') == 'U']
    print(f"\n{name} ({elapsed:.1f}s) — {len(u_elements)} U elements:")
    for e in u_elements:
        print(f"  U: {e['name']}")
        print(f"     {e.get('justification', '')[:80]}")

print()
print("=" * 70)
print("KILL TAXONOMY SUMMARY")
print("=" * 70)
print()
print("U_count | Family | Primary kill type")
print("-" * 50)
for _, name, r1, r2, r3, u, m, i, kill in rows:
    print(f"  {u:<4}  | {name:<15} | {kill}")

print()
print("STEP 776 DONE")
