"""
Step 724 (A5): Plain LSH k=12 baselines on all chain components.

R3 hypothesis: plain LSH is ℓ_0 (no refinement). R3 dynamics should be
near-zero for encoding (hash planes fixed), only edge_count_update changes.
Control baseline for Group B experiments.

4 environments x 5 min each = 20 min (Leo-authorized per batch spec).
Measure: L1 rate per game, CIFAR NMI, R3 dynamics per environment.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.plain_lsh import PlainLSH
from substrates.chain import SplitCIFAR100Wrapper
from substrates.judge import ConstitutionalJudge

print("=" * 65)
print("STEP 724 (A5) — PLAIN LSH BASELINES ON ALL CHAIN COMPONENTS")
print("=" * 65)

SEED = 0
PER_ENV_TIME = 280   # 5 min (with buffer)
N_STEPS_GAME = 10_000

judge = ConstitutionalJudge()


def _make_env(game):
    try:
        import arcagi3
        return arcagi3.make(game)
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game)


def run_game_plain(game_name, n_steps=N_STEPS_GAME, per_time=PER_ENV_TIME):
    """Run PlainLSH on a single ARC game, measure L1 + R3 dynamics."""
    print(f"\n-- {game_name} --")
    try:
        env = _make_env(game_name)
        n_valid = len(env._action_space) if hasattr(env, '_action_space') else 4
        print(f"  action_space: {n_valid}")

        sub = PlainLSH(n_actions=n_valid, seed=SEED)
        sub.reset(SEED)
        obs = env.reset(seed=SEED)
        level = 0
        l1_step = l2_step = None
        steps = 0
        fresh = True
        obs_seq = []
        t_start = time.time()

        while steps < n_steps and (time.time() - t_start) < per_time:
            if obs is None:
                obs = env.reset(seed=SEED)
                sub.on_level_transition()
                fresh = True
                continue

            obs_arr = np.array(obs, dtype=np.float32)
            obs_seq.append(obs_arr)
            action = sub.process(obs_arr)
            obs, reward, done, info = env.step(action % n_valid)
            steps += 1

            if fresh:
                fresh = False
                continue

            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                if cl == 1 and l1_step is None:
                    l1_step = steps
                if cl == 2 and l2_step is None:
                    l2_step = steps
                level = cl
                sub.on_level_transition()

            if done:
                obs = env.reset(seed=SEED)
                sub.on_level_transition()
                fresh = True

        elapsed = time.time() - t_start
        state = sub.get_state()
        print(f"  steps={steps} elapsed={elapsed:.1f}s l1={l1_step} l2={l2_step}")
        print(f"  G_size={state['G_size']} live={state['live_count']}")

        # R3 dynamics
        n_obs = min(len(obs_seq), 2000)
        class _plain(PlainLSH):
            def __init__(self): super().__init__(n_actions=n_valid, seed=0)

        r3 = judge.measure_r3_dynamics(
            _plain, obs_sequence=obs_seq[:n_obs],
            n_steps=n_obs, n_checkpoints=10
        )
        print(f"  R3 dynamic score: {r3.get('r3_dynamic_score')} "
              f"profile: {r3.get('dynamics_profile')}")
        print(f"  Verified M: {r3.get('verified_M_elements')}")

        return {
            "game": game_name, "steps": steps, "elapsed": elapsed,
            "l1": l1_step, "l2": l2_step,
            "G_size": state["G_size"], "live": state["live_count"],
            "r3_score": r3.get("r3_dynamic_score"),
            "r3_profile": r3.get("dynamics_profile"),
        }
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return {"game": game_name, "error": str(e)}


# --- Run all games ---
game_results = {}
for game in ["LS20", "FT09", "VC33"]:
    game_results[game] = run_game_plain(game)

# --- CIFAR ---
print("\n-- Split-CIFAR-100 --")
wrapper = SplitCIFAR100Wrapper(n_images_per_task=500)
cifar_result = None
if wrapper._load():
    sub_cifar = PlainLSH(n_actions=5, seed=SEED)
    cifar_result = wrapper.run_seed(sub_cifar, SEED)

    # NMI per task (rerun for metric)
    sub_nmi = PlainLSH(n_actions=5, seed=SEED)
    sub_nmi.reset(SEED)
    rng = np.random.RandomState(SEED)
    task_nmis = []
    for task_id in range(min(20, len(wrapper._data))):
        task_images, task_labels = wrapper._data[task_id]
        idx = rng.choice(len(task_images), min(500, len(task_images)), replace=False)
        preds = [sub_nmi.process(task_images[i].astype(np.float32)/255.0) % 5 for i in idx]
        true = [int(task_labels[i]) for i in idx]
        n = len(true)
        joint = np.zeros((5, 5))
        for t, p in zip(true, preds):
            joint[t][p] += 1
        joint /= max(n, 1)
        px = joint.sum(1); py = joint.sum(0)
        hx = -sum(float(p)*np.log2(float(p)+1e-15) for p in px if p > 1e-15)
        hy = -sum(float(p)*np.log2(float(p)+1e-15) for p in py if p > 1e-15)
        hxy = -sum(float(joint[i,j])*np.log2(float(joint[i,j])+1e-15)
                   for i in range(5) for j in range(5) if joint[i,j] > 1e-15)
        mi = hx + hy - hxy
        denom = (hx + hy) / 2
        task_nmis.append(mi / denom if denom > 1e-12 else 0.0)
        sub_nmi.on_level_transition()

    print(f"  avg_accuracy={cifar_result.get('avg_accuracy')} "
          f"bwt={cifar_result.get('backward_transfer')}")
    print(f"  avg_NMI={np.mean(task_nmis):.4f}")
else:
    print("  CIFAR not available")

# --- Summary ---
print("\n" + "=" * 65)
print("PLAIN LSH SUMMARY")
print("=" * 65)
print(f"  PlainLSH R3 static: M=1 I=2 U=2 (no ref_hyperplanes, no aliased_set)")
for game, r in game_results.items():
    if "error" not in r:
        print(f"  {game}: l1={r['l1']} R3_dyn={r.get('r3_score')} "
              f"profile={r.get('r3_profile')}")
if cifar_result:
    print(f"  CIFAR: acc={cifar_result.get('avg_accuracy')}")
print(f"\n  Compare to 674:")
print(f"    674 R3 dynamic score (step723): 2/3 or 3/3")
print(f"    PlainLSH R3 dynamic score: expected 1/1 (only edge_count_update M)")
print(f"    Delta = ℓ_π contribution to R3 dynamics")
print("=" * 65)
print("STEP 724 DONE")
print("=" * 65)
