"""
Step 1286b — Topological diagnostic: do KB-action windows produce distinct h-trajectories?

From Leo mail 3610 (2026-03-28): analyze whether h during KB sequences drifts differently
from h during click periods. Tests whether encoding captures action-type information.

Method: Run SP80 with KBI (forced KB every 50 steps) for 3000 steps.
Log h vector at every step + action type (KB=0-6, click=7+).
PCA the h vectors and report:
1. Mean distance between KB-window h-centroids and click-window h-centroids
2. Within-type vs between-type cosine distance ratio
3. Variance explained by PC1/PC2 separated by action type

Diagnostic, not experiment. One seed, 3000 steps, < 5 minutes.
No PRISM — SP80 only.

Spec: Leo mail 3610, 2026-03-28.
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM
ETA_H = 0.05
ETA_PRED = 0.01
PE_EMA_ALPHA = 0.05
SELECTION_ALPHA = 0.1
DECAY = 0.001

KB_INJECT_INTERVAL = 50
KB_INDEX_MAX = 7
N_STEPS = 3000
SEED = 42
GAME = 'sp80'

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1286')


def make_game(name):
    try:
        import arcagi3
        return arcagi3.make(name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(name.upper())


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng_kb = np.random.RandomState(SEED + 999)

    print("=" * 60)
    print("Step 1286b — Topological diagnostic: SP80 h-trajectory PCA")
    print("Q: do KB sequences drift in distinct encoding direction?")
    print("=" * 60)

    env = make_game(GAME)
    obs = env.reset(seed=SEED)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    # Build substrate (KBI condition: forced KB every 50 steps)
    rng_w = np.random.RandomState(SEED + 10000)
    running_mean = np.zeros(ENC_DIM, np.float32)
    n_obs_count = 0
    W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
    W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
    h = np.zeros(H_DIM, np.float32)
    scale = 1.0 / np.sqrt(float(EXT_DIM))
    W_action = rng_w.randn(n_actions, EXT_DIM).astype(np.float32) * scale
    action_counts = np.zeros(n_actions, np.float32)
    W_pred = rng_w.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.01
    pe_ema = np.zeros(n_actions, np.float32)

    # Trajectory logging
    h_log = []          # (step, h_vector, action, is_kb, level)
    enc_log = []        # (step, enc_vector, action, is_kb, level)

    step = 0
    level = 0
    prev_enc_flow = None
    prev_ext = None
    last_action = None
    fresh_episode = True
    t_start = time.time()

    while step < N_STEPS:
        if obs is None:
            obs = env.reset(seed=SEED)
            prev_enc_flow = None
            fresh_episode = True
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        x = _enc_frame(obs_arr)
        n_obs_count += 1
        a_rm = 1.0 / n_obs_count
        running_mean = (1 - a_rm) * running_mean + a_rm * x
        enc = x - running_mean
        h = np.tanh(W_h @ h + W_in @ enc)
        ext = np.concatenate([enc, h])

        # Forced KB injection or normal selection
        forced = (step > 0 and step % KB_INJECT_INTERVAL == 0)
        if forced:
            action = int(rng_kb.randint(0, KB_INDEX_MAX))
        else:
            score = action_counts - SELECTION_ALPHA * pe_ema
            action = int(np.argmin(score))

        is_kb = bool(action < KB_INDEX_MAX)
        h_log.append((step, h.copy(), action, is_kb, level))
        enc_log.append((step, enc.copy(), action, is_kb, level))

        action_counts[action] += 1
        obs_next, reward, done, info = env.step(action)
        step += 1

        # Update weights
        if obs_next is not None and not fresh_episode and prev_enc_flow is not None:
            enc_after = _enc_frame(np.asarray(obs_next, dtype=np.float32)) - running_mean
            pred_enc = W_pred @ prev_enc_flow
            pe = float(np.linalg.norm(enc_after - pred_enc))
            pred_error = enc_after - pred_enc
            W_pred += ETA_PRED * np.outer(pred_error, prev_enc_flow)
            pe_ema[action] = (1 - PE_EMA_ALPHA) * pe_ema[action] + PE_EMA_ALPHA * pe
            delta = enc_after - prev_enc_flow
            flow = float(np.linalg.norm(delta))
            W_action[action] += ETA_H * flow * ext
            W_action *= (1 - DECAY)

        prev_enc_flow = enc.copy()
        prev_ext = ext.copy()
        last_action = action

        if fresh_episode:
            fresh_episode = False
        else:
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl

        if done:
            obs = env.reset(seed=SEED)
            prev_enc_flow = None
            fresh_episode = True
            level = 0
        else:
            obs = obs_next

    elapsed = time.time() - t_start
    print(f"Simulation done: {step} steps in {elapsed:.1f}s. Max level: {level}")
    print(f"KB actions logged: {sum(1 for _, _, _, kb, _ in h_log if kb)}")
    print(f"Click actions logged: {sum(1 for _, _, _, kb, _ in h_log if not kb)}")
    print()

    # --- PCA Analysis ---
    h_matrix = np.array([h for _, h, _, _, _ in h_log], dtype=np.float32)
    kb_mask = np.array([kb for _, _, _, kb, _ in h_log], dtype=bool)
    click_mask = ~kb_mask

    # Center
    h_mean = h_matrix.mean(axis=0)
    h_centered = h_matrix - h_mean

    # PCA via SVD (top 10 components)
    n_comp = min(10, h_centered.shape[0], h_centered.shape[1])
    U, S, Vt = np.linalg.svd(h_centered, full_matrices=False)
    explained = S[:n_comp] ** 2 / (S ** 2).sum()

    h_projected = h_centered @ Vt[:n_comp].T  # (N_STEPS, n_comp)

    kb_proj = h_projected[kb_mask]
    click_proj = h_projected[click_mask]

    kb_centroid = kb_proj.mean(axis=0) if len(kb_proj) > 0 else np.zeros(n_comp)
    click_centroid = click_proj.mean(axis=0) if len(click_proj) > 0 else np.zeros(n_comp)

    centroid_dist = float(np.linalg.norm(kb_centroid - click_centroid))
    kb_var = float(kb_proj.var(axis=0).sum()) if len(kb_proj) > 1 else 0.0
    click_var = float(click_proj.var(axis=0).sum()) if len(click_proj) > 1 else 0.0
    total_var = float(h_projected.var(axis=0).sum())

    # Within-type vs between-type distance in full h-space
    rng_test = np.random.RandomState(0)
    kb_indices = np.where(kb_mask)[0]
    click_indices = np.where(click_mask)[0]
    n_sample = min(200, len(kb_indices), len(click_indices))

    within_kb = []
    within_click = []
    between = []
    for _ in range(200):
        if len(kb_indices) >= 2:
            i, j = rng_test.choice(kb_indices, 2, replace=False)
            na, nb = np.linalg.norm(h_matrix[i]), np.linalg.norm(h_matrix[j])
            if na > 1e-8 and nb > 1e-8:
                within_kb.append(1.0 - float(np.dot(h_matrix[i], h_matrix[j]) / (na * nb)))
        if len(click_indices) >= 2:
            i, j = rng_test.choice(click_indices, 2, replace=False)
            na, nb = np.linalg.norm(h_matrix[i]), np.linalg.norm(h_matrix[j])
            if na > 1e-8 and nb > 1e-8:
                within_click.append(1.0 - float(np.dot(h_matrix[i], h_matrix[j]) / (na * nb)))
        if len(kb_indices) > 0 and len(click_indices) > 0:
            i = rng_test.choice(kb_indices)
            j = rng_test.choice(click_indices)
            na, nb = np.linalg.norm(h_matrix[i]), np.linalg.norm(h_matrix[j])
            if na > 1e-8 and nb > 1e-8:
                between.append(1.0 - float(np.dot(h_matrix[i], h_matrix[j]) / (na * nb)))

    mean_within_kb = float(np.mean(within_kb)) if within_kb else None
    mean_within_click = float(np.mean(within_click)) if within_click else None
    mean_between = float(np.mean(between)) if between else None

    # PC1/PC2 loadings for KB vs click
    pc1_kb_mean = float(kb_proj[:, 0].mean()) if len(kb_proj) > 0 else None
    pc1_click_mean = float(click_proj[:, 0].mean()) if len(click_proj) > 0 else None
    pc2_kb_mean = float(kb_proj[:, 1].mean()) if n_comp > 1 and len(kb_proj) > 0 else None
    pc2_click_mean = float(click_proj[:, 1].mean()) if n_comp > 1 and len(click_proj) > 0 else None

    print("=== TOPOLOGICAL DIAGNOSTIC RESULTS ===")
    print(f"PCA variance explained (top 5): {[round(float(e), 4) for e in explained[:5]]}")
    print()
    print(f"Centroid distance (KB vs click, in PC space): {centroid_dist:.4f}")
    print(f"  KB centroid PC1/PC2: {pc1_kb_mean:.4f} / {pc2_kb_mean:.4f}" if pc1_kb_mean is not None else "  KB centroid: null")
    print(f"  Click centroid PC1/PC2: {pc1_click_mean:.4f} / {pc2_click_mean:.4f}" if pc1_click_mean is not None else "  Click centroid: null")
    print()
    print(f"Cosine distances:")
    print(f"  Within-KB: {mean_within_kb:.4f}" if mean_within_kb is not None else "  Within-KB: null")
    print(f"  Within-click: {mean_within_click:.4f}" if mean_within_click is not None else "  Within-click: null")
    print(f"  KB vs click (between): {mean_between:.4f}" if mean_between is not None else "  KB vs click: null")
    print()

    # Verdict
    if mean_between is not None and mean_within_kb is not None and mean_within_click is not None:
        sep_ratio = mean_between / max(0.5 * (mean_within_kb + mean_within_click), 1e-8)
        print(f"Separation ratio (between / mean-within): {sep_ratio:.3f}")
        if sep_ratio > 1.2:
            verdict = "SEPARABLE — KB and click h-trajectories occupy distinct encoding regions. Sequence discovery is topologically possible."
        elif sep_ratio > 1.05:
            verdict = "MARGINAL — slight separation. Sequence discovery possible but weak signal."
        else:
            verdict = "NOT SEPARABLE — KB and click h-trajectories overlap. Topological wall: encoding does not represent action type. Neither pairwise nor eigenoptions will help."
        print(f"\nVERDICT: {verdict}")
    else:
        verdict = "INSUFFICIENT DATA"
        sep_ratio = None

    print()

    result = {
        'game': GAME,
        'seed': SEED,
        'n_steps': step,
        'elapsed_seconds': round(elapsed, 2),
        'n_kb_steps': int(kb_mask.sum()),
        'n_click_steps': int(click_mask.sum()),
        'pca_variance_explained': [round(float(e), 6) for e in explained[:n_comp]],
        'centroid_distance_pc_space': round(centroid_dist, 6),
        'pc1_kb_mean': round(pc1_kb_mean, 6) if pc1_kb_mean is not None else None,
        'pc1_click_mean': round(pc1_click_mean, 6) if pc1_click_mean is not None else None,
        'pc2_kb_mean': round(pc2_kb_mean, 6) if pc2_kb_mean is not None else None,
        'pc2_click_mean': round(pc2_click_mean, 6) if pc2_click_mean is not None else None,
        'cosine_within_kb': round(mean_within_kb, 6) if mean_within_kb is not None else None,
        'cosine_within_click': round(mean_within_click, 6) if mean_within_click is not None else None,
        'cosine_between': round(mean_between, 6) if mean_between is not None else None,
        'separation_ratio': round(sep_ratio, 4) if sep_ratio is not None else None,
        'verdict': verdict,
    }

    out_path = os.path.join(RESULTS_DIR, 'step1286b_topo_diagnostic.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {out_path}")
    print("STEP 1286b DONE")


if __name__ == '__main__':
    main()
