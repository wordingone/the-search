# Step 122 — recovered from CC session (inline Bash)
cd B:/M/foldcore && cp B:/M/avir/research/fluxcore/RESEARCH_STATE.md ./RESEARCH_STATE.md && git add RESEARCH_STATE.md && git commit -m "$(cat <<'EOF'
Step 122: Confidence gating doesn't help — density matters, not labels

Spawning ALL eval samples (including 62% mislabeled) gives best result.
More spawns = better, regardless of label accuracy. Anti-forgetting
works through geometric DENSITY, not discriminative correctness.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)" && git push origin master