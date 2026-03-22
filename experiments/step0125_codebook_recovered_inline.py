# Step 125 — recovered from CC session (inline Bash)
cd B:/M/foldcore && cp B:/M/avir/research/fluxcore/RESEARCH_STATE.md ./RESEARCH_STATE.md && git add RESEARCH_STATE.md && git commit -m "$(cat <<'EOF'
Steps 125-127: k-NN fully characterized, feature discovery is frontier

Step 125: No class hierarchy in ResNet features. Two-level readout -12.9pp.
Step 126: k-NN solves XOR at d=2 (99.9%) but fails at d=50+ (noise dims).
  Adding oracle quadratic feature x0*x1: +10pp. The RIGHT feature works.
Step 127: Random quadratic expansion doesn't help — noise dilutes signal.
  Feature discovery (selecting WHICH features matter) is the unsolved problem.

30 experiments this session (97-127). System = k-NN. Feature discovery without
backprop is the frontier the atomic substrate must cross.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)" && git push origin master