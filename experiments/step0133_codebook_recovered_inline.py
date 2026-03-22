# Step 133 — recovered from CC session (inline Bash)
cd B:/M/foldcore && cp B:/M/avir/research/fluxcore/RESEARCH_STATE.md ./RESEARCH_STATE.md && git add RESEARCH_STATE.md && git commit -m "$(cat <<'EOF'
Step 133: Coherence discovery no-op on MNIST (already coherent)

Raw MNIST pixels already have high class coherence — no quadratic
feature improves it. Mechanism works where raw features are weak
(XOR, d=20-100) but not where they're strong (MNIST, 93.4% baseline).
Scalability wall: d=3072 has 4.7M pairs, random sampling insufficient.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)" && git push origin master