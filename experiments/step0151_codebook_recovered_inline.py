# Step 151 — recovered from CC session (inline Bash)
cd B:/M/foldcore && git add substrates/topk-fold/self_improving_substrate.py && git add RESEARCH_STATE.md && git commit -m "$(cat <<'EOF'
Step 151: Self-improving substrate implementation — 100% on parity

Clean implementation of the discovered mechanism:
- Store all exemplars (always-spawn)
- Margin-guided feature discovery (products + aggregations)
- Top-k(5) per-class cosine vote
Discovers cos_sum_pi + sum_mod2 = parity. 75% -> 100%.
54 experiments (97-151).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)" && git push origin master