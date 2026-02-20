# Codex Re-Audit: Verify Fixes + Find Remaining Bugs

## Context

A previous audit found 8 bugs. A developer claimed to fix all 8. **Your job: verify the fixes are correct AND find any bugs the previous audit missed.**

The developer has a track record of sloppy work — off-by-one errors, wrong ordering, features that look fixed but aren't. **Do not trust anything. Read every line. Verify everything.**

Files: `molequla.go`, `molequla.py`, `molequla.c`, `molequla.js`, `molequla.rs`, `README.md`

Read ALL files completely before writing your report.

---

## PART 1: Verify Previous 8 Fixes

### Fix 1: Go self-enrichment order
**Bug was:** `cooccur.IngestTokens(user_input)` ran BEFORE `cooccur.BuildFromCorpus()`, getting wiped.
**Claimed fix:** Moved ingest AFTER BuildFromCorpus.
**Verify in molequla.go:** Find the REPL loop. Confirm `BuildFromCorpus` runs BEFORE `IngestTokens(userText)`. Check that the answer's `IngestTokens` also runs AFTER build (not before).

### Fix 2: Go per-token sigmoid corpus fade
**Bug was:** `modelAlpha` was computed once per message in REPL, passed as constant to `GenerateResonant`.
**Claimed fix:** Per-token alpha computed inside `GenerateResonant` from local entropy.
**Verify in molequla.go:** Inside `GenerateResonant`'s token loop, find where `tokenAlpha` (or equivalent) is computed. Confirm it uses `1/(1+exp(-k*(threshold-entropy)))` where `entropy` is the per-token entropy computed in the same iteration. Confirm the old REPL-level `ComputeModelEntropy` + message-level alpha is gone.

### Fix 3: C head_types array capacity
**Bug was:** `head_types[4]` couldn't hold 5 or 8 entries.
**Claimed fix:** Changed to `head_types[8]`.
**Verify in molequla.c:** Find the Config struct. Confirm `head_types` array has >= 8 slots. Confirm `head_types_for_n_head()` no longer caps at 4 and correctly fills all N entries for N=5 and N=8.

### Fix 4: head_types_for_n_head(5) — all 5 files
**Bug was:** `n/2` floor division gave 2 content + 3 hybrid for n=5.
**Claimed fix:** Ceiling division gives 3 content + 2 hybrid.
**Verify in ALL 5 files:** Find `head_types_for_n_head` (or equivalent). For EACH file, mentally compute the result for n=1, n=2, n=4, n=5, n=8:
- n=1 → [content]
- n=2 → [content, hybrid]
- n=4 → [content, content, hybrid, hybrid]
- n=5 → [content, content, content, hybrid, hybrid]
- n=8 → [content, content, content, content, hybrid, hybrid, hybrid, hybrid]

If ANY file gives different results, report it.

### Fix 5: Init warmup scaling — Go, Python, C, Rust
**Bug was:** Init path used raw `warmup_steps` instead of `warmup_steps * (n_embd / embryo_embd)`.
**Claimed fix:** Added scaling.
**Verify in Go, Python, C, Rust init paths:** Find the cold-start per-stage warmup loop. Confirm it computes `effective_warmup = warmup_steps * (current_embd / embryo_embd)` and passes `effective_warmup` to `train_steps`. JS was already correct — verify JS still works too.

### Fix 6: Go background trainer warmup scaling
**Bug was:** Background trainer used raw `WarmupSteps`.
**Claimed fix:** Added scaling like other files.
**Verify in molequla.go:** Find the background trainer's per-stage warmup section. Confirm it computes `effectiveWarmup` with embd scaling.

### Fix 7: Rust fallback BPE
**Bug was:** If checkpoint load fails, fallback path didn't call `maybe_enable_bpe`.
**Claimed fix:** Added `tok.maybe_enable_bpe(&docs, &cfg)` in fallback.
**Verify in molequla.rs:** Find the checkpoint load section. Confirm BOTH paths (fresh start AND checkpoint-load-failure fallback) call `maybe_enable_bpe` before creating GPT.

### Fix 8: JS config consistency
**Bug was:** 7 JS config values differed from Go/Python/C/Rust.
**Claimed fix:** Aligned all values.
**Verify:** Compare these specific values across ALL 5 files (report a table):

| Config Key | Go | Python | C | JS | Rust | README |
|---|---|---|---|---|---|---|
| qb_min_bytes | | | | | | 1024 |
| qb_cooldown_seconds | | | | | | 60.0 |
| corpus_gen_max_tokens | | | | | | 120 |
| syntropy_window | | | | | | 8 |
| field_deviation_floor | | | | | | 0.1 |
| field_deviation_ceiling | | | | | | 12.0 |
| syntropy_delta_grow_boost | | | | | | 0.15 |

Fill in the actual values from each file. Flag any mismatch.

---

## PART 2: Full Re-Audit (Find NEW Bugs)

Now do a complete audit. Check everything below across ALL 5 files.

### A. Growth Stages
6 stages, exact values:
```
(0,16,1,1) (20000,32,1,2) (50000,64,2,4) (200000,128,4,4) (350000,224,5,8) (500000,320,6,8)
```
Stage name arrays must have 6 entries including "teen".

### B. Config Parity
Compare EVERY config default across all 5 files. Report ANY difference. Key values to check (beyond the 7 from Fix 8):
- freq_penalty, presence_penalty (0.1)
- temperature (0.85), top_k (40), top_p (0.92), min_p (0.06), typical_p (0.95)
- warmup_steps (1200), micro_steps (32), learning_rate (0.01)
- block_size (96), grad_clip (1.0), batch_size (4)
- delta_rank (8), max_delta_modules (12)
- bpe_num_merges (384), enable_bpe_after_chars (20000)
- max_gen_tokens (180), min_gen_tokens (16), repetition_guard (4)
- noise_drift_threshold (-0.1)
- All consciousness params (dissonance, anti_field, overthinkc, conscience)

### C. Self-Enrichment
In EACH file's REPL/chat loop:
1. User input → ingest into corpus field (AFTER any rebuild)
2. Own output → ingest into corpus field (AFTER generation)
Both must exist. Check ordering relative to any corpus rebuild.

### D. BPE Before Warmup
In EACH file, BPE must be enabled BEFORE the first training call. Check:
- Fresh start path
- Checkpoint load failure path (if applicable)

### E. Sigmoid Corpus Fade
In EACH file's generation function, corpus fade must be computed PER TOKEN from local entropy inside the token loop. NOT once per message.

### F. Sampling Pipeline
In the generation token loop of EACH file, trace the exact order of operations:
1. Get logits from forward pass
2. Frequency/presence penalties
3. Temperature scaling
4. Softmax → model probs
5. Per-token entropy computation
6. Dissonance feedback (rescale if needed)
7. Corpus blend (with per-token sigmoid)
8. Pattern breaking (anti-field bypass)
9. Top-k / top-p / min-p / typical sampling
10. Repetition guard

Report if ANY file has a different order.

### G. Checkpoint Persistence
For EACH file, list what is saved/loaded in checkpoints:
- Model weights (base + deltas)
- Tokenizer state (vocab, BPE merges)
- global_step
- last_warmup_stage
- growth_step_offset (or equivalent)
- Conscience state (delta_alpha_scale, entropy history)
- Syntropy state

Report any fields that are saved but not loaded, or vice versa.

### H. Background Trainer
For EACH file:
- Does QuantumBuffer trigger correctly? (bytes OR novelty) AND cooldown
- Does per-stage warmup fire with scaled steps?
- Does SyntropyTracker measure + decide + apply correctly?
- Does immune system (snapshot → train → drift check → rollback) work?

### I. Swarm Ecology
For Go/Python/C/Rust: shared SQLite mesh.db with organisms table, heartbeat, mitosis, hibernation.
For JS: BroadcastChannel alternative (acceptable).
Report any file where mitosis or hibernation logic is missing or broken.

### J. Edge Cases & Safety
- What happens if corpus is empty (0 lines)?
- What happens if vocab_size changes during generation (BPE retrain)?
- KV cache: what happens when sequence length exceeds block_size?
- RRPRAM pattern matrix: are there bounds checks to prevent OOB?
- Thread safety: are shared structures properly locked?
- Integer overflow: can warmup_scale * warmup_steps overflow?

---

## Output Format

```
# RE-AUDIT REPORT

## PART 1: Fix Verification
| Fix # | Status | Details |
|-------|--------|---------|
| 1 | OK/BROKEN | ... |
| 2 | OK/BROKEN | ... |
...

## PART 2A: New Critical Bugs
1. [FILE:LINE] Description

## PART 2B: New Consistency Issues
1. [CONFIG] file1=X file2=Y expected=Z

## PART 2C: New Logic Errors
1. [FILE:LINE] Description

## PART 2D: New Missing Features
1. [FEATURE] missing in FILE

## PART 2E: Edge Case Risks
1. [FILE:LINE] Risk description

## SUMMARY
Fixes verified: N/8 OK
New critical bugs: N
New consistency issues: N
New logic errors: N
New edge case risks: N
```
