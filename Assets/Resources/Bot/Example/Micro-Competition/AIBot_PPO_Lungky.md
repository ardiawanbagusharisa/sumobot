# Lungky_AIBot_PPO

A PPO sumo bot built around five layered improvements over the baseline `AIBot_PPO.cs`.

- **File:** [Lungky_AIBot_PPO.cs](Lungky_AIBot_PPO.cs)
- **Bot ID:** `Bot_Lungky_PPO`
- **Default skill:** `SkillType.Boost`
- **External dependencies:** none — no ONNX, no pretrained models, no async I/O.

## The five layers (Phases 1–4 + 6)

| Phase | Improvement | Where it lives |
|---|---|---|
| 1 | **Rich observation + running-mean/std normalisation** | `LungkyObservation`, `ObservationNormalizer` |
| 2 | **Action masking** (dash/skill never sampled while on cooldown) | `ActionSpace.LegalMask`, `MaskedSoftmax` |
| 3 | **Proper GAE-λ** (replaces single-step bootstrap) | `EpisodeBuffer.ComputeGAE` |
| 4 | **Imitation warm-start from the rule-based Utility bot** (BC + KL anchor) | `UtilityTeacher`, blended into the action distribution and MCTS prior |
| 6 | **MCTS at inference** using policy as prior + value head as leaf + `api.Simulate()` as forward model | `MCTSPlanner` |

Phase 5 (self-play league) and Phase 7+ (LSTM, RND, distributional critic) are intentionally out of scope.

---

## Why this design

The baseline PPO in the repo has three structural caps that bound its performance:

1. 4-feature observation, no normalisation.
2. No action masking — wastes probability mass on cooldown-blocked actions.
3. Single-step bootstrap instead of GAE — high variance.
4. No warm-start — trains from random weights, never reaches the rule-based bot's level within a session.
5. No planning at inference — reactive only.

This bot fixes all five. The critical insight: by using the **Lungky Utility bot's scoring logic as an in-process teacher**, we get a strong day-1 baseline (Utility-bot behaviour, plus MCTS planning on top) without any pretrained model file. Then PPO fine-tunes from there.

---

## Day-1 behaviour (no training, no checkpoint)

When you press Play on a fresh project with default settings:

1. `OnBotInit` finds no checkpoint → fresh Xavier-initialised network.
2. Teacher is constructed (it's pure C#, always available).
3. `netIsTrained = false`, so `TeacherBlendAlpha() = 1.0` in MatchPlay.
4. Every tick:
   - Build 24-feature observation, normalise (running stats; barely active for first dozen ticks).
   - Compute action mask.
   - Compute teacher distribution from Utility bot's scoring rules.
   - Run MCTS with prior = `(1−1.0) · randomNet + 1.0 · teacher` = pure teacher prior. The MCTS still uses the random value head at leaves, which adds some noise, but the prior dominates because PUCT exploration scales with prior probability.
   - Emit argmax of visit counts → action.

In effect, on day-1 the bot plays approximately like the **Utility bot with MCTS planning bolted on**. That's already a competitive baseline. Train mode then improves from there.

If a checkpoint exists, `netIsTrained = true` and α drops to 0.1 — the trained policy dominates, with a light KL pull toward the teacher.

---

## Components

### `LungkyObservation` (24 floats)

| Idx | Feature |
|---|---|
| 0, 1 | My (x, y) offset from arena centre, divided by radius |
| 2, 3 | My facing as (sin, cos) — avoids the ±π discontinuity |
| 4, 5 | My linear velocity / `MoveSpeed` |
| 6 | My angular velocity / `RotateSpeed` |
| 7 | Distance to foe / arena diameter |
| 8, 9 | Bearing to foe in my frame (sin, cos) |
| 10, 11 | Bearing to arena centre in my frame (sin, cos) |
| 12 | My distance from centre / radius |
| 13 | Critical-edge binary (>0.92) |
| 14 | Dash on cooldown (0/1) |
| 15 | Skill cooldown normalised |
| 16, 17 | Dash active, skill active (0/1) |
| 18 | Foe distance from centre / radius |
| 19 | Foe dash active (0/1) |
| 20 | Foe velocity component along me-foe line |
| 21 | Foe speed magnitude / `MoveSpeed` |
| 22 | Time-left fraction |
| 23 | Round-score signum (Left ahead, tied, Right ahead) |

### `ObservationNormalizer`

Welford running mean & variance, applied in-place per tick. Clips to `±5σ`. Updates only in Train mode; in MatchPlay it just applies the saved statistics. Doubles or triples PPO sample efficiency in practice — without this, the network spends its first few hundred steps just calibrating to the input scale.

### `ActionSpace`

5 discrete actions: Accelerate (0.1s), TurnLeft (0.2s), TurnRight (0.2s), Dash, Skill.

`LegalMask(api)` blocks Dash when `IsDashOnCooldown || IsDashActive`, Skill when `IsSkillOnCooldown || IsActive`. The mask is applied **inside the softmax** so illegal actions get zero probability mass and zero policy gradient. `BlendDistributions(a, b, α, mask)` mixes two distributions and re-normalises within the legal set.

### `UtilityTeacher` (the Phase 4 imitation source)

A self-contained port of the scoring logic from `Lungky_AIBot_RuleBased_Utility.cs`. It does *not* depend on the Utility bot class — the considerations, snapshot, and 12 action templates are copied into the teacher.

**Output:** a 5-way distribution over the PPO action space.

**How the 12 Utility actions map to 5 PPO actions:**

| Utility action | PPO bucket |
|---|---|
| Charge, EscapeInward | Accelerate |
| TurnLeftToEnemy, TurnLeftToCenter | TurnLeft |
| TurnRightToEnemy, TurnRightToCenter, AboutFace | TurnRight |
| DashStrike, FinishingDash | Dash |
| BoostBeforeRam, SkillEscape, PunishEnemyDash | Skill |

Each template's product-of-considerations score is added to its PPO bucket. Then a temperature-0.5 softmax converts the bucket scores into a distribution. A tiny uniform floor (`1e-3`) over legal actions keeps KL/CE losses finite.

### `LungkyAZNet`

Hand-rolled actor-critic MLP: 24 → 64 → 64 → {5 logits, 1 value}. Tanh activations, Xavier init, Adam optimiser, global L2 gradient clipping.

The PPO update combines five gradient signals on the policy logits:

```
dLogits[j] = dPG[j] + dEntropy[j] + dBC[j] + dKL[j]
                                  └─ behavioural cloning
                                          └─ KL-anchor toward teacher
```

Plus the standard MSE gradient on the value head against the GAE return.

### `EpisodeBuffer`

On-policy rollout, GAE-λ recursion in a single backward pass. Advantages are normalised to zero-mean unit-std per update epoch.

### `MCTSPlanner`

PUCT search. At the root the prior is `(1−α)·policyPrior + α·teacherPrior`, where α is the teacher blend factor described above. Selection uses

```
PUCT(a) = Q(a) + c · P(a|s) · √(ΣN) / (1 + N(a))
```

Leaves are evaluated with the value head (no Monte-Carlo rollouts — this is the AlphaZero design). The action returned is the **argmax over visit counts**, which is more stable than argmax over Q-values.

**Approximation:** the world model is simplified. `api.Simulate()` only forwards one robot kinematically and doesn't model collisions. We keep depth shallow (default 4) and let the value head carry longer-horizon credit. For now the scaffold doesn't apply the simulated next-state to the child node observation — see *Known limitations* below.

---

## Run modes

`mode` inspector field:

- **MatchPlay** — frozen weights, MCTS at inference, no episode bookkeeping. Action distribution is `(1−α) · policy + α · teacher`. α is `1.0` if no checkpoint loaded, `0.1` otherwise.
- **Train** — collects experience, calls PPO update at episode end, can save checkpoints. MCTS is skipped (it would break the PPO importance-sampling assumption). α decays linearly from `1.0` to `0` over `bcWarmupEpisodes`.

---

## Training recipe

| Episodes | Effective `bcWeight` | `klAnchorCoef` | What's happening |
|---|---|---|---|
| 0–10 | 1.0 | 0.02 | BC-dominated — policy converges to a Utility-bot clone. |
| 10–50 | 1.0 → 0.0 (linear) | 0.02 | PPO takes over; BC weight decays to zero by `bcWarmupEpisodes`. |
| 50+ | 0 | 0.02 | Free-form RL with a light KL anchor. |

Concrete steps in the Unity Editor:

1. Set `mode = Train`, `useUtilityTeacher = true`, `saveCheckpoint = true`, `useMCTSAtInference = false`.
2. Run battles until `episodeIndex` reaches `maxTotalEpisodes`.
3. Switch to `mode = MatchPlay`, `loadCheckpoint = true`, `useMCTSAtInference = true`. Play tournaments.

---

## Known limitations

- **MCTS child-node observation is not updated** to the simulated state. The scaffold reuses the root observation at children, which makes MCTS more like "policy improvement via repeated value queries" than true game-tree search. Wiring `api.Simulate()` output back into an observation rebuild is the obvious follow-up (~30 lines, mostly mechanical).
- **MCTS world model ignores the opponent.** `api.Simulate()` only forwards one robot. Past depth ~5 plies the prediction diverges from reality; we mitigate via shallow depth + learned value head.
- **KL gradient is simplified** — implemented as cross-entropy shape (off by a constant). The optimum is unchanged but step magnitudes differ slightly from the textbook KL gradient.
- **No checkpoint shipped.** First run uses fresh weights and leans entirely on the teacher prior; that's intentional and the design accounts for it via `TeacherBlendAlpha()`.
- **CPU-only.** Network is small enough that 48 MCTS sims fit comfortably in a 100ms tick. If you raise `mctsSimulations` past ~200 you may approach the budget.
- **Manual backprop.** Adding a new loss term requires hand-deriving the gradient.

---

## Future work (in payoff order)

1. **Apply `api.Simulate()` to child observations in MCTS** — turns the scaffold into proper game-tree search.
2. **Frame stacking or a tiny GRU** for detecting opponent windups (turning + accelerating → incoming dash).
3. **Self-play league** (Phase 5) — keep a rolling pool of past checkpoints, sample opponents from it during training to prevent overfit / cycles.
4. **PFSP opponent sampling** — bias the self-play league toward opponents we currently lose against.
5. **Curriculum** — train against weak fixed opponents first (Primitive → FSM → Utility), then graduate to self-play.
6. **Distributional value head** for risk-aware behaviour near the edge.
7. **GPU inference** via Unity InferenceEngine if MCTS budget tightens.
