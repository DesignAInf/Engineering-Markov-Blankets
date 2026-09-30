# Engineering Markov Blankets

**When sensory states descend from active states**

*Luca M. Possati* · University of Twente · [l.m.possati@utwente.nl](mailto:l.m.possati@utwente.nl)

Reference implementation, paper and results for *Engineering Markov Blankets: When Sensory States Descend from Active States* (draft v9).

---

## The idea in one paragraph

Markov blankets are usually analysed as if the observation channel were exogenous. In that case, comparing two boundary designs is just comparing two statistical experiments, which Blackwell's theorem already handles. A Markov blanket, however, contains **active states**, and these can be causal ancestors of later **sensory states**: an agent can end up sensing its own echo.

This repository shows three things:

- this property is invisible to every criterion defined on per-step channel statistics;
- it nevertheless reduces the value of evidence, even for a perfectly specified agent;
- an agent that ignores it falls into confident, persistent error.

It then turns the property into a design problem at two levels:

- **external design**: should the loop be *cut*, *modelled* or *learned*?
- **self-engineering**: what happens when the agent chooses its own boundary?

## The model

A binary hidden state η, fixed within an episode of T steps. A decision agent announces its best guess a_t at every step (an active state). Observer A1 always reports an independent signal of reliability ρ. Observer A2 behaves differently in the two systems:

| System | Observer A2 | Blanket topology |
|---|---|---|
| **A**, closed loop | repeats the agent's previous announcement a_{t−1} with probability κ; otherwise reports an independent signal | s₂,t ∈ de(a_{t−1}) |
| **B**, open loop | independent signal whose reliability equals, step by step, the marginal reliability of s₂,t in A | no active → sensory path |

We compare two agents:

- a **naive** agent, which uses a factorised likelihood with the exact per-step marginal reliability;
- an **aligned** agent, which models the path a_{t−1} → s₂,t (an efference copy).

## Main results

### Theory

| # | Statement | Code |
|---|---|---|
| Prop. 1 | A and B are **the same experiment at every step**: p_A(s_t\|η) = p_B(s_t\|η). Blackwell ordering, Lindley information and one-step value of information cannot tell them apart. | `emb.theory.reff_from_alpha`, test `test_stepwise_marginals_match_between_A_and_B` |
| Prop. 2 | Their joint laws differ: Cov_A(s₂,t, a_{t−1} \| η) = κ·Var(a_{t−1} \| η) > 0. The factorised model is exact in B and misspecified in A. | `emb.theory.cov_echo` |
| Prop. 3 | The closed loop is a **sequential garbling** of its step-equivalent open loop, so even a perfectly specified agent loses decision value. | `emb.theory.garbling_flip_prob` |
| Prop. 4 | Above **κ\*(ρ, r)** the naive agent's wrong beliefs are self-reinforcing in expectation. The aligned agent's never are. | `emb.theory.kappa_star`, `naive_drift_when_wrong` |
| Prop. 5 | **Self-sealing boundaries.** Cutting the loop removes all evidence about the loop, so a greedy agent that hides its announcements once hides them forever. | test `test_hidden_boundary_leaves_kappa_belief_unchanged` |

### Simulations (ρ = 0.6, κ = 0.5, T = 30)

| Condition | Final accuracy | Mean confidence | Confident lock-in on wrong state |
|---|---|---|---|
| B, open loop, naive (exact model) | 0.985 | 0.985 | 0.06 % |
| A, closed loop, aligned | 0.902 | 0.902 | 0.10 % |
| A, closed loop, naive | **0.762** | **0.995** | **19.3 %** |

- **Early errors persist.** The naive agent wrong at step 10 is still wrong at step 30 in 86 % of cases, against 27 % for the aligned agent and 6 % in B.
- **The threshold is where the damage starts.** The analytic κ\* ≈ 0.25–0.30 marks where simulated lock-in accelerates.
- **Remedies at fixed κ.** Cutting the loop gives 0.940, modelling it with known κ gives 0.902, and learning κ passively gives 0.890 (91 % of the loss recovered). One-step expected-free-energy probing within an episode is almost never selected and adds nothing measurable: an honest negative result.
- **Self-engineering (40 episodes, cutting cost C = 0.03).** A greedy agent seals its boundary in every world and never learns κ (cumulative regret 0.297). An agent that adds expected information gain about its own blanket topology (an EFE term) avoids the trap: regret falls to 0.115 (−61 %), and it matches the oracle boundary policy in 89 % of worlds.

> **Design principle.** For every sensory state, determine whether it has active-state ancestors. Where it does, cut the path, model it, or learn it. If the system chooses for itself, do not let it cut what it has not yet learned.

## Installation

```bash
git clone <this-repository-url>
cd emb-markov-blankets
pip install -e ".[dev]"
```

Requires Python ≥ 3.9, NumPy and Matplotlib.

## Usage

```bash
# unit tests (seconds)
pytest -q

# quick smoke run of all experiments (a few minutes, small N)
python -m emb.experiments --quick --out runs/quick

# full reproduction of every number and figure in the paper
python -m emb.experiments --out runs/full

# only E1-E4 (equivalence, sweeps, threshold, remedies)
python -m emb.experiments --only v8
# only E5-E6 (design map, self-engineering)
python -m emb.experiments --only v9
```

After `pip install`, the command `emb-reproduce` is equivalent to `python -m emb.experiments`. The full run takes on the order of an hour on a laptop, most of it spent in the EFE learners of E4 and in E6.

`--quick` is a smoke test only. With small samples the numbers are noisy: individual regrets can even come out negative, because the oracle is evaluated on tabulated accuracies. Use the full run to reproduce the paper.

### Reproducibility

- All randomness is seeded (`emb.SEED = 20260930`).
- The reference outputs of the full run are in [`results/`](results/).
- The package was checked against the original research scripts by running both at reduced sample sizes with identical seeds. The output JSON files coincided exactly, with zero differences. See [`results/README.md`](results/README.md).

### Using the library directly

```python
import emb

T, rho, kappa = 30, 0.6, 0.5
reff = emb.fixed_point_reff(50_000, T, rho, kappa)      # step-matched open loop

loop_naive   = emb.simulate("A", "naive",   50_000, T, rho, kappa, reff)["final"]
loop_aligned = emb.simulate("A", "aligned", 50_000, T, rho, kappa, reff)["final"]
open_loop    = emb.simulate("B", "naive",   50_000, T, rho, kappa, reff, seed=1)["final"]

print(emb.kappa_star(rho, reff[-1]))                     # self-reinforcement threshold
```

## Repository structure

```
emb-markov-blankets/
├── src/emb/
│   ├── core.py          # systems A/B, naive and aligned agents, fixed point for rho_eff
│   ├── learner.py       # joint (eta, kappa) learner, probing, multi-episode boundary choice
│   ├── theory.py        # closed-form quantities of Propositions 1-4
│   └── experiments.py   # E1-E6, figures, JSON; CLI entry point
├── tests/               # unit tests for the propositions and the simulator
├── paper/               # LaTeX source, compiled PDF, figures
├── results/             # reference outputs (JSON + figures) of the full run
├── scripts/reproduce.sh
├── .github/workflows/ci.yml
├── CITATION.cff
├── LICENSE              # MIT
└── pyproject.toml
```

## Experiments and outputs

| Experiment | Paper section | Figure | JSON key |
|---|---|---|---|
| E1 Step-wise equivalence, value loss, miscalibration | 5.1 | `fig1_equivalence.pdf` | `results_v8.json → E1` |
| E2 κ sweep, ρ robustness | 5.2 | `fig2_kappa_sweep.pdf` | `E2_kappa_sweep`, `E2_rho_robustness` |
| E3 Self-reinforcement threshold | 5.3 | `fig3_threshold.pdf` | `E3_threshold` |
| E4 Remedies (structural / inferential / epistemic) | 6 | `fig4_remedies.pdf` | `E4_remedies`, `E4_learner_by_kappa` |
| E5 Design map | 7.1 | `fig5_design_map.pdf` | `results_v9.json → E5_design_map` |
| E6 Self-engineering | 7.2 | `fig6_self_engineering.pdf` | `E6_self_engineering` |

No external data are used.

## Relation to *Design for Entropy*

This work is the formal companion to chapter 4 ("Design as the Engineering of Markov Blankets: A Roadmap") and part of chapter 6 of L. M. Possati, *Design for Entropy: Active Inference and Technology* (MIT Press, 2026).

- **The pavilion constraint.** The book's design constraint that sensors must not depend on actuators ("no remote control of the input") is exactly the property studied here. The paper shows why it matters, and that it must be stated over time (S_t ⊥ A_{<t} \| E), not only instant by instant.
- **Echo chambers.** The book's "over-synchronization" between user and artifact is modelled here as the closed loop A.
- **Desynchronization.** The book's "desynchronization spaces" correspond to the structural remedy and to the self-engineering agent's choice to hide.

## Limitations

- The model is minimal: binary state, stationary environment, known ρ, a single loop and greedy announcements.
- The comparator B is a construction defined relative to a given announcement process.
- Proposition 3 assumes that the announcement accuracy is at least ρ at every step. This holds in all runs, but not universally.
- The EFE agents use a one-step horizon (within episode) or a bank-based estimate of information gain (across episodes).
- Costs are stipulated in accuracy units.

## Citation

```bibtex
@misc{possati2026engineering,
  author = {Possati, Luca M.},
  title  = {Engineering Markov Blankets: When Sensory States Descend from Active States},
  year   = {2026},
  note   = {Preprint, draft v9. Code: engineering-markov-blankets v0.9.0}
}

@book{possati2026design,
  author    = {Possati, Luca M.},
  title     = {Design for Entropy: Active Inference and Technology},
  publisher = {The MIT Press},
  address   = {Cambridge, MA},
  year      = {2026}
}
```

## Contact

Luca M. Possati — [l.m.possati@utwente.nl](mailto:l.m.possati@utwente.nl)

## License

MIT © 2026 Luca M. Possati
