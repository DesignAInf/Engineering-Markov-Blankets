"""Reproduce every experiment and figure of the paper.

    python -m emb.experiments                 # full run (paper numbers)
    python -m emb.experiments --quick         # small N, for smoke tests / CI
    python -m emb.experiments --only v8       # E1-E4 only
    python -m emb.experiments --only v9       # E5-E6 only

E1  Step-wise equivalence of closed loop (A) and open loop (B); naive vs aligned.
E2  kappa sweep and rho robustness.
E3  Analytic self-reinforcement threshold vs simulated lock-in.
E4  Remedies: structural, inferential, epistemic (passive / random / EFE probes).
E5  Design map (external designer, explicit costs).
E6  Self-engineering: the agent chooses its own boundary configuration.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .core import SEED, simulate, fixed_point_reff  # noqa: E402
from .learner import DEFAULT_GRID, entropy, run_episode, simulate_learner  # noqa: E402
from .theory import kappa_star, naive_drift_when_wrong  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


# ============================================================================
# E1-E4
# ============================================================================
def run_v8(out: str, N1=200_000, N2=100_000, N4=50_000, T=30, RHO=0.60, KAPPA=0.50) -> dict:
    os.makedirs(out, exist_ok=True)
    R = {"params": dict(T=T, rho=RHO, kappa=KAPPA, seed=SEED, N_main=N1, N_sweep=N2, N_remedies=N4)}

    # ---------------- E1 ----------------
    reff = fixed_point_reff(N1, T, RHO, KAPPA)
    An = simulate("A", "naive", N1, T, RHO, KAPPA, reff, track=True)
    Bn = simulate("B", "naive", N1, T, RHO, KAPPA, reff, seed=SEED + 1, track=True)
    Aa = simulate("A", "aligned", N1, T, RHO, KAPPA, reff, track=True)

    def persistence(d):
        w = d["wrong"]
        return dict(P_wrong_at_10=float(w[9].mean()),
                    P_wrong_at_T_given_wrong_at_10=float(w[-1][w[9]].mean()) if w[9].any() else float("nan"))

    R["E1"] = dict(
        reff=reff.round(5).tolist(),
        max_abs_marginal_diff=float(np.max(np.abs(An["s2acc"] - Bn["s2acc"]))),
        A_naive=An["final"], B_naive=Bn["final"], A_aligned=Aa["final"],
        persistence=dict(A_naive=persistence(An), A_aligned=persistence(Aa), B_naive=persistence(Bn)),
    )
    # flat keys kept for compatibility with the reference results file
    R["E1"]["naive_A_P_wrong_at_10"] = R["E1"]["persistence"]["A_naive"]["P_wrong_at_10"]
    R["E1"]["naive_A_P_wrong_at_T_given_wrong_at_10"] = \
        R["E1"]["persistence"]["A_naive"]["P_wrong_at_T_given_wrong_at_10"]
    ts = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for d, lab in [(Bn, "B open loop, naive"), (Aa, "A closed loop, aligned"), (An, "A closed loop, naive")]:
        ax[0].plot(ts, d["acc"], label=lab)
        ax[1].plot(ts, d["conf"] - d["acc"], label=lab)
    ax[0].set_xlabel("step $t$"); ax[0].set_ylabel(r"$P(a_t=\eta)$"); ax[0].set_title("(a) Accuracy")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("step $t$"); ax[1].set_ylabel("mean confidence $-$ accuracy")
    ax[1].set_title("(b) Overconfidence"); ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/fig1_equivalence.pdf"); plt.close(fig)

    # ---------------- E2 ----------------
    sweep = []
    for k in np.round(np.arange(0.0, 0.91, 0.1), 2):
        rf = fixed_point_reff(N2, T, RHO, k)
        fn = simulate("A", "naive", N2, T, RHO, k, rf)["final"]
        fb = simulate("B", "naive", N2, T, RHO, k, rf, seed=SEED + 1)["final"]
        fa = simulate("A", "aligned", N2, T, RHO, k, rf)["final"]
        r_last = float(rf[-1])
        sweep.append(dict(kappa=float(k), r_final=r_last,
                          drift_wrong=float(naive_drift_when_wrong(RHO, k, r_last)) if k > 0 else None,
                          A_naive=fn, B_naive=fb, A_aligned=fa))
    R["E2_kappa_sweep"] = sweep

    rob = []
    for rho in [0.55, 0.60, 0.70]:
        rf = fixed_point_reff(N2, T, rho, KAPPA)
        rob.append(dict(rho=rho,
                        A_naive=simulate("A", "naive", N2, T, rho, KAPPA, rf)["final"],
                        B_naive=simulate("B", "naive", N2, T, rho, KAPPA, rf, seed=SEED + 1)["final"],
                        A_aligned=simulate("A", "aligned", N2, T, rho, KAPPA, rf)["final"]))
    R["E2_rho_robustness"] = rob

    ks = [s["kappa"] for s in sweep]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, lab in [("B_naive", "B open loop, naive"), ("A_aligned", "A closed loop, aligned"),
                     ("A_naive", "A closed loop, naive")]:
        ax[0].plot(ks, [s[key]["accuracy"] for s in sweep], marker="o", label=lab)
        ax[1].plot(ks, [s[key]["lockin"] for s in sweep], marker="o", label=lab)
    ax[0].set_xlabel(r"$\kappa$"); ax[0].set_ylabel(f"accuracy at $T={T}$"); ax[0].set_title("(a) Accuracy")
    ax[1].set_xlabel(r"$\kappa$"); ax[1].set_ylabel(r"$P(q(\eta_{\mathrm{true}})<0.01)$")
    ax[1].set_title("(b) Confident lock-in on the wrong state")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/fig2_kappa_sweep.pdf"); plt.close(fig)

    # ---------------- E3 ----------------
    R["E3_threshold"] = [dict(kappa=s["kappa"], r=s["r_final"], drift=s["drift_wrong"],
                              kappa_star=float(kappa_star(RHO, s["r_final"])),
                              lockin=s["A_naive"]["lockin"]) for s in sweep[1:]]
    kgrid = np.linspace(0.0, 0.9, 200)
    r_interp = np.interp(kgrid, ks, [s["r_final"] for s in sweep])
    fig, ax1 = plt.subplots(figsize=(5.5, 3.6))
    ax1.plot(kgrid, naive_drift_when_wrong(RHO, kgrid, r_interp), color="C3")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel(r"$\kappa$"); ax1.set_ylabel("expected drift (nats/step)", color="C3")
    ax2 = ax1.twinx()
    ax2.plot(ks, [s["A_naive"]["lockin"] for s in sweep], "o-", color="C0")
    ax2.set_ylabel("simulated lock-in", color="C0"); ax2.spines["right"].set_visible(True)
    fig.tight_layout(); fig.savefig(f"{out}/fig3_threshold.pdf"); plt.close(fig)

    # ---------------- E4 ----------------
    grid = DEFAULT_GRID
    rf = fixed_point_reff(N4, T, RHO, KAPPA)
    rem = {
        "none (naive)": simulate("A", "naive", N4, T, RHO, KAPPA, rf)["final"],
        "structural (cut loop)": simulate("A", "aligned", N4, T, RHO, 0.0, np.full(T, RHO))["final"],
        "inferential (known kappa)": simulate("A", "aligned", N4, T, RHO, KAPPA, rf)["final"],
        "epistemic: passive learner": simulate_learner(N4, T, RHO, KAPPA, grid, "passive"),
    }
    for eps in [0.1, 0.3]:
        rem[f"epistemic: random probes eps={eps}"] = simulate_learner(N4, T, RHO, KAPPA, grid, "random", eps=eps)
    for beta in [0.5, 1.0, 2.0, 4.0]:
        rem[f"epistemic: EFE beta={beta}"] = simulate_learner(N4, T, RHO, KAPPA, grid, "efe", beta=beta)
    R["E4_remedies"] = rem

    lr = []
    for kt in [0.2, 0.5, 0.8]:
        rfk = fixed_point_reff(N4, T, RHO, kt)
        lr.append(dict(
            kappa_true=kt,
            naive=simulate("A", "naive", N4, T, RHO, kt, rfk)["final"]["accuracy"],
            known=simulate("A", "aligned", N4, T, RHO, kt, rfk)["final"]["accuracy"],
            passive=simulate_learner(N4, T, RHO, kt, grid, "passive"),
            efe1=simulate_learner(N4, T, RHO, kt, grid, "efe", beta=1.0),
            random30=simulate_learner(N4, T, RHO, kt, grid, "random", eps=0.3),
        ))
    R["E4_learner_by_kappa"] = lr

    keys = ["epistemic: passive learner", "epistemic: random probes eps=0.1",
            "epistemic: random probes eps=0.3", "epistemic: EFE beta=0.5",
            "epistemic: EFE beta=1.0", "epistemic: EFE beta=2.0", "epistemic: EFE beta=4.0"]
    labs = ["passive learner", "random eps=0.1", "random eps=0.3", "EFE b=0.5", "EFE b=1", "EFE b=2", "EFE b=4"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].scatter([rem[k]["announcement_accuracy"] for k in keys], [rem[k]["kappa_abs_error"] for k in keys])
    for k, lab in zip(keys, labs):
        ax[0].annotate(lab, (rem[k]["announcement_accuracy"], rem[k]["kappa_abs_error"]), fontsize=7,
                       xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel("mean announcement accuracy (pragmatic)")
    ax[0].set_ylabel(r"$|\hat\kappa-\kappa|$ at $T$ (epistemic)"); ax[0].set_title("(a) Probing trade-off")
    names = ["none (naive)", "structural (cut loop)", "inferential (known kappa)",
             "epistemic: passive learner", "epistemic: EFE beta=1.0"]
    vals = [rem[n].get("accuracy", rem[n].get("final_accuracy")) for n in names]
    ax[1].bar(range(len(names)), vals, color=["C3", "C2", "C1", "C0", "C4"])
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(["none", "structural", "inferential", "passive\nlearner", "EFE\nlearner"], fontsize=8)
    ax[1].set_ylim(0.5, 1.0); ax[1].set_ylabel(f"final accuracy ($T={T}$)"); ax[1].set_title("(b) Remedies")
    fig.tight_layout(); fig.savefig(f"{out}/fig4_remedies.pdf"); plt.close(fig)

    with open(f"{out}/results_v8.json", "w") as f:
        json.dump(R, f, indent=1)
    return R


# ============================================================================
# E5-E6
# ============================================================================
def run_v9(out: str, B=16_000, N_ref=50_000, N_hid=20_000, NW=3000, M=40, T=30, RHO=0.60,
           costs=(0.01, 0.03, 0.05)) -> dict:
    os.makedirs(out, exist_ok=True)
    GRID = DEFAULT_GRID
    K = len(GRID)
    R = {"params": dict(T=T, rho=RHO, grid=GRID.tolist(), seed=SEED, B=B, NW=NW, M=M)}
    state = {"rng": np.random.default_rng(SEED)}

    # ---------------- bank: uniform-prior visible learner ----------------
    uniform = np.full((B, K), 1.0 / K)
    acc_learn, bank = [], []
    for k in GRID:
        c, _, lk = run_episode(state["rng"], uniform, np.full(B, k), np.ones(B, bool), T=T, rho=RHO)
        acc_learn.append(float(c.mean()))
        bank.append(lk)
    bank = np.stack(bank)
    acc_learn = np.array(acc_learn)
    c_hid, _, _ = run_episode(state["rng"], np.full((N_hid, K), 1.0 / K), np.zeros(N_hid),
                              np.zeros(N_hid, bool), T=T, rho=RHO)
    acc_hid = float(c_hid.mean())

    acc_naive, acc_known = [], []
    for k in GRID:
        rf = fixed_point_reff(N_ref, T, RHO, k)
        acc_naive.append(simulate("A", "naive", N_ref, T, RHO, k, rf)["final"]["accuracy"])
        acc_known.append(simulate("A", "aligned", N_ref, T, RHO, k, rf)["final"]["accuracy"])
    acc_naive, acc_known = np.array(acc_naive), np.array(acc_known)
    R["tables"] = dict(kappa=GRID.tolist(), acc_naive=acc_naive.tolist(), acc_learn=acc_learn.tolist(),
                       acc_known=acc_known.tolist(), acc_hidden=acc_hid)

    # ---------------- E5: design map ----------------
    Ccut = np.linspace(0, 0.10, 101)
    maps = {}
    cmap = ListedColormap(["#d62728", "#2ca02c", "#1f77b4"])  # none, cut, learn
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, Cm in zip(axes, [0.0, 0.03]):
        J = np.stack([
            np.broadcast_to(1 - acc_naive[None, :], (len(Ccut), K)),
            (1 - acc_hid) + Ccut[:, None] * np.ones((1, K)),
            np.broadcast_to(1 - acc_learn[None, :] + Cm, (len(Ccut), K)),
        ])
        choice = J.argmin(axis=0)
        maps[f"C_model={Cm}"] = choice.tolist()
        ax.imshow(choice, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2,
                  extent=[GRID[0] - 0.05, GRID[-1] + 0.05, Ccut[0], Ccut[-1]])
        ax.set_xlabel(r"loop strength $\kappa$"); ax.set_title(rf"$C_{{\mathrm{{model}}}}={Cm}$")
    axes[0].set_ylabel(r"cost of cutting the loop $C_{\mathrm{cut}}$")
    axes[1].legend(handles=[Patch(color=c, label=lab) for c, lab in
                            zip(cmap.colors, ["none (naive)", "cut", "learn"])],
                   frameon=True, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}/fig5_design_map.pdf"); plt.close(fig)
    R["E5_design_map"] = dict(Ccut=Ccut.tolist(), maps=maps)

    # ---------------- E6: self-engineering ----------------
    def ig_visible(q, S=64):
        rng = state["rng"]
        N = q.shape[0]
        ks = np.array([rng.choice(K, size=S, p=qi) for qi in q])
        idx = rng.integers(0, B, size=(N, S))
        lk = bank[ks, idx]
        lp = np.log(q[:, None, :] + 1e-300) + lk
        lp -= lp.max(axis=2, keepdims=True)
        p = np.exp(lp); p /= p.sum(axis=2, keepdims=True)
        return entropy(q) - entropy(p).mean(axis=1)

    kap_w = np.repeat(GRID, NW // K)
    NW = len(kap_w)
    ki = np.searchsorted(GRID, kap_w)
    labels = [("greedy (beta=0)", 0.0), ("EFE beta=0.05", 0.05), ("EFE beta=0.1", 0.1),
              ("EFE beta=0.2", 0.2), ("always visible", None), ("always hidden", None)]
    agents = {}
    for C in costs:
        oracle_util = np.maximum(acc_known, acc_hid - C)
        oracle_hide = (acc_hid - C) > acc_known
        for label, beta in labels:
            erng = np.random.default_rng(SEED + 7)
            state["rng"] = np.random.default_rng(SEED + 11)
            q = np.full((NW, K), 1.0 / K)
            util = np.zeros((M, NW)); hide_hist = np.zeros((M, NW), bool)
            for m in range(M):
                if label == "always visible":
                    hide = np.zeros(NW, bool)
                elif label == "always hidden":
                    hide = np.ones(NW, bool)
                else:
                    err_vis = q @ (1 - acc_learn)
                    G_vis = err_vis - (beta * ig_visible(q) if beta > 0 else 0.0)
                    hide = (1 - acc_hid) + C < G_vis
                correct, q, _ = run_episode(erng, q, kap_w, ~hide, T=T, rho=RHO)
                util[m] = correct.astype(float) - C * hide
                hide_hist[m] = hide
            regret = oracle_util[ki][None, :] - util
            final_match = np.mean(hide_hist[-5:].mean(axis=0).round().astype(bool) == oracle_hide[ki])
            per_k = {f"{k:.1f}": dict(hide_rate_last5=float(hide_hist[-5:, ki == i].mean()),
                                      hide_rate_first=float(hide_hist[0, ki == i].mean()),
                                      mean_util=float(util[:, ki == i].mean()),
                                      oracle_hide=bool(oracle_hide[i])) for i, k in enumerate(GRID)}
            agents[f"C={C} | {label}"] = dict(
                mean_util=float(util.mean()), mean_regret=float(regret.mean()),
                cum_regret=float(regret.sum(axis=0).mean()),
                final_policy_matches_oracle=float(final_match),
                hide_rate_first_episode=float(hide_hist[0].mean()),
                kappa_posterior_entropy_final=float(entropy(q).mean()),
                regret_curve=regret.mean(axis=1).tolist(), per_kappa=per_k)
            print(f"  C={C:.2f} {label:18s} cum_regret={regret.sum(axis=0).mean():.3f} "
                  f"match={final_match:.3f}", flush=True)
    R["E6_self_engineering"] = agents

    Cfig = 0.03 if 0.03 in costs else costs[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for label, _ in labels:
        ax[0].plot(np.arange(1, M + 1), np.cumsum(agents[f"C={Cfig} | {label}"]["regret_curve"]), label=label)
    ax[0].set_xlabel("episode"); ax[0].set_ylabel("cumulative regret vs oracle")
    ax[0].set_title(rf"(a) Cumulative regret, $C={Cfig}$"); ax[0].legend(frameon=False, fontsize=7)
    for label in ["greedy (beta=0)", "EFE beta=0.1"]:
        a = agents[f"C={Cfig} | {label}"]
        ax[1].plot(GRID, [a["per_kappa"][f"{k:.1f}"]["hide_rate_last5"] for k in GRID], "o-", label=label)
    orc = [agents[f"C={Cfig} | greedy (beta=0)"]["per_kappa"][f"{k:.1f}"]["oracle_hide"] for k in GRID]
    ax[1].step(GRID, orc, where="mid", color="k", lw=1, ls="--", label="oracle")
    ax[1].set_xlabel(r"true loop strength $\kappa$"); ax[1].set_ylabel("P(hide), last 5 episodes")
    ax[1].set_title("(b) Learned boundary policy"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/fig6_self_engineering.pdf"); plt.close(fig)

    with open(f"{out}/results_v9.json", "w") as f:
        json.dump(R, f, indent=1)
    return R


QUICK_V8 = dict(N1=20_000, N2=10_000, N4=5_000)
QUICK_V9 = dict(B=2_000, N_ref=10_000, N_hid=5_000, NW=500, M=12)


def main(argv=None):
    p = argparse.ArgumentParser(description="Reproduce the experiments of 'Engineering Markov Blankets'.")
    p.add_argument("--out", default="results", help="output directory for figures and JSON")
    p.add_argument("--quick", action="store_true", help="small sample sizes (minutes, not hours)")
    p.add_argument("--only", choices=["v8", "v9"], help="run only E1-E4 (v8) or E5-E6 (v9)")
    args = p.parse_args(argv)
    if args.only in (None, "v8"):
        print("Running E1-E4 ...", flush=True)
        run_v8(args.out, **(QUICK_V8 if args.quick else {}))
    if args.only in (None, "v9"):
        print("Running E5-E6 ...", flush=True)
        run_v9(args.out, **(QUICK_V9 if args.quick else {}))
    print(f"Done. Results in {args.out}/", flush=True)


if __name__ == "__main__":
    main()
