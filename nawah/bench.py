"""
Experiment harness. Everything in NAWAH is measured here; every number in
the report passes through it.

Two rules I committed to after getting burned twice:
  - No number from a single seed.
  - No effect without an ablation — the difference could come from anything.
"""

from __future__ import annotations

import random

import numpy as np

from .agent import BaselineAgent, NawahAgent
from .skill import distill
from .store import SkillStore, capacity_limit, measure_precision
from .tasks import (
    FAMILIES,
    GENERATORS,
    PRIMITIVES,
    StringTask,
    all_pairs,
    load_composed_tasks,
    load_primitive_tasks,
    load_string_tasks,
)
from .theory import (
    beta_cross_check,
    fit_beta_costs,
    fit_capacity_exponent,
    gap_exponent_from_capacity,
    regime,
)


# ─────────────────────────────────────────────────────────────
def split_tasks(tasks, n_probe_per_family=6):
    by = {}
    for t in tasks:
        by.setdefault(t.family, []).append(t)
    probes, rest = {}, {}
    for f, lst in by.items():
        probes[f] = lst[:n_probe_per_family]
        rest[f] = lst[n_probe_per_family:]
    return rest, probes


def interleave(per_fam, order=None):
    out, depth = [], max(len(v) for v in per_fam.values())
    fams = order or list(per_fam.keys())
    for d in range(depth):
        for f in fams:
            if d < len(per_fam[f]):
                out.append(per_fam[f][d])
    return out


def fit_beta(results):
    """cost ∝ N^(-beta).  beta > 0 means each task makes the next cheaper."""
    xs = np.log(np.arange(1, len(results) + 1))
    ys = np.log([max(1e-9, r.cost) for r in results])
    if len(xs) < 4:
        return float("nan"), float("nan")
    if np.std(ys) < 1e-9:
        # flat cost is not "unmeasurable" — it is exactly beta = 0
        return 0.0, 1.0
    sl, ic = np.polyfit(xs, ys, 1)
    pr = np.polyval([sl, ic], xs)
    r2 = 1 - np.sum((ys - pr) ** 2) / max(1e-12, np.sum((ys - ys.mean()) ** 2))
    return -sl, r2


def learning_curve(results, blocks=6):
    _n, sz = len(results), max(1, len(results) // blocks)
    return [
        100 * np.mean([r.ok for r in results[b * sz : (b + 1) * sz]])
        for b in range(blocks)
        if b * sz < len(results)
    ]


# ─────────────────────────────────────────────────────────────
def _m(a):
    """Mean of an array, NaN when there is nothing to average."""
    return float(a.mean()) if a.size else float("nan")


def _s(a):
    """Standard deviation of an array, NaN when it is undefined."""
    return float(a.std(ddof=1)) if a.size > 1 else float("nan")


def headline(make_backend, curriculum, probes, seeds=range(8), **kw):
    """
    The first experiment: does the layer do anything?

    Reports beta three ways and, crucially, reports whether the power-law
    specification SURVIVES. A beta quoted without that check is a number
    with no error bar on its own assumptions, which is how v2 ended up
    printing +0.257 as though it were a measurement of a law.
    """
    B, N, BETA, BS, LR, SPEC, COST = [], [], [], [], [], [], []
    for s in seeds:
        b = BaselineAgent(make_backend(seed=s), k_max=kw.get("k_max", 4))
        b.run(curriculum, probes)
        n = NawahAgent(make_backend(seed=s), seed=s, **kw)
        n.run(curriculum, probes)
        B.append(100 * np.mean([r.ok for r in b.results]))
        N.append(100 * np.mean([r.ok for r in n.results]))
        BETA.append(fit_beta(n.results)[0])
        # the cost that Law 1 is actually about: what it cost to add each
        # skill, not how many tries each task took (that one is censored)
        sc = skill_acquisition_costs(n)
        BS.append(fit_beta_costs(sc)[0] if len(sc) >= 8 else float("nan"))
        LR.append(learning_ratio(n.results))
        SPEC.append(
            bool(beta_cross_check([r.cost for r in n.results])["consistent"])
        )
        COST.append(np.mean([r.cost for r in n.results]))
    B, N, BETA, BS, LR, COST = map(np.array, (B, N, BETA, BS, LR, COST))
    # A seed that never acquired a skill contributes NaN, not a number.
    # np.nanmean on an all-NaN array returns NaN *and* warns, so filter
    # first rather than asking numpy to divide by zero quietly.
    ok_bs = BS[~np.isnan(BS)]
    ok_lr = LR[~np.isnan(LR)]
    return {
        "base_mean": B.mean(),
        "base_sd": B.std(),
        "naw_mean": N.mean(),
        "naw_sd": N.std(),
        "delta": N - B,
        "delta_mean": (N - B).mean(),
        "wins": int(((N - B) > 0).sum()),
        "n_seeds": len(seeds),
        "beta_mean": BETA.mean(),
        "beta_sd": BETA.std(),
        "beta_wins": int((BETA > 0).sum()),
        "beta_skill_mean": _m(ok_bs),
        "beta_skill_sd": _s(ok_bs),
        "beta_skill_wins": int((ok_bs > 0).sum()),
        "lr_mean": _m(ok_lr),
        "lr_sd": _s(ok_lr),
        "lr_wins": int((ok_lr > 1.0).sum()),
        "spec_pass": int(sum(SPEC)),
        "cost_mean": COST.mean(),
    }


def ablation(make_backend, curriculum, probes, seeds=range(4), **kw):
    """Which component is responsible?"""
    variants = {
        "full": {"decorrelate": True, "gate": True, "consolidate": True},
        "no forced decorrelation": {
            "decorrelate": False,
            "gate": True,
            "consolidate": True,
        },
        "no validation gate": {
            "decorrelate": True,
            "gate": False,
            "consolidate": True,
        },
        "no consolidation": {
            "decorrelate": True,
            "gate": True,
            "consolidate": False,
        },
        "no hierarchy": {
            "decorrelate": True,
            "gate": True,
            "consolidate": True,
            "hierarchical": False,
        },
    }
    out = {}
    for label, over in variants.items():
        accs, junk = [], []
        for s in seeds:
            kwv = dict(kw)
            kwv.update(over)
            ag = NawahAgent(make_backend(seed=s), seed=s, **kwv)
            ag.run(curriculum, probes)
            accs.append(100 * np.mean([r.ok for r in ag.results]))
            # genuinely corrupted (truth=False), not merely "unverified"
            junk.append(sum(1 for x in ag.store.skills if not x.truth))
        out[label] = (
            float(np.mean(accs)),
            float(np.std(accs)),
            float(np.mean(junk)),
        )
    return out


def transfer(make_backend, curriculum, probes, seeds=(0,), **kw):
    """Do skills move to an agent that has never seen the task?"""
    rows = []
    for s in seeds:
        teacher = NawahAgent(make_backend(seed=1000 + s), seed=1000 + s, **kw)
        teacher.run(curriculum, probes)

        virgin = NawahAgent(
            make_backend(seed=2000 + s), seed=2000 + s, decorrelate=False, **kw
        )
        for t in curriculum:
            virgin.solve(t, (), learn=False)

        student = NawahAgent(
            make_backend(seed=2000 + s), seed=2000 + s, decorrelate=False, **kw
        )
        offered = len(teacher.store.skills)
        accepted = teacher.donate(student, probes)
        student.run(curriculum, probes)

        rows.append(
            {
                "teacher": 100 * np.mean([r.ok for r in teacher.results]),
                "virgin": 100 * np.mean([r.ok for r in virgin.results]),
                "student": 100 * np.mean([r.ok for r in student.results]),
                "offered": offered,
                "accepted": accepted,
            }
        )
    return rows


def capacity_sweep(
    make_backend, curriculum, probes, widths=(8, 16, 32, 64), seeds=(0,), **kw
):
    """The capacity law: does it hold, and does it show up?"""
    rows = []
    for H in widths:
        accs, ns, mg = [], [], []
        for s in seeds:
            kwv = dict(kw)
            kwv["width"] = H
            ag = NawahAgent(make_backend(seed=s), seed=s, **kwv)
            ag.run(curriculum, probes)
            accs.append(100 * np.mean([r.ok for r in ag.results]))
            ns.append(len(ag.store))
            mg.append(ag.store.merges)
        rows.append(
            {
                "H": H,
                "T_star": capacity_limit(H),
                "acc": float(np.mean(accs)),
                "skills": float(np.mean(ns)),
                "merges": float(np.mean(mg)),
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────
def report_md(head, abl, tr, cap, title="NAWAH report") -> str:
    L = [f"# {title}", ""]
    L.append("## Effect")
    L.append("| | Accuracy |")
    L.append("|---|---|")
    L.append(
        f"| Model alone | {head['base_mean']:.1f}% ± {head['base_sd']:.1f} |"
    )
    L.append(
        f"| With NAWAH | **{head['naw_mean']:.1f}% ± {head['naw_sd']:.1f}** |"
    )
    L.append(
        f"| Difference | **{head['delta_mean']:+.1f} pp** "
        f"({head['wins']}/{head['n_seeds']} seeds) |"
    )
    L.append("")
    L.append(
        f"**beta = {head['beta_mean']:+.3f} ± {head['beta_sd']:.3f}** "
        f"({head['beta_wins']}/{head['n_seeds']} positive)"
    )
    L.append("")
    L.append(
        "Efficiency, measured three ways -- because a single exponent "
        "fitted to a curve that is not a power law is not a finding:"
    )
    L.append("")
    L.append("| Measure | Value | Seeds |")
    L.append("|---|---|---|")
    L.append(
        f"| Cost per task, first quarter vs last "
        f"| **x{head['lr_mean']:.2f}** ± {head['lr_sd']:.2f} "
        f"| {head['lr_wins']}/{head['n_seeds']} cheaper |"
    )
    L.append(
        f"| beta on cost per acquired skill "
        f"| {head['beta_skill_mean']:+.3f} ± {head['beta_skill_sd']:.3f} "
        f"| {head['beta_skill_wins']}/{head['n_seeds']} positive |"
    )
    L.append(
        f"| beta on cost per task "
        f"| {head['beta_mean']:+.3f} ± {head['beta_sd']:.3f} "
        f"| {head['beta_wins']}/{head['n_seeds']} positive |"
    )
    L.append("")
    L.append(
        f"Is the cost curve really a power law? "
        f"**{head['spec_pass']}/{head['n_seeds']} seeds pass** the "
        f"four-estimator cross-check. "
        + (
            "The form holds."
            if head["spec_pass"] > head["n_seeds"] // 2
            else "It does not, so the exponent above is indicative and the "
            "form-free ratio is the claim. Cost per task is censored "
            "between 1 and k_max, which is enough on its own to bend a "
            "power law out of shape."
        )
    )
    L.append("")
    L.append("## Which component is responsible")
    L.append("| Setting | Accuracy | Corrupted skills admitted |")
    L.append("|---|---|---|")
    for k, (m, sd, j) in abl.items():
        L.append(f"| {k} | {m:.1f}% ± {sd:.1f} | {j:.0f} |")
    L.append("")
    L.append("## Transfer")
    for r in tr:
        L.append(
            f"- teacher {r['teacher']:.1f}% · student with no skills "
            f"{r['virgin']:.1f}% · after inheritance "
            f"**{r['student']:.1f}%** "
            f"({r['accepted']}/{r['offered']} accepted)"
        )
    L.append("")
    L.append("## Capacity law")
    L.append("| Width H | Ceiling T* | Accuracy | Skills | Merges |")
    L.append("|---|---|---|---|---|")
    for r in cap:
        L.append(
            f"| {r['H']} | {r['T_star']:.1f} | {r['acc']:.1f}% "
            f"| {r['skills']:.0f} | {r['merges']:.0f} |"
        )
    L.append("")
    return "\n".join(L)


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 4 — measuring the ceiling instead of assuming it
# ══════════════════════════════════════════════════════════════
# In v2 the ceiling T* = 0.2234 H^1.494 was hard-coded into the noise and
# then printed in the results table. Measuring it would have returned the
# input. Here nothing is assumed: a library is grown until retrieval
# precision falls through a threshold, and whatever number that happens at
# is the number that gets reported.
def variant_probes(seed=0):
    """One representative task per variant of the string suite."""
    out, seen = [], set()
    rng = random.Random(seed)
    for _ in range(200):
        for f in FAMILIES:
            d, a, b, v = GENERATORS[f](rng)
            if v not in seen:
                seen.add(v)
                out.append(StringTask(len(out), f, d, a, b, v))
    return out


def synthetic_library(n, variants, seed=0, **store_kw):
    """
    A library of n skills spread evenly over the given variants, where each
    duplicate is a near-copy rather than a clone -- which is what a real
    library looks like after a few hundred distillations.
    """
    st = SkillStore(seed=seed, **store_kw)
    for i in range(n):
        fam, var = variants[i % len(variants)]
        s = distill(
            fam,
            f"how to handle {var}",
            f"[procedure draft {i}]",
            [("x", "y")],
            var,
        )
        s.truth = True
        s.is_sound = True
        st.add(s)
    return st


def measure_critical_library(
    width,
    n_warm=8,
    thresh=0.90,
    n_max=4096,
    seed=0,
    probes=None,
    seeds=3,
    **store_kw,
):
    """
    Grow the library until routing starts failing. Returns the last size
    that held, with the precision curve so the shape can be inspected
    rather than trusted.

    seeds: precision is averaged over this many independent runs. A single
    run is far too noisy to bisect on -- measured on one seed, a model 16x
    wider came out with a SMALLER ceiling, which is not a finding about
    models, it is a finding about sample size.
    """
    probes = probes or variant_probes(seed)
    variants = sorted({(t.family, t.variant) for t in probes})

    def prec(n):
        vals = []
        for k in range(max(1, seeds)):
            st = synthetic_library(
                n, variants, seed=seed + k, width=width, **store_kw
            )
            vals.append(
                measure_precision(
                    st,
                    probes,
                    rng=np.random.default_rng(seed + k),
                    n_warm=n_warm,
                )
            )
        return float(np.mean(vals))

    floor = max(1, len(variants))  # below this the library simply does
    # not cover every kind of task yet,
    # which is a different failure
    curve, n = [], floor
    while n <= n_max:
        p = prec(n)
        curve.append((n, p))
        if p < thresh:
            break
        n *= 2
    if n > n_max:
        return {"T_star": float(n_max), "saturated": True, "curve": curve}
    lo = floor if len(curve) < 2 else curve[-2][0]
    hi = n
    for _ in range(9):
        mid = (lo + hi) // 2
        if mid <= lo:
            break
        if prec(mid) >= thresh:
            lo = mid
        else:
            hi = mid
    return {"T_star": float(lo), "saturated": False, "curve": curve}


def capacity_two_sided(
    widths=(8, 32, 128, 512),
    n_warms=(2, 8, 32),
    thresh=0.90,
    seed=0,
    **store_kw,
):
    """
    The two-sided bound, measured. Sweep the model and sweep the
    experience, and look for the plateau that says each one alone stops
    being enough.
    """
    rows = []
    for nw in n_warms:
        for H in widths:
            m = measure_critical_library(
                H, n_warm=nw, thresh=thresh, seed=seed, **store_kw
            )
            rows.append(
                {
                    "H": H,
                    "n_warm": nw,
                    "T_star": m["T_star"],
                    "saturated": m["saturated"],
                }
            )
    return rows


def fit_measured_capacity(rows, n_warm):
    """Fit T* = A H^a on the measured row set. Returns (a, A, r2, theta)."""
    sub = [r for r in rows if r["n_warm"] == n_warm and r["T_star"] > 1]
    if len(sub) < 3:
        return {
            "a": float("nan"),
            "A": float("nan"),
            "r2": float("nan"),
            "theta": float("nan"),
            "n": len(sub),
        }
    a, A, r2 = fit_capacity_exponent(
        [r["H"] for r in sub], [r["T_star"] for r in sub]
    )
    return {
        "a": a,
        "A": A,
        "r2": r2,
        "theta": gap_exponent_from_capacity(a),
        "n": len(sub),
    }


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 5 — composition: can it chain what it learned apart?
# ══════════════════════════════════════════════════════════════
def composition_experiment(
    make_backend, per_op=32, seeds=range(4), k_max=4, test_per_pair=3, **kw
):
    """
    Train on single operations ONLY. Test on compositions of them that
    appear nowhere in training.

    This is the systematicity question: not "does it remember" but "does it
    combine". A lookup table cannot do this. A library that has genuinely
    abstracted the pieces can.
    """
    prim = load_primitive_tasks(per_op=per_op, seed=7)
    rest, probes = split_tasks(prim, n_probe_per_family=4)
    curr = interleave(rest)

    pairs = all_pairs(2)
    triples = all_pairs(3)
    test2 = load_composed_tasks(pairs, per_pair=test_per_pair, seed=11)
    test3 = load_composed_tasks(triples[:40], per_pair=2, seed=12)

    out = {
        "train": [],
        "base2": [],
        "naw2": [],
        "base3": [],
        "naw3": [],
        "skills": [],
        "n_train": len(curr),
        "n_test2": len(test2),
        "n_test3": len(test3),
    }
    for s in seeds:
        b = BaselineAgent(make_backend(seed=s), k_max=k_max)
        out["base2"].append(
            100 * np.mean([r.ok for r in b.run_composed(test2)])
        )
        BaselineAgent(make_backend(seed=s), k_max=k_max)
        out["base3"].append(
            100 * np.mean([r.ok for r in b.run_composed(test3)])
        )

        g = NawahAgent(make_backend(seed=s), seed=s, k_max=k_max, **kw)
        g.run(curr, probes)
        out["train"].append(100 * np.mean([r.ok for r in g.results]))
        out["skills"].append(len(g.store))
        g.results = []
        out["naw2"].append(
            100 * np.mean([r.ok for r in g.run_composed(test2)])
        )
        g.results = []
        out["naw3"].append(
            100 * np.mean([r.ok for r in g.run_composed(test3)])
        )
    for k in ("train", "base2", "naw2", "base3", "naw3", "skills"):
        out[k] = np.array(out[k], dtype=float)
    out["d2"] = out["naw2"] - out["base2"]
    out["d3"] = out["naw3"] - out["base3"]
    return out


def composition_scaling(
    make_backend,
    train_sizes=(44, 88, 143, 220, 330),
    seeds=range(4),
    k_max=4,
    n_probe=3,
    **kw,
):
    """Does systematicity GROW with the library, or is it all or nothing?"""
    pairs = all_pairs(2)
    test = load_composed_tasks(pairs, per_pair=3, seed=11)
    rows = []
    for n in train_sizes:
        per_op = max(1, round(n / len(PRIMITIVES)))
        prim = load_primitive_tasks(per_op=per_op, seed=7)
        rest, probes = split_tasks(prim, n_probe_per_family=n_probe)
        curr = interleave(rest)
        if not curr:  # a size too small to train on
            continue
        accs, comps, nsk = [], [], []
        for s in seeds:
            g = NawahAgent(make_backend(seed=s), seed=s, k_max=k_max, **kw)
            g.run(curr, probes)
            accs.append(100 * np.mean([r.ok for r in g.results]))
            nsk.append(len(g.store))
            keep = g.results
            g.results = []
            comps.append(100 * np.mean([r.ok for r in g.run_composed(test)]))
            g.results = keep
        rows.append(
            {
                "n_train": len(curr),
                "train_acc": float(np.mean(accs)),
                "compose_acc": float(np.mean(comps)),
                "compose_sd": float(np.std(comps)),
                "skills": float(np.mean(nsk)),
            }
        )
    return rows


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 6 — who chooses what to practise
# ══════════════════════════════════════════════════════════════
def curriculum_experiment(
    make_backend,
    budget=200,
    seeds=range(4),
    eval_every=50,
    k_max=4,
    suite="primitive",
    difficulty=None,
    **kw,
):
    """
    Same budget, same held-out test, five different answers to "what should
    it see next". The agent's own answer is the one that matters.

    difficulty: optional {variant: multiplier} handed to the backend. If
    every variant is equally hard there is nothing for a curriculum to
    discover -- practising uniformly is already Bayes-optimal and an
    adaptive rule can only add variance. Real skills are not equally hard,
    so the honest version of this experiment uses a spread.
    """
    from .curriculum import (
        AutoCurriculum,
        fixed_order_curriculum,
        primitive_factory,
        run_curriculum,
        string_factory,
    )

    if suite == "primitive":
        variants, gen = primitive_factory()
        test = load_composed_tasks(all_pairs(2)[:40], per_pair=3, seed=99)
    else:
        variants, gen = string_factory()
        test = load_string_tasks(per_family=6, seed=555)

    # Probe tasks for the gate. These MUST exist for every curriculum: with
    # an empty probe set validate() returns "no probe tasks" and rejects
    # every skill, so the agent never learns anything and the comparison
    # quietly measures a bug instead of a curriculum.
    src = (
        load_primitive_tasks(per_op=48, seed=7)
        if suite == "primitive"
        else load_string_tasks(per_family=48, seed=7)
    )
    _rest, probes = split_tasks(src, n_probe_per_family=4)

    def backend_for(seed):
        try:
            return make_backend(seed=seed, difficulty=difficulty)
        except TypeError:
            return make_backend(seed=seed)

    def evaluate(agent):
        res = list(agent.results)
        agent.results = []
        if suite == "primitive":
            acc = 100 * np.mean([r.ok for r in agent.run_composed(test)])
        else:
            acc = 100 * np.mean(
                [r.ok for r in (agent.solve(t, (), learn=False) for t in test)]
            )
        agent.results = res
        return acc

    specs = [
        ("interleaved", None),
        ("blocked", None),
        ("random", None),
        ("autocurriculum", "posterior_variance"),
        ("autocurriculum (naive)", "uncertainty"),
    ]
    out = {}
    for label, rule in specs:
        curves = []
        for s in seeds:
            ag = NawahAgent(backend_for(s), seed=s, k_max=k_max, **kw)
            if rule is None:
                ac = None
                pool = (
                    load_primitive_tasks(per_op=48, seed=7)
                    if suite == "primitive"
                    else load_string_tasks(48, seed=7)
                )
                rest, _p = split_tasks(pool, n_probe_per_family=4)
                base_pool = fixed_order_curriculum(
                    interleave(rest), seed=s, mode=label
                )
            else:
                ac = AutoCurriculum(variants, gen, seed=s)
                base_pool = []
            _res, tr = run_curriculum(
                ag,
                base_pool,
                budget,
                probes,
                autocurriculum=ac,
                evaluate=evaluate,
                eval_every=eval_every,
                rule=rule,
            )
            curves.append([p for _n, p in tr])
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves])
        out[label] = {
            "mean": arr.mean(axis=0).tolist(),
            "sd": arr.std(axis=0).tolist(),
            "final": float(arr[:, -1].mean()),
            "final_sd": float(arr[:, -1].std()),
        }
    out["_x"] = list(range(eval_every, budget + 1, eval_every))[:n]
    return out


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 7 — routing: what the context is worth
# ══════════════════════════════════════════════════════════════
def routing_experiment(
    contexts=(4, 9, 18, 36),
    per_context=4,
    budget=288,
    width=64,
    seed=0,
    n_warm=8,
    thresh=0.90,
):
    """
    Fixed library size, fixed experience budget, growing number of distinct
    situations the library has to serve.

    theory.routing_dilution says a context-blind router pays exactly M
    times more for its samples. So as M grows with the budget held fixed,
    the flat router should fall away and the contextual one should not.
    """
    rows = []
    for M in contexts:
        variants = [(f"fam{i % 6}", f"fam{i % 6}:synth{i}") for i in range(M)]
        probes = [
            StringTask(
                i,
                f"fam{i % 6}",
                f"Apply transformation number {i} to the input",
                f"input-{i}",
                f"output-{i}",
                f"fam{i % 6}:synth{i}",
            )
            for i in range(M)
        ]
        n_skills = per_context * M
        # total warm-up is held constant, so per-context experience thins
        # out exactly as the number of contexts grows
        # total supervised observations held constant at `budget`;
        # each round costs len(probes) = M of them, so the rounds
        # thin out exactly as the number of contexts grows
        n_warm_here = max(1, round(budget / max(1, len(probes))))
        res = {}
        for label, hier in (("contextual", True), ("flat", False)):
            st = synthetic_library(
                n_skills, variants, seed=seed, width=width, hierarchical=hier
            )
            p = measure_precision(
                st, probes, rng=np.random.default_rng(seed), n_warm=n_warm_here
            )
            res[label] = p
        rows.append(
            {
                "M": M,
                "skills": n_skills,
                "n_warm": n_warm_here,
                "contextual": res["contextual"],
                "flat": res["flat"],
                "gain": res["contextual"] - res["flat"],
            }
        )
    return rows


# ══════════════════════════════════════════════════════════════
# The audit — every claim in the report, re-checked
# ══════════════════════════════════════════════════════════════
def theory_audit(results_by_seed):
    """
    Run all four beta estimators on every seed and report whether they
    agree. A power law that only one estimator can see is not a power law.
    """
    out = []
    for s, res in results_by_seed.items():
        costs = [max(1e-9, r.cost) for r in res]
        cc = beta_cross_check(costs)
        cc["seed"] = s
        cc["regime"] = regime(cc["beta_direct"])
        out.append(cc)
    return out


# ══════════════════════════════════════════════════════════════
# MEASURING THE RIGHT COST
# ══════════════════════════════════════════════════════════════
def skill_acquisition_costs(agent):
    """
    Attempts spent between one ACCEPTED skill and the next.

    This is the quantity Law 1 is actually about: not "how many tries did
    task N take" but "what did it cost to add skill N to the library".

    The distinction is not cosmetic. Per-task cost is censored from both
    sides -- it cannot be below 1 or above k_max -- and fitting a power law
    to a censored variable produces a number that means nothing. That is
    exactly what was happening: the per-task curve is S-shaped on a bounded
    scale, and the strict specification test failed it (the second half
    gave beta = -0.39 against +0.10 for the first, and the power-law
    forecast lost to the flat null). Cost-to-next-skill is unbounded, so
    the law at least has a chance of being true of it.
    """
    costs, run = [], 0.0
    for r in agent.results:
        run += max(1e-9, r.cost)
        if r.learned:
            costs.append(run)
            run = 0.0
    return costs


def learning_ratio(results, frac=0.25):
    """
    The form-free version of the same claim, and the one to quote when the
    power-law specification is in doubt: what did the first stretch cost
    compared with the last?

    No functional form is assumed, so nothing can be wrong about it except
    the data.
    """
    n = len(results)
    k = max(2, int(n * frac))
    if n < 4 * k:
        return float("nan")
    early = np.mean([r.cost for r in results[:k]])
    late = np.mean([r.cost for r in results[-k:]])
    return float(early / max(1e-9, late))
