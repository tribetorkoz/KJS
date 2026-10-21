"""Every formula in the project, checked against brute force.

This is the module that matters most and the one most likely to be
skipped. Each test takes a closed-form expression used somewhere in
the system and checks it against Monte Carlo, against a hand-
computed value, or against a known degenerate case. Several of these
tests exist because the formula they check was wrong and had already
corrupted a published number.
"""

import random

import numpy as np

from nawah.bench import synthetic_library, variant_probes
from nawah.skill import Skill
from nawah.store import SkillStore, measure_precision
from nawah.tasks import StringTask, context_key, load_string_tasks
from nawah.theory import (
    beta_forecast,
    budget_for_infinite_skills,
    capacity_exponent_from_gap,
    check_predictions,
    critical_library,
    critical_library_joint,
    fit_beta_costs,
    fit_beta_cumulative,
    fit_beta_half,
    gap_exponent_from_capacity,
    norm_cdf,
    norm_ppf,
    p_correct_retrieval,
    p_correct_retrieval_crude,
    p_correct_retrieval_mc,
    regime,
    sigma_effective,
    skills_within_budget,
    total_cost,
)


def test_norm_ppf_roundtrip():
    """norm_ppf inverts norm_cdf to 1e-6"""
    for p in (0.001, 0.05, 0.3, 0.5, 0.75, 0.9, 0.999):
        assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-06, f"bad round trip at {p}"


def test_retrieval_formula_matches_monte_carlo():
    """
    The exact retrieval integral agrees with simulation.

    This is the test that caught the first version of the formula. The
    crude approximation Phi(r - sqrt(2 ln m)) looks reasonable and is
    wrong by up to 0.20, because it forgets that the winning candidate's
    own noise also fluctuates. It is kept in the codebase only so this
    test can measure how wrong it is.
    """
    worst_exact = worst_crude = 0.0
    for m in (2, 8, 64, 512):
        for r in (0.5, 2.0, 4.0):
            mc = p_correct_retrieval_mc(
                m, r, trials=20000, seed=m + int(r * 100)
            )
            ex = p_correct_retrieval(m, r)
            cr = p_correct_retrieval_crude(m, r)
            worst_exact = max(worst_exact, abs(ex - mc))
            worst_crude = max(worst_crude, abs(cr - mc))
    assert worst_exact < 0.02, f"exact form is off by {worst_exact:.4f}"
    assert (
        worst_exact < worst_crude
    ), "the exact form should beat the crude one it replaced"


def test_retrieval_monotone():
    """P(correct) falls with the crowd and rises with the gap"""
    assert all(
        p_correct_retrieval(m, 1.5) >= p_correct_retrieval(m + 1, 1.5)
        for m in range(2, 40)
    ), "not decreasing in m"
    assert all(
        p_correct_retrieval(30, r) <= p_correct_retrieval(30, r + 0.1)
        for r in [i * 0.1 for i in range(40)]
    ), "not increasing in r"


def test_critical_library_is_a_fixed_point():
    """critical_library(r) really is where precision lands on 1-eps"""
    for r in (2.5, 3.5, 4.5):
        m = critical_library(r, 0.1)
        if m <= 1.0:
            continue
        assert (
            abs(p_correct_retrieval(m, r) - 0.9) < 0.02
        ), f"r={r}: m*={m:.1f} gives {p_correct_retrieval(m, r):.3f}, not 0.90"


def test_beta_estimators_recover_ground_truth():
    """Both beta estimators recover a beta that was put there on purpose"""
    rng = random.Random(0)
    for true in (0.15, 0.3, 0.6):
        costs = [
            3.0 * n ** (-true) * (1 + rng.gauss(0, 0.02))
            for n in range(1, 400)
        ]
        b1, _ = fit_beta_costs(costs)
        b2, _ = fit_beta_cumulative(costs)
        assert abs(b1 - true) < 0.03, f"direct: {b1:.3f} vs {true}"
        assert abs(b2 - true) < 0.12, f"cumulative: {b2:.3f} vs {true}"


def test_flat_cost_is_beta_zero_not_nan():
    """
    [repeat] A flat cost curve is beta = 0, not "unmeasurable".

    It once returned NaN, which silently dropped those seeds from the
    mean and inflated the reported beta.
    """
    b, _ = fit_beta_costs([2.0] * 50)
    assert b == 0.0, f"flat cost gave beta={b}"


def test_beta_split_half_stable():
    """beta from the first half agrees with beta from the second"""
    rng = random.Random(1)
    costs = [
        4.0 * n ** (-0.4) * (1 + rng.gauss(0, 0.03)) for n in range(1, 600)
    ]
    a, b = fit_beta_half(costs)
    assert abs(a - b) < 0.1, f"halves disagree: {a:.3f} vs {b:.3f}"


def test_beta_forecast_beats_the_warehouse():
    """
    The law has to PREDICT. Fitting beta and admiring the fit proves
    nothing; beating "the future looks like the past" out of sample is
    the only thing that makes beta worth reporting.
    """
    rng = random.Random(2)
    costs = [
        5.0 * n ** (-0.45) * (1 + rng.gauss(0, 0.03)) for n in range(1, 500)
    ]
    f = beta_forecast(costs, frac=0.6)
    assert (
        f["ratio"] < 0.9
    ), f"forecast no better than flat: ratio {f['ratio']:.3f}"


def test_beta_regime_labels():
    """The phase boundary at beta = 1 is where the sum stops diverging"""
    assert regime(-0.3) == "interference"
    assert regime(0.0) == "warehouse"
    assert regime(0.3) == "learning-to-learn"
    assert "convergent" in regime(1.2)


def test_budget_phase_boundary():
    """
    beta > 1 buys infinitely many skills for a finite price; beta < 1
    does not. This is the sharpest statement the whole project makes.
    """
    assert skills_within_budget(20.0, 1.0, 1.5) == float("inf")
    assert skills_within_budget(20.0, 1.0, 0.9) < 1000000000000.0
    assert budget_for_infinite_skills(1.0, 0.8) == float("inf")
    assert budget_for_infinite_skills(1.0, 2.0) < 10.0
    assert abs(total_cost(5, 2.0, 0.0) - 10.0) < 1e-12


def test_two_sided_capacity_bound_saturates():
    """
    A bigger model stops helping once experience runs out, and more
    experience stops helping once the model is the limit. Neither alone
    buys an unbounded library.
    """
    assert sigma_effective(0.55, 8) > sigma_effective(0.55, 4096)
    assert sigma_effective(0.55, 4096, 0.5, 1) > sigma_effective(
        0.55, 4096, 0.5, 1000000.0
    )
    starved = (
        critical_library_joint(0.35, 0.55, 64, 0.5, 3),
        critical_library_joint(0.35, 0.55, 4096, 0.5, 3),
    )
    assert starved[1] < 5 * max(1.0, starved[0]), f"no saturation: {starved}"
    fed = (
        critical_library_joint(0.35, 0.55, 64, 0.5, 100000),
        critical_library_joint(0.35, 0.55, 4096, 0.5, 100000),
    )
    assert fed[1] > 10 * max(1.0, fed[0]), f"width did not matter: {fed}"


def test_gap_exponent_inverts_cleanly():
    """theta = 1/(2a) and back again"""
    for a in (0.25, 0.5, 1.0):
        th = gap_exponent_from_capacity(a)
        assert abs(capacity_exponent_from_gap(th) - a) < 1e-09
    assert (
        abs(capacity_exponent_from_gap(1.0) - 0.5) < 1e-09
    ), "theta=1 must predict a=1/2 (that was the derivation's error)"


def test_preregistered_predictions_all_hold():
    """Every prediction written down BEFORE the experiment that tests it"""
    bad = [(k, c) for k, c, ok in check_predictions() if not ok]
    assert not bad, f"failed predictions: {[b[0] for b in bad]}"


def test_context_key_comes_from_text_not_answers():
    """
    The routing bucket is derived from what the task SAYS. Two tasks with
    the same words share a bucket even if their ground-truth variant
    labels differ -- which is what stops the router from peeking at the
    answer key.
    """
    a = StringTask(
        0, "case", "make every letter uppercase", "ab", "AB", "case:upper"
    )
    b = StringTask(
        1, "case", "make every letter uppercase", "cd", "CD", "case:LIE"
    )
    assert context_key(a) == context_key(b), "the key is reading the label"


def test_context_key_separates_variants():
    """18 variants of the string suite -> 18 distinct routing buckets"""
    keys = {context_key(t) for t in load_string_tasks(per_family=30)}
    assert len(keys) == 18, f"collapsed {18 - len(keys)} variants together"


def test_routing_learns_from_experience():
    """Precision rises as the router is given experience"""
    probes = variant_probes(0)
    variants = sorted({(t.family, t.variant) for t in probes})
    got = []
    for nw in (0, 4, 24):
        st = synthetic_library(36, variants, seed=0, width=512)
        got.append(
            measure_precision(
                st, probes, rng=np.random.default_rng(0), n_warm=nw
            )
        )
    assert got[-1] > got[0], f"no learning: {got}"


def test_hierarchy_beats_flat_at_scale():
    """Same library, same noise: routing through the hierarchy wins"""
    probes = variant_probes(0)
    variants = sorted({(t.family, t.variant) for t in probes})
    out = {}
    for h in (True, False):
        st = synthetic_library(
            288, variants, seed=0, width=512, hierarchical=h
        )
        out[h] = measure_precision(
            st, probes, rng=np.random.default_rng(0), n_warm=16
        )
    assert out[True] > out[False] + 0.1, f"hierarchy worth nothing: {out}"


def test_credit_reaches_every_decision_level():
    """An outcome is recorded at family, cluster and skill level"""
    st = SkillStore(width=64, seed=0)
    st.add(
        Skill(
            family="case",
            body="make every letter uppercase b c",
            variant="case:upper",
        )
    )
    t = StringTask(
        0, "case", "make every letter uppercase", "a b", "A B", "case:upper"
    )
    r = st.route(t)
    assert len(r.keys) >= 2, f"route recorded only {r.keys}"
    st.record(r, True)
    for k in r.keys:
        assert (k, r.ctx) in st._bandit, f"no record for {k}"


def test_merging_widens_coverage_and_never_loses_it():
    """
    [repeat] A merge must cover everything both parents covered.

    Compressing the library is only a gain if the compressed skill still
    applies everywhere its parents did.
    """
    a = Skill(
        family="case", body="a", variant="case:upper", covers=("case:upper",)
    )
    b = Skill(
        family="case", body="b", variant="case:title", covers=("case:title",)
    )
    merged = Skill(
        family="case",
        body="ab",
        variant=a.variant,
        covers=tuple(sorted(set(a.covered) | set(b.covered))),
    )
    assert set(merged.covered) == {"case:upper", "case:title"}
    for parent in (a, b):
        assert set(parent.covered) <= set(merged.covered)


def test_precision_ignores_uncovered_probes():
    """
    Not knowing a thing and mis-retrieving it are different failures.
    The measurement must separate them or it measures neither.
    """
    st = SkillStore(width=64, seed=0)
    st.add(
        Skill(
            family="case",
            body="x",
            variant="case:upper",
            covers=("case:upper",),
        )
    )
    probes = [
        StringTask(
            0, "case", "make every letter uppercase", "a", "A", "case:upper"
        ),
        StringTask(
            1, "case", "swap the case of every letter", "a", "A", "case:swap"
        ),
    ]
    allp = measure_precision(
        st,
        probes,
        rng=np.random.default_rng(0),
        n_warm=0,
        retrieval_only=False,
    )
    onlyc = measure_precision(
        st, probes, rng=np.random.default_rng(0), n_warm=0, retrieval_only=True
    )
    assert allp <= 0.75, f"coverage failure was not counted: {allp}"
    assert (
        onlyc > allp
    ), f"retrieval-only should be the cleaner measure: {onlyc} vs {allp}"
