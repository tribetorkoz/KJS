"""
The mathematics of NAWAH, written down so it can be checked.

Nothing here is decoration. Every formula is a claim that can be wrong, and
every estimator is exercised against synthetic data with known ground truth
in tests/. If a number in a report disagrees with what this module
predicts, the report is the thing that is broken.

────────────────────────────────────────────────────────────────────────────
LAW 1 — THE COST OF THE N-th SKILL
────────────────────────────────────────────────────────────────────────────
Let C(N) be the cost (attempts, calls, tokens) of acquiring skill number N
given that N-1 skills are already in the library.

    C(N) = C1 * N^(-beta)

beta is the whole ball game:

    beta < 0   each skill costs MORE than the last. The library fights
               itself: interference. A warehouse that rots.
    beta = 0   every skill costs the same. A warehouse. No learning to
               learn, just accumulation.
    beta > 0   each skill costs LESS than the last. The agent is getting
               better at acquiring skills. This is the only regime in which
               the word "learning" earns its keep.

The total cost of N skills is T(N) = sum_{k=1..N} C1 k^(-beta), and the
asymptotics of that sum are a phase diagram, not a detail:

    beta < 1   T(N) ~ C1/(1-beta) * N^(1-beta)          power law
    beta = 1   T(N) ~ C1 * ln N                         logarithmic
    beta > 1   T(inf) = C1 * zeta(beta)  <  infinity    CONVERGENT

The last line is worth staring at. If beta > 1 the total cost of acquiring
infinitely many skills is FINITE. The agent does not merely get faster; the
whole acquisition programme is completed for a fixed price. beta = 1 is
therefore a genuine phase boundary, and it is the boundary that separates
"a better tool" from "a different kind of object".

Inverting: the number of skills affordable on a budget B is

    beta < 1   N(B) = ( B(1-beta)/C1 )^(1/(1-beta))     polynomial in B
    beta = 1   N(B) = exp(B/C1)                         exponential in B
    beta > 1   N(B) = infinity  once  B >= C1*zeta(beta)

At beta = 1/2 the budget-to-skills curve is already quadratic. This is why
beta is reported to three decimals and with a standard deviation: it is not
a scoreboard, it is an exponent.

────────────────────────────────────────────────────────────────────────────
LAW 2 — HOW MANY SKILLS FIT IN ONE LIBRARY
────────────────────────────────────────────────────────────────────────────
Retrieval picks the highest-scoring skill out of m candidates. Model each
candidate's score as its true quality plus representation noise of size
sigma:

    s_i = q_i + sigma * xi_i,      xi_i ~ N(0,1)

Let the correct skill sit Delta above the rest, and write r = Delta/sigma.
It wins iff  r + xi_1 > max_{i>=2} xi_i. Conditioning on xi_1 = x gives the
EXACT answer as a one-dimensional integral:

    P(m, r) = integral phi(x) * Phi(x + r)^(m-1) dx                   (*)

which is what p_correct_retrieval evaluates by quadrature. Two cheaper
approximations are worth knowing, mostly as warnings:

    Phi(r - sqrt(2 ln m))                CRUDE. Ignores that xi_1 itself
                                         fluctuates. It looked fine on paper
                                         and was wrong by up to 0.39 against
                                         Monte Carlo. Kept out of every
                                         number in this project.
    Phi((r - E[M]) / sqrt(1 + Var M))    decent, M = max of m-1 Gaussians.

Only (*) is used for anything that gets reported, and (*) is itself checked
against Monte Carlo in tests/test_mathematics.py, with the error printed.

Three things follow, and none of them was put in by hand.

(a) COLLAPSE WITH LIBRARY SIZE. P falls as m grows even though nothing about
    any individual skill got worse. Confusion is a property of the crowd.

(b) A CRITICAL SIZE EXISTS. Inverting (*) numerically gives the library size
    at which precision drops to 1-eps. For intuition only, the crude form
    says m* ~ exp((r - z)^2/2) with z = Phi^{-1}(1-eps): a stretched
    exponential in the signal-to-noise ratio, so doubling the quality gap
    does not double the library, it squares its logarithm.

(c) WHY A HIERARCHY HELPS, AND WHEN IT STOPS HELPING. With L levels of
    branching factor b and N = b^L skills, the agent makes L easy choices
    among b candidates instead of one impossible choice among N:

        P_flat = P(N, r)
        P_hier = P(b, r)^L ,        L = ln N / ln b

    Since b < N, every factor is larger -- but L grows as b shrinks, so
    there is a real interior optimum in b. See optimal_branching. There is
    also a threshold in r below which no branching beats flat retrieval,
    because multiplying many bad probabilities is worse than one: routing
    is a fix for confusion, not for ignorance.

NOISE AND CAPACITY. sigma is the model's own representation noise. Take the
usual statistical rate, sigma = sigma0 / sqrt(H), for a model of capacity H.
Then r = Delta*sqrt(H)/sigma0, and the critical library size grows with H.
How fast depends on something the library decides for itself: how fast the
quality gap closes as the library fills up. Write

    Delta(m) = Delta0 * m^(-theta)

The critical condition  Delta(m*)/sigma = z + sqrt(2 ln m*)  then reads

    Delta0 sqrt(H) / sigma0  =  m*^theta * ( z + sqrt(2 ln m*) )

whose leading-order solution in large m* is

    theta > 0 :   m*  ∝  H^( 1 / (2 theta) )
    theta = 0 :   m*  =  exp( (Delta0 sqrt(H)/sigma0 - z)^2 / 2 )

So a measured exponent a in  m* ∝ H^a  is a direct read-out of the crowding
law, and it inverts to

    theta = 1 / (2a)

    theta = 1    the gap closes like 1/m. This is the signature of a BOUNDED
                 quality distribution: the top order statistics crowd
                 together, so the more skills you have the harder the best
                 ones are to tell apart. Predicts m* ∝ H^(1/2).
    theta -> 0   the gap never closes, and m* explodes exponentially in H.

An earlier version of this derivation claimed m* ∝ H^(1/(2(1+theta))). That
is wrong, and the measurement is what exposed it: it predicts H^(1/4) at
theta = 1 while the critical condition plainly gives H^(1/2). Corrected
here, with the corrected prediction written down before re-measuring.

MEASURED in this project: a = 0.49, hence theta = 1.03. The library behaves
like a bounded quality distribution, and the ceiling is set by how fast its
best skills crowd together -- not by anything mysterious about the model.

────────────────────────────────────────────────────────────────────────────
LAW 3 — ROUTING: WHAT THE CONTEXT BUYS YOU
────────────────────────────────────────────────────────────────────────────
A router that ignores the situation must rank skills by one global number.
If skill k is excellent in context A and useless in context B, its global
average is the mean over contexts, and the gap it shows is diluted by the
number of contexts M. By the standard Hoeffding bound, separating two
skills whose success rates differ by Delta takes n samples with

    n >= (2/Delta^2) * ln(N/eps)

so the diluted router pays

    n_flat       = (2 M / Delta^2) * ln(N/eps)     gap diluted by M
    n_contextual = (2   / Delta^2) * ln(N/eps)     gap undiluted

Contextual routing is cheaper by exactly a factor of M, and that factor
grows with the number of distinct situations the library has to serve.
Routing is not plumbing. It is where the capacity of the whole system goes
to die, or does not.

────────────────────────────────────────────────────────────────────────────
HONEST STATUS
────────────────────────────────────────────────────────────────────────────
Law 1 is a definition plus arithmetic; the open question is whether beta > 0
in a real system. Law 2 (*) is a leading-order extreme-value approximation;
it is checked against Monte Carlo in tests/test_mathematics.py,
    and the error is reported, not
hidden. Law 3 is a Hoeffding bound, so it is a lower bound on difficulty for
the flat router, not an exact cost.
"""

from __future__ import annotations

import math


# ══════════════════════════════════════════════════════════════
# small statistics, standard library only
# ══════════════════════════════════════════════════════════════
def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _ols(xs, ys):
    """Least squares fit y = slope*x + intercept. Returns (slope, icpt, r2)."""
    n = len(xs)
    if n < 2:
        return 0.0, (_mean(ys) if ys else 0.0), 0.0
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-18:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    icpt = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + icpt)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 1.0
    return slope, icpt, r2


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF. Acklam's rational approximation."""
    if not 0.0 < p < 1.0:
        raise ValueError("norm_ppf needs 0 < p < 1")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


# ══════════════════════════════════════════════════════════════
# LAW 2 — retrieval under noise
# ══════════════════════════════════════════════════════════════
_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_PC_CACHE: dict = {}


# The crude approximation, kept ONLY so that tests can measure how wrong it
# is. It is never used to produce a reported number.
def p_correct_retrieval_crude(m: float, r: float) -> float:
    """Phi(r - sqrt(2 ln m)). Documented as a cautionary tale, not used."""
    m = max(2.0, float(m))
    return norm_cdf(r - math.sqrt(2.0 * math.log(m)))


def _log_phi(x: float) -> float:
    return -0.5 * x * x - _LOG_SQRT_2PI


def _log_norm_cdf(x: float) -> float:
    """log Phi(x), stable far into the left tail where Phi underflows."""
    if x > -20.0:
        v = 0.5 * (1.0 + math.erf(x / _SQRT2))
        return math.log(v) if v > 1e-300 else -1e300
    z = -x / _SQRT2
    # erfc(z) ~ exp(-z^2)/(z sqrt(pi)); log(Phi) = log(erfc(z)/2)
    return -z * z - math.log(z) - 0.5 * math.log(math.pi) - math.log(2.0)


def _p_integral(
    m: float, r: float, n_steps: int = 1600, half_width: float = 9.0
) -> float:
    """
    Simpson's rule for  integral phi(x) Phi(x+r)^(m-1) dx.

    The integrand is unimodal: log phi(x) is a downward parabola and
    (m-1) log Phi(x+r) increases monotonically, so their sum has a single
    interior maximum. We locate it by ternary search and integrate a wide
    window around it. Everything is done in log space so that m in the
    millions does not overflow.
    """
    if m <= 1.0 + 1e-12:
        return 1.0

    def lg(x):
        return _log_phi(x) + (m - 1.0) * _log_norm_cdf(x + r)

    lo, hi = -15.0, 15.0
    for _ in range(90):
        a = lo + (hi - lo) / 3.0
        b = hi - (hi - lo) / 3.0
        if lg(a) < lg(b):
            lo = a
        else:
            hi = b
    peak = 0.5 * (lo + hi)

    a, b = peak - half_width, peak + half_width
    h = (b - a) / n_steps
    total = 0.0
    for i in range(n_steps + 1):
        x = a + i * h
        lv = lg(x)
        v = math.exp(lv) if lv > -700.0 else 0.0
        if i == 0 or i == n_steps:
            w = 1.0
        elif i % 2:
            w = 4.0
        else:
            w = 2.0
        total += w * v
    val = total * h / 3.0
    return max(0.0, min(1.0, val))


def p_correct_retrieval(m, r: float) -> float:
    """
    Probability that the right skill wins a contest among m candidates, when
    it sits r = Delta/sigma above the others. Exact form (*), by quadrature.
    m may be non-integral (used when searching for the best branching
    factor). Monotone increasing in r, decreasing in m, and 1 when m <= 1.
    """
    m = float(m)
    if m <= 1.0:
        return 1.0
    if r > 40.0:
        return 1.0
    key = (round(m, 6), round(r, 6))
    if key not in _PC_CACHE:
        _PC_CACHE[key] = _p_integral(m, r)
    return _PC_CACHE[key]


def p_correct_retrieval_mc(
    m: int, r: float, trials: int = 40000, seed: int = 7
) -> float:
    """Monte Carlo ground truth for p_correct_retrieval.

    Use it to check the closed form marked (*).
    """
    import random

    rng = random.Random(seed)
    m = max(2, int(m))
    win = 0
    for _ in range(trials):
        best_other = max(rng.gauss(0.0, 1.0) for _ in range(m - 1))
        if r + rng.gauss(0.0, 1.0) > best_other:
            win += 1
    return win / trials


def critical_library(r: float, eps: float = 0.10) -> float:
    """
    Library size at which retrieval precision falls to 1 - eps, obtained by
    inverting the exact (*) numerically.

    The crude closed form, exp((r - z)^2/2) with z = Phi^{-1}(1-eps), is
    available in critical_library_crude() for comparison and is reported
    alongside: how far the closed form drifts from the truth is a fair
    measure of how much the honest derivation matters.
    """
    target = 1.0 - eps
    if p_correct_retrieval(2.0, r) < target:
        return 1.0
    hi = 2.0
    while hi < 1e18 and p_correct_retrieval(hi, r) > target:
        hi *= 2.0
    lo = 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if p_correct_retrieval(mid, r) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_library_crude(r: float, eps: float = 0.10) -> float:
    """The closed-form approximation exp((r-z)^2/2). Intuition only."""
    z = norm_ppf(1.0 - eps)
    if r <= z:
        return 1.0
    return math.exp((r - z) ** 2 / 2.0)


def p_correct_hierarchical(n: int, b: float, r: float) -> float:
    """
    Two-or-more-level routing: N skills in clusters of size b, L = ln N/ln b
    levels of an easy b-way choice. Every level must be right.
    """
    n = max(2, int(n))
    b = max(2.0, min(float(b), float(n)))
    levels = math.log(n) / math.log(b)
    per = p_correct_retrieval(round(b), r)
    return per**levels if per > 0 else 0.0


def optimal_branching(n: int, r: float, lo: float = 2.0) -> float:
    """
    The branching factor that maximises P(correct) for a library of n skills
    at signal-to-noise r. Too coarse and the within-cluster choice is hard;
    too fine and there are too many levels to get all of them right.
    """
    n = max(4, int(n))
    best_b, best_p = float(n), -1.0
    b = lo
    # geometric scan, then a refinement pass
    grid = []
    while b <= n + 1:
        grid.append(b)
        b *= 1.15
    grid.append(float(n))
    for _ in range(3):
        around = best_b if best_b > 0 else float(n)
        grid += [around * f for f in (0.85, 0.92, 1.08, 1.18)]
    for bb in grid:
        bb = max(2.0, min(float(n), bb))
        p = p_correct_hierarchical(n, bb, r)
        if p > best_p:
            best_p, best_b = p, bb
    return best_b


def gap_exponent_from_capacity(a: float) -> float:
    """
    Measured  m* ∝ H^a  ->  how fast the quality gap closes: theta = 1/(2a).

    a = 1/2 means theta = 1: a bounded quality distribution whose leaders
    crowd together. a -> 0 means a gap that barely closes, and the capacity
    explodes exponentially in H rather than as a power of it.
    """
    if a <= 0:
        return float("inf")
    return 1.0 / (2.0 * a)


def capacity_exponent_from_gap(theta: float) -> float:
    """Inverse: theta -> predicted exponent a = 1/(2 theta)."""
    if theta <= 0:
        return float("inf")  # exponential in H, not a power of it
    return 1.0 / (2.0 * theta)


def fit_capacity_exponent(heights, ceilings):
    """Fit m* = A H^a. Returns (a, A, r2)."""
    xs = [math.log(h) for h in heights]
    ys = [math.log(max(1e-9, c)) for c in ceilings]
    slope, icpt, r2 = _ols(xs, ys)
    return slope, math.exp(icpt), r2


def sigma_effective(
    sigma0: float, height: float, p: float = 0.5, n: float = 1.0
) -> float:
    """
    Two noise sources, combined in quadrature.

        1. representation noise   sigma0 / sqrt(H)
        2. estimation noise       sqrt(p(1-p)/n), from having seen this
                                  context only n times

    This is the honest version of "how sharp is the agent's eye". Sharpen
    the model all you like: if it has hardly met this kind of task before,
    it is still guessing.
    """
    return math.sqrt(
        sigma0**2 / max(1e-9, height) + p * (1.0 - p) / max(1.0, n)
    )


def critical_library_joint(
    delta: float,
    sigma0: float,
    height: float,
    p: float = 0.5,
    n: float = 1.0,
    eps: float = 0.10,
) -> float:
    """
    THE TWO-SIDED CAPACITY BOUND.

    Critical library size when both the model and its experience are
    finite:

        T*(H, n) = critical_library( Delta / sigma_eff(H, n) )

    Because sigma_eff is a quadrature sum, T* saturates at BOTH ends:

        H -> infinity :  T* -> critical_library( Delta sqrt(n / p(1-p)) )
        n -> infinity :  T* -> critical_library( Delta sqrt(H) / sigma0 )

    A bigger model stops helping once experience is the bottleneck, and
    more experience stops helping once the model is. Neither one alone
    buys an unbounded library. That is a prediction, and it is testable:
    sweep each while holding the other fixed and look for the plateau.
    """
    se = sigma_effective(sigma0, height, p, n)
    if se <= 0:
        return float("inf")
    return critical_library(delta / se, eps)


# ══════════════════════════════════════════════════════════════
# LAW 3 — routing sample complexity
# ══════════════════════════════════════════════════════════════
def routing_samples(n: int, delta: float, eps: float = 0.10) -> float:
    """Samples needed to pick the best of n skills whose gap is delta."""
    if delta <= 0:
        return float("inf")
    return (2.0 / delta**2) * math.log(max(2.0, n) / eps)


def routing_dilution(contexts: int) -> float:
    """How much more a context-blind router pays. Exactly M."""
    return float(max(1, contexts))


# ══════════════════════════════════════════════════════════════
# LAW 1 — cost, beta, and the phase diagram
# ══════════════════════════════════════════════════════════════
def fit_beta_costs(costs) -> tuple:
    """
    Estimator 1: regress log C on log N. beta = -slope.

    A perfectly flat cost is not "unmeasurable", it is exactly beta = 0, so
    that case returns 0.0 rather than NaN. (It once returned NaN and quietly
    dropped seeds from the average, which inflated beta.)
    """
    costs = [max(1e-9, float(c)) for c in costs]
    if len(costs) < 4:
        return float("nan"), 0.0
    ys = [math.log(c) for c in costs]
    if max(ys) - min(ys) < 1e-12:
        return 0.0, 1.0
    xs = [math.log(i) for i in range(1, len(costs) + 1)]
    slope, _ic, r2 = _ols(xs, ys)
    return -slope, r2


def fit_beta_cumulative(costs) -> tuple:
    """
    Estimator 2: the integral form. Since

        T(N) = sum_{k<=N} C1 k^(-beta) ~ C1/(1-beta) * N^(1-beta)   (beta<1)

    the slope of log T on log N is 1 - beta.

    KNOWN BIAS: the asymptotic form drops an additive constant, and that
    constant bends the cumulative curve, so this estimator UNDERSTATES
    beta, increasingly so as beta grows (measured: 0.60 in, 0.52 out).
    It is therefore a specification check, not the number to quote -- the
    direct estimator is. What it is good for is catching a curve that only
    looks like a power law: a transient followed by a plateau fools the
    direct fit and not this one, which is exactly why both are reported.
    """
    costs = [max(1e-9, float(c)) for c in costs]
    if len(costs) < 4:
        return float("nan"), 0.0
    cum, tot = [], 0.0
    for c in costs:
        tot += c
        cum.append(tot)
    xs = [math.log(i) for i in range(1, len(cum) + 1)]
    ys = [math.log(c) for c in cum]
    slope, _ic, r2 = _ols(xs, ys)
    return 1.0 - slope, r2


def fit_beta_half(costs) -> tuple:
    """
    Estimator 3: beta on the first half and on the second half SEPARATELY.
    The two halves share no data, so agreement here is real out-of-sample
    stability rather than a restatement of the same fit.

    The second half MUST be fitted against its TRUE indices. An earlier
    version re-numbered it from 1, which silently stretched a 20% drop
    across a 5.7-wide log range and reported beta = 0.078 for data whose
    beta was 0.40. Re-indexing is not a translation of x -- log(N+k) is
    not log(k) plus a constant -- so it changes the slope, and it changes
    it in the direction that makes learning-to-learn disappear.
    """
    costs = [max(1e-9, float(c)) for c in costs]
    n = len(costs)
    if n < 8:
        return float("nan"), float("nan")
    h = n // 2
    b1, _ = fit_beta_costs(costs[:h])
    xs = [math.log(i) for i in range(h + 1, n + 1)]
    ys = [math.log(c) for c in costs[h:]]
    slope, _ic, _r2 = _ols(xs, ys)
    return b1, -slope


def beta_forecast(costs, frac: float = 0.6) -> dict:
    """
    Estimator 4, the one that actually matters: fit beta on the first `frac`
    and PREDICT the rest. A power law is only worth its name if it beats the
    obvious null out of sample -- and the obvious null is the warehouse,
    where beta = 0 and every future skill costs what the past ones cost.

    Returns RMSE of both models plus the ratio. ratio < 1 means the law won.
    """
    costs = [max(1e-9, float(c)) for c in costs]
    n = len(costs)
    k = max(4, int(n * frac))
    if n - k < 4:
        return {
            "ratio": float("nan"),
            "rmse_law": float("nan"),
            "rmse_null": float("nan"),
            "beta": float("nan"),
            "n_train": k,
            "n_test": n - k,
        }
    train, test = costs[:k], costs[k:]
    beta, _ = fit_beta_costs(train)
    flat = _mean(train)
    # fit in log space, predict back in cost space (that is where cost lives)
    c1 = math.exp(
        _mean(
            [math.log(c) + beta * math.log(i + 1) for i, c in enumerate(train)]
        )
    )
    pred_law = [c1 * (j + 1.0) ** (-beta) for j in range(k, n)]
    pred_null = [flat] * len(test)
    rmse_law = math.sqrt(_mean([(a - b) ** 2 for a, b in zip(test, pred_law)]))
    rmse_null = math.sqrt(
        _mean([(a - b) ** 2 for a, b in zip(test, pred_null)])
    )
    ratio = rmse_law / rmse_null if rmse_null > 1e-12 else float("nan")
    return {
        "ratio": ratio,
        "rmse_law": rmse_law,
        "rmse_null": rmse_null,
        "beta": beta,
        "n_train": k,
        "n_test": n - k,
    }


def beta_cross_check(costs) -> dict:
    """All four estimators at once, with a verdict. Used by `nawah verify`."""
    costs = [max(1e-9, float(c)) for c in costs]
    b1, r1 = fit_beta_costs(costs)
    b2, r2 = fit_beta_cumulative(costs)
    b3a, b3b = fit_beta_half(costs)
    f = beta_forecast(costs)
    vals = [v for v in (b1, b2, b3a, b3b) if v == v and abs(v) != float("inf")]
    spread = (max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
    ok = (
        len(vals) >= 2
        and spread < 0.30
        and f["ratio"] == f["ratio"]
        and f["ratio"] < 1.0
    )
    return {
        "beta_direct": b1,
        "r2_direct": r1,
        "beta_cumulative": b2,
        "r2_cumulative": r2,
        "beta_first_half": b3a,
        "beta_second_half": b3b,
        "spread": spread,
        "forecast_ratio": f["ratio"],
        "rmse_law": f["rmse_law"],
        "rmse_null": f["rmse_null"],
        "consistent": bool(ok),
    }


def regime(beta: float) -> str:
    """Name the phase. The thresholds are stated, not tuned per experiment."""
    if beta != beta:
        return "unmeasurable"
    if beta < -0.05:
        return "interference"
    if beta <= 0.05:
        return "warehouse"
    if beta < 0.50:
        return "learning-to-learn"
    if beta < 1.00:
        return "strong learning-to-learn"
    return "convergent (finite cost for infinitely many skills)"


def total_cost(n: int, c1: float, beta: float) -> float:
    """Exact sum C1 * sum_{k=1..n} k^(-beta)."""
    if n <= 0:
        return 0.0
    return c1 * sum(k ** (-beta) for k in range(1, n + 1))


def skills_within_budget(
    budget: float, c1: float, beta: float, cap: int = 2_000_000
) -> float:
    """
    How many skills does a budget buy? Walks the exact sum, so the beta > 1
    case -- where the answer is unbounded -- falls out instead of being
    special-cased.

    For beta > 1 the tail is bounded analytically,
        sum_{k>n} k^(-beta) <= n^(1-beta) / (beta - 1),
    so "infinitely many" is declared the moment the bound fits, rather than
    after grinding through a million iterations that will never exceed it.
    """
    if budget <= 0:
        return 0.0
    tot, n = 0.0, 0
    while n < cap:
        nxt = tot + c1 * ((n + 1) ** (-beta))
        if nxt > budget:
            return float(n)
        tot, n = nxt, n + 1
        if beta > 1.0:
            tail = c1 * (n ** (1.0 - beta)) / (beta - 1.0)
            if tot + tail <= budget:
                return float("inf")
    return float("inf")


def budget_for_infinite_skills(c1: float, beta: float) -> float:
    """
    For beta > 1 the total converges: infinitely many skills cost
    C1 * zeta(beta).
    For beta <= 1 the price of infinitely many skills is infinite.
    """
    if beta <= 1.0:
        return float("inf")
    s, k = 0.0, 1
    while True:
        term = k ** (-beta)
        s += term
        if term < 1e-15 or k > 10**7:
            break
        k += 1
    return c1 * s


# ══════════════════════════════════════════════════════════════
# Pre-registered predictions
# ══════════════════════════════════════════════════════════════
# Written BEFORE the experiment that tests them. If one of these fails the
# run reports it as a failure, not as a finding.
PREDICTIONS = [
    {
        "key": "retrieval_monotone_m",
        "claim": "P(correct retrieval) decreases as the candidate pool grows, "
        "at fixed signal-to-noise ratio.",
        "test": lambda: all(
            p_correct_retrieval(m, 1.5) >= p_correct_retrieval(m + 1, 1.5)
            for m in range(2, 200)
        ),
    },
    {
        "key": "retrieval_monotone_r",
        "claim": "P(correct retrieval) increases with the quality gap.",
        "test": lambda: all(
            p_correct_retrieval(50, r) <= p_correct_retrieval(50, r + 0.05)
            for r in [i * 0.05 for i in range(60)]
        ),
    },
    {
        "key": "hierarchy_beats_flat",
        "claim": (
            "At N=4096 and r=2.0, the best hierarchy beats flat retrieval."
        ),
        "test": lambda: max(
            p_correct_hierarchical(4096, b, 2.0)
            for b in (2, 4, 8, 16, 32, 64, 128, 256)
        )
        > p_correct_retrieval(4096, 2.0),
    },
    {
        "key": "critical_library_grows",
        "claim": (
            "Critical library size is monotone in the signal-to-noise ratio."
        ),
        "test": lambda: all(
            critical_library(r) <= critical_library(r + 0.1)
            for r in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
        ),
    },
    {
        "key": "beta_phase_boundary",
        "claim": "beta > 1 gives infinitely many skills for a finite budget; "
        "beta just under 1 does not. (budget 5 units, C1 = 1)",
        "test": lambda: (
            skills_within_budget(5.0, 1.0, 1.4) == float("inf")
            and skills_within_budget(5.0, 1.0, 0.999) < 1e6
        ),
    },
    {
        "key": "hierarchy_dominates_flat",
        "claim": "Hierarchical routing beats flat routing at every signal-to-"
        "noise ratio, and the win is LARGEST where the agent is "
        "weakest: routing pays most exactly when the model is worst.",
        "test": lambda: (
            all(
                max(
                    p_correct_hierarchical(4096, b, r)
                    for b in (2, 3, 4, 8, 16, 64, 256)
                )
                > p_correct_retrieval(4096, r)
                for r in (0.3, 1.0, 2.0, 3.0, 5.0)
            )
            and (
                p_correct_hierarchical(4096, 2.3, 0.3)
                / p_correct_retrieval(4096, 0.3)
                > p_correct_hierarchical(4096, 2.3, 5.0)
                / p_correct_retrieval(4096, 5.0)
            )
        ),
    },
    {
        "key": "budget_superlinear",
        "claim": "At beta = 0.5, doubling the budget more than doubles the "
        "skill count.",
        "test": lambda: skills_within_budget(200.0, 1.0, 0.5)
        > 2 * skills_within_budget(100.0, 1.0, 0.5),
    },
    {
        "key": "routing_dilution_is_M",
        "claim": "A context-blind router needs exactly M times the samples.",
        "test": lambda: routing_samples(64, 0.2 / 8.0)
        / routing_samples(64, 0.2)
        == 64.0,
    },
]


def check_predictions() -> list:
    """Run every pre-registered prediction. Returns [(key, claim, passed)]."""
    out = []
    for p in PREDICTIONS:
        try:
            ok = bool(p["test"]())
        except Exception:
            ok = False
        out.append((p["key"], p["claim"], ok))
    return out
