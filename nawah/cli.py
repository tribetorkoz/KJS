"""Command line interface:  python3 -m nawah.cli bench --backend sim"""

import argparse
import json
import os
import sys
import urllib.request


def _run(mk, curr, probes, a, kw, bench):
    head = bench.headline(mk, curr, probes, seeds=range(a.seeds), **kw)
    abl = bench.ablation(
        mk, curr, probes, seeds=range(max(2, a.seeds // 2)), **kw
    )
    tr = bench.transfer(mk, curr, probes, seeds=(0,), **kw)
    cap = bench.capacity_sweep(mk, curr, probes, seeds=(0,), **kw)
    md = bench.report_md(head, abl, tr, cap)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\n-> saved to {a.out}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="nawah", description="NAWAH — a layer over any model"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="Run the full experiment harness")
    b.add_argument(
        "--backend",
        default="auto",
        choices=[
            "sim",
            "anthropic",
            "openai",
            "ollama",
            "groq",
            "openrouter",
            "together",
            "nvidia",
            "auto",
        ],
    )
    b.add_argument("--model", default=None)
    b.add_argument("--suite", default="strings", choices=["strings", "code"])
    b.add_argument("--per-family", type=int, default=48)
    b.add_argument("--seeds", type=int, default=8)
    b.add_argument("--k-max", type=int, default=4)
    b.add_argument("--width", type=int, default=64)
    b.add_argument("--out", default="report.md")
    b.add_argument(
        "--budget",
        type=float,
        default=None,
        help="maximum USD to spend before stopping",
    )
    b.add_argument("--max-calls", type=int, default=None)
    b.add_argument("--max-tokens", type=int, default=None)
    b.add_argument(
        "--estimate",
        action="store_true",
        help="estimate cost only, do not run",
    )

    e = sub.add_parser("export", help="Export skills to files")
    e.add_argument("--dir", default="skills")
    e.add_argument("--from", dest="src", default=None)

    pf = sub.add_parser("probe", help="One call to test the connection first")
    pf.add_argument(
        "--backend",
        default="nvidia",
        choices=[
            "anthropic",
            "openai",
            "ollama",
            "groq",
            "openrouter",
            "together",
            "nvidia",
        ],
    )
    pf.add_argument("--model", default=None)
    pf.add_argument("--suite", default="code", choices=["code", "strings"])

    lm = sub.add_parser("models", help="List models available on the service")
    lm.add_argument(
        "--backend",
        default="nvidia",
        choices=[
            "anthropic",
            "openai",
            "ollama",
            "groq",
            "openrouter",
            "together",
            "nvidia",
        ],
    )
    lm.add_argument("--grep", default=None, help="filter by name")

    i = sub.add_parser("inspect", help="Show saved skills")
    i.add_argument("--dir", default="skills")

    v = sub.add_parser("verify", help="Re-check every claim the project makes")
    v.add_argument("--seeds", type=int, default=6)
    v.add_argument("--per-family", type=int, default=36)

    sub.add_parser("theory", help="Print the laws and their numbers")

    cp = sub.add_parser(
        "compose", help="Composition: chain skills never chained before"
    )
    cp.add_argument("--per-op", type=int, default=32)
    cp.add_argument("--seeds", type=int, default=4)
    cp.add_argument(
        "--scaling",
        action="store_true",
        help="show how systematicity grows with the library",
    )
    cp.add_argument("--out", default=None)

    cu = sub.add_parser("curriculum", help="Who chooses what to practise")
    cu.add_argument("--budget", type=int, default=200)
    cu.add_argument("--seeds", type=int, default=6)
    cu.add_argument("--out", default=None)

    ca = sub.add_parser("capacity", help="Measure the ceiling, assume nothing")
    ca.add_argument("--widths", default="32,128,512,2048")
    ca.add_argument("--n-warm", type=int, default=20)
    ca.add_argument("--thresh", type=float, default=0.85)
    ca.add_argument("--seeds", type=int, default=3)
    ca.add_argument("--n-max", type=int, default=1024)
    ca.add_argument(
        "--flat",
        action="store_true",
        help="also measure a flat router for comparison",
    )

    ro = sub.add_parser("routing", help="What the context is worth")
    ro.add_argument("--contexts", default="4,9,18,36,72")
    ro.add_argument("--budget", type=int, default=576)

    a = p.parse_args(argv)

    if a.cmd == "bench":
        from . import bench
        from .backend import PRICES, BudgetExceeded, estimate
        from .backend import make_backend as _mb
        from .tasks import load_code_tasks, load_string_tasks

        tasks = (
            load_code_tasks()
            if a.suite == "code"
            else load_string_tasks(per_family=a.per_family)
        )
        # the code suite is small (16 tasks): one probe per family
        rest, probes = bench.split_tasks(
            tasks, n_probe_per_family=1 if a.suite == "code" else 6
        )
        curr = bench.interleave(rest)

        n_cur = len(bench.interleave(rest))
        if a.suite == "code" and n_cur < 8:
            print(
                f"! code suite is only {n_cur} tasks: qualitative, not "
                f"statistical. Extend CODE_TASKS in nawah/tasks.py"
            )

        if a.suite == "code" and a.backend == "sim":
            print(
                "! the simulator cannot write code. The code suite is for "
                "real models:\n"
                "  python3 -m nawah.cli bench --backend nvidia --suite code"
            )
            return 0

        price = PRICES.get(a.backend)
        est = estimate(
            len(curr),
            a.k_max,
            6 if a.suite != "code" else 1,
            n_families=7 if a.suite == "code" else 6,
            price=price,
        )
        print(
            f"Tasks: {len(curr)}  |  Backend: {a.backend}  |  "
            f"Attempts: {a.k_max}"
        )
        print(
            f"Estimate: {est['calls']:,} calls · {est['tokens']:,} tokens"
            + (f" · ~${est['usd']:.2f}" if "usd" in est else "")
        )
        if a.estimate:
            print("(--estimate: nothing was run)")
            return 0
        if a.budget and "usd" in est and est["usd"] > a.budget:
            print(
                f"! estimate ${est['usd']:.2f} exceeds your budget "
                f"${a.budget:.2f}."
            )
            print("  lower --seeds, --per-family or --k-max")
            return 1

        base = {"model": a.model} if a.model else {}

        def mk(seed=0, **x):
            bk = _mb(a.backend, **{**base, **x, "seed": seed})
            if any(
                v is not None for v in (a.budget, a.max_calls, a.max_tokens)
            ):
                bk.set_budget(
                    usd=a.budget, calls=a.max_calls, tokens=a.max_tokens
                )
            return bk

        kw = {"k_max": a.k_max, "width": a.width}
        try:
            return _run(mk, curr, probes, a, kw, bench)
        except BudgetExceeded as e:
            print(f"\n[stopped] {e}")
            print("  that is the budget guard — your trial key is intact.")
            return 0

    elif a.cmd == "probe":
        from .backend import make_backend
        from .tasks import load_code_tasks, load_string_tasks

        kw = {"model": a.model} if a.model else {}
        try:
            b = make_backend(a.backend, **kw)
        except Exception as e:
            print(f"! could not build the backend: {e}")
            return 1
        tasks = (
            load_code_tasks()
            if a.suite == "code"
            else load_string_tasks(per_family=2)
        )
        t = tasks[0]
        print(f"Backend: {b.name}  |  Model: {getattr(b, 'model', '?')}")
        print(f"Task: {t.family} — {t.desc}")
        print("-" * 60)
        try:
            b.begin_task()
            raw = b.attempt(t, None, decorrelate=False)
        except Exception as e:
            print(f"! call failed: {type(e).__name__}: {e}")
            print("  check the key, the base URL and your connection.")
            return 1
        print(f"Raw output (first 300 chars):\n{str(raw)[:300]}")
        print("-" * 60)
        print(f"Success: {'yes' if t.verify(raw) else 'no'}")
        print(f"Cost: {b.cost}")
        return 0

    elif a.cmd == "models":
        from .backend import ENDPOINTS, OllamaBackend

        if a.backend == "ollama":
            host = OllamaBackend().host
            with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as r:
                ids = [
                    m["name"] for m in json.loads(r.read()).get("models", [])
                ]
        else:
            base = ENDPOINTS[a.backend][1]
            key = os.environ.get(ENDPOINTS[a.backend][2])
            if not key:
                print(f"! {ENDPOINTS[a.backend][2]} not set")
                return 1
            rq = urllib.request.Request(
                f"{base}/models", headers={"Authorization": f"Bearer {key}"}
            )
            with urllib.request.urlopen(rq, timeout=20) as r:
                ids = sorted(
                    m["id"] for m in json.loads(r.read()).get("data", [])
                )
        if a.grep:
            ids = [m for m in ids if a.grep.lower() in m.lower()]
        print(
            f"{len(ids)} models" + (f" (filter: {a.grep})" if a.grep else "")
        )
        for m in ids[:60]:
            print("  " + m)
        if len(ids) > 60:
            print(f"  ... and {len(ids) - 60} more")
        return 0

    elif a.cmd == "verify":

        from . import bench as _bench
        from .agent import NawahAgent
        from .backend import SimBackend
        from .tasks import load_string_tasks
        from .theory import (
            beta_cross_check,
            check_predictions,
            p_correct_retrieval,
            p_correct_retrieval_mc,
        )

        ok = True
        print("=" * 68)
        print("  NAWAH verify -- every claim, re-checked")
        print("=" * 68)
        print("\n1. Predictions registered before the experiments")
        for key, _claim, passed in check_predictions():
            print(f"   [{'ok ' if passed else 'FAIL'}] {key}")
            ok = ok and passed

        print("\n2. The retrieval formula against simulation")
        worst = 0.0
        for m in (2, 16, 128, 1024):
            for r in (0.5, 2.0, 4.0):
                mc = p_correct_retrieval_mc(
                    m, r, trials=15000, seed=m + int(r * 100)
                )
                worst = max(worst, abs(p_correct_retrieval(m, r) - mc))
        good = worst < 0.02
        print(
            f"   [{'ok ' if good else 'FAIL'}] worst error {worst:.4f} "
            f"(tolerance 0.02)"
        )
        ok = ok and good

        print("\n3. Efficiency: the claim, and the test on it")
        tasks = load_string_tasks(per_family=a.per_family)
        rest, probes = _bench.split_tasks(tasks, n_probe_per_family=6)
        curr = _bench.interleave(rest)
        lrs, bss, consistent = [], [], 0
        for s0 in range(a.seeds):
            ag = NawahAgent(SimBackend(0.10, seed=s0), seed=s0)
            ag.run(curr, probes)
            lrs.append(_bench.learning_ratio(ag.results))
            sc = _bench.skill_acquisition_costs(ag)
            if len(sc) >= 8:
                bss.append(_bench.fit_beta_costs(sc)[0])
            consistent += bool(
                beta_cross_check([r.cost for r in ag.results])["consistent"]
            )
        need = max(1, int(0.75 * a.seeds))
        n_lr = sum(1 for x in lrs if x == x and x > 1.0)
        n_bs = sum(1 for x in bss if x > 0)
        g1 = n_lr >= need
        g2 = n_bs >= need
        mean_lr = sum(x for x in lrs if x == x) / max(
            1, sum(1 for x in lrs if x == x)
        )
        mean_bs = sum(bss) / max(1, len(bss))
        print(
            f"   [{'ok ' if g1 else 'FAIL'}] cost falls from the first "
            f"quarter to the last on {n_lr}/{a.seeds} seeds "
            f"(mean x{mean_lr:.2f})"
        )
        print(
            f"   [{'ok ' if g2 else 'FAIL'}] beta on cost per acquired "
            f"skill is positive on {n_bs}/{len(bss)} seeds "
            f"(mean {mean_bs:+.3f})"
        )
        ok = ok and g1 and g2
        print(
            f"   [note] the cost curve passes the four-estimator "
            f"power-law cross-check on {consistent}/{a.seeds} seeds."
        )
        print("          the exponent is therefore reported as indicative,")
        print("          and the form-free ratio is the claim.")
        print("\n" + "=" * 68)
        print("  ALL CHECKS PASSED" if ok else "  SOME CHECKS FAILED")
        print("=" * 68)
        return 0 if ok else 1

    elif a.cmd == "theory":
        from .theory import (
            critical_library,
            critical_library_joint,
            optimal_branching,
            p_correct_hierarchical,
            p_correct_retrieval,
            regime,
            skills_within_budget,
        )

        print("LAW 1  cost of skill N  =  C1 * N^(-beta)")
        print("       total cost behaves as a phase diagram:")
        for b in (0.0, 0.5, 0.9, 1.0, 1.4):
            n100 = skills_within_budget(20.0, 1.0, b)
            print(
                f"         beta={b:<4} {regime(b):<48} "
                f"skills on a budget of 20:   "
                f"{'infinite' if n100 == float('inf') else f'{n100:.0f}'}"
            )
        print(
            "\nLAW 2  P(correct) = int phi(x) Phi(x+r)^(m-1) dx,"
            " r = gap/noise"
        )
        print("        library    gap r=2.0   r=3.0     r=4.0     r=5.0")
        for m in (8, 64, 512, 4096):
            row = "  ".join(
                f"{p_correct_retrieval(m, r):.3f}"
                for r in (2.0, 3.0, 4.0, 5.0)
            )
            print(f"        {m:<8}  {row}")
        print("\n        critical library size m* at precision 0.90:")
        print(
            "        "
            + "  ".join(
                f"r={r}:{critical_library(r):.0f}"
                for r in (2.0, 3.0, 4.0, 5.0, 6.0)
            )
        )
        print("\n        best branching factor for N=4096:")
        for r in (1.0, 2.0, 3.0, 5.0):
            b = optimal_branching(4096, r)
            print(
                f"          r={r:<4} b*={b:7.2f}  "
                f"hier {p_correct_hierarchical(4096, b, r):.3f}  "
                f"flat {p_correct_retrieval(4096, r):.3f}"
            )
        print(
            "\nLAW 3  two-sided capacity bound, T*(H, n), delta=0.35:"
            "\n        (rows = experience per context, cols = model width)"
        )
        print("            " + "".join(f"H={h:<10}" for h in (8, 128, 2048)))
        for n in (5, 50, 5000):
            row = "".join(
                f"{critical_library_joint(0.35, 0.55, h, 0.5, n):<12.0f}"
                for h in (8, 128, 2048)
            )
            print(f"        n={n:<6}  {row}")
        print(
            "\n        note how each row saturates going right, and each"
            "\n        column saturates going down: neither the model nor"
            "\n        the experience alone buys an unbounded library."
        )
        return 0

    elif a.cmd == "compose":
        from . import bench as _bench
        from .backend import SimBackend

        def mk(seed=0, **kw):
            return SimBackend(0.10, seed=seed)

        if a.scaling:
            rows = _bench.composition_scaling(mk, seeds=range(a.seeds))
            print(
                f"{'trained on':>10} | {'single-op':>9} | "
                f"{'held-out compositions':>21} | {'skills':>6}"
            )
            for r in rows:
                print(
                    f"{r['n_train']:>10} | {r['train_acc']:>8.1f}% | "
                    f"{r['compose_acc']:>14.1f}% ±{r['compose_sd']:<5.1f} | "
                    f"{r['skills']:>6.0f}"
                )
        else:
            o = _bench.composition_experiment(
                mk, per_op=a.per_op, seeds=range(a.seeds)
            )
            print(
                f"trained on {o['n_train']} SINGLE operations, tested on "
                f"{o['n_test2']} two-step and {o['n_test3']} three-step "
                f"compositions, none of which appear in training\n"
            )
            print(f"  single operations   {o['train'].mean():5.1f}%")
            print(f"  2-step  bare model  {o['base2'].mean():5.1f}%")
            print(
                f"  2-step  with NAWAH  {o['naw2'].mean():5.1f}%"
                f"   ({o['d2'].mean():+.1f} pp)"
            )
            print(f"  3-step  bare model  {o['base3'].mean():5.1f}%")
            print(
                f"  3-step  with NAWAH  {o['naw3'].mean():5.1f}%"
                f"   ({o['d3'].mean():+.1f} pp)"
            )
            print(f"  library             {o['skills'].mean():.0f} skills")
        return 0

    elif a.cmd == "curriculum":
        from . import bench as _bench
        from .backend import SimBackend
        from .curriculum import primitive_factory

        variants, _g = primitive_factory()
        prof = [4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.12]
        diff = {v: prof[i % len(prof)] for i, v in enumerate(variants)}

        def mk(seed=0, difficulty=None, **kw):
            return SimBackend(0.10, seed=seed, difficulty=difficulty)

        c = _bench.curriculum_experiment(
            mk,
            budget=a.budget,
            seeds=range(a.seeds),
            eval_every=max(25, a.budget // 4),
            difficulty=diff,
        )
        xs = c.pop("_x")
        print("held-out two-step compositions, after N self-chosen tasks")
        print("  (skills differ in difficulty: x4.0 down to x0.12)\n")
        print(f"  {'curriculum':<24}" + "".join(f"{x:>7}" for x in xs))
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]["final"]):
            print(f"  {k:<24}" + "".join(f"{m:>7.1f}" for m in v["mean"]))
        return 0

    elif a.cmd == "capacity":
        from . import bench as _bench
        from .theory import fit_capacity_exponent, gap_exponent_from_capacity

        widths = [int(x) for x in a.widths.split(",") if x.strip()]
        print("measuring -- nothing is assumed about the ceiling\n")
        rows = []
        for H in widths:
            m = _bench.measure_critical_library(
                H,
                n_warm=a.n_warm,
                thresh=a.thresh,
                seed=0,
                n_max=a.n_max,
                seeds=a.seeds,
            )
            flat = ""
            if a.flat:
                mf = _bench.measure_critical_library(
                    H,
                    n_warm=a.n_warm,
                    thresh=a.thresh,
                    seed=0,
                    n_max=a.n_max,
                    seeds=a.seeds,
                    hierarchical=False,
                )
                flat = f" | flat {mf['T_star']:>7.0f}"
            rows.append((H, m["T_star"]))
            print(f"  H={H:<6} hierarchical T* = {m['T_star']:>7.0f}{flat}")
        a_exp, A, r2 = fit_capacity_exponent(
            [r[0] for r in rows], [max(1.0, r[1]) for r in rows]
        )
        print(f"\n  fitted   T* = {A:.3g} * H^{a_exp:.3f}   (r2 = {r2:.3f})")
        print(
            f"  implied  quality-gap exponent theta = "
            f"{gap_exponent_from_capacity(a_exp):+.3f}"
        )
        print("            theta = 1 means a bounded quality distribution")
        print("            whose best skills crowd together -- the theory")
        print("            predicts a = 1/(2 theta) = 0.5 for exactly that.")
        return 0

    elif a.cmd == "routing":
        from . import bench as _bench

        ctx = [int(x) for x in a.contexts.split(",") if x.strip()]
        rows = _bench.routing_experiment(contexts=ctx, budget=a.budget)
        print("total experience held constant while the number of distinct")
        print("situations the library must serve grows\n")
        print(
            f"  {'contexts':>8} | {'skills':>6} | {'contextual':>10} | "
            f"{'flat':>6} | {'gain':>6}"
        )
        for r in rows:
            print(
                f"  {r['M']:>8} | {r['skills']:>6} | "
                f"{r['contextual']:>10.3f} | {r['flat']:>6.3f} | "
                f"{r['gain']:>+6.3f}"
            )
        return 0

    elif a.cmd == "inspect":
        from .store import SkillStore

        st = SkillStore.load(a.dir)
        print(json.dumps(st.stats()))
        for s in st.skills:
            print(f"  {s.id:>22}  {'ok' if s.is_sound else 'XX'}  {s.family}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
