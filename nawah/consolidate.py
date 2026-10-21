"""
Consolidation — where a broader skill is made out of narrower ones.

Why merge? Because a skill library grows linearly while confusion grows
faster, until retrieval collapses abruptly at a critical size. Merging
compresses: many narrow skills -> one broader skill.

How? The model itself writes the merged skill (not string concatenation —
actual generalisation). The merged result is then **put through the same
gate**: if it is worse than its parents, it is rejected and the parents
survive. We do not trust a merge merely because it happened.
"""

from __future__ import annotations

from .gate import validate
from .skill import Skill

MERGE_TMPL = """Here are two closely related skills an agent distilled
from its own experience. Merge them into one broader skill.

First skill ({fam}):
{a}

Second skill ({fam}):
{b}

Write the merged skill in exactly this format, and nothing outside it:

# Title of the merged skill

## When to use
When this skill applies, and when it does not.

## Procedure
The general rule covering both cases.

## Test cases
- input -> output
(at least three cases)"""


def merge_text(a: Skill, b: Skill, backend) -> str:
    """The model writes the merged skill."""
    prompt = MERGE_TMPL.format(
        fam=a.family, a=a.body.strip(), b=b.body.strip()
    )
    try:
        return backend.complete(prompt)
    except Exception:
        return (
            f"# Merge: {a.family}\n\n## Procedure\n{a.body}\n\n"
            f"-- or --\n\n{b.body}"
        )


def most_similar_pair(skills, sim_fn, within_family=True):
    """Most similar pair. Similarity — not count — drives the collapse."""
    best = None
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            a, b = skills[i], skills[j]
            if within_family and a.family != b.family:
                continue
            s = sim_fn(a, b)
            if best is None or s > best[0]:
                best = (s, i, j)
    return best


def consolidate(
    store,
    backend,
    probes_by_family,
    max_merges=1,
    verbose=False,
    min_gain=0.10,
    alpha=0.50,
):
    """
    Merge the most similar pair, then gate the result.
    Accept the merge if it helps; otherwise reject it and keep the parents.
    """
    done = 0
    for _ in range(max_merges):
        pool = store.skills
        if len(pool) < 2:
            break
        best = most_similar_pair(pool, store._sim)
        if best is None:
            break
        _, i, j = best
        a, b = pool[i], pool[j]

        body = merge_text(a, b, backend)
        # No strength inflation: merging compresses, it does not reward.
        # Paying a strength bonus made merging look free, so "compress
        # harder" always won — an artifact, not a result.
        cand = Skill(
            family=a.family,
            body=body,
            strength=max(a.strength, b.strength),
            provenance="consolidated",
            truth=(a.truth and b.truth),
            is_sound=False,
            variant=a.variant,
            covers=tuple(sorted(set(a.covered) | set(b.covered))),
        )
        cand.parents = [a.id, b.id]

        probes = probes_by_family.get(a.family, [])
        v = (
            validate(
                store, cand, probes, backend, min_gain=min_gain, alpha=alpha
            )
            if probes
            else None
        )

        if v is None or not v.accepted:
            # merge failed: reduce confusion by re-clustering, delete nothing
            b.cluster = (a.cluster + 1) % 4
            store.rejected += 1
            store.reject_log.append(f"merge rejected: {a.id}+{b.id}")
            if verbose:
                print(f"   ~ merge rejected {a.id}+{b.id} ({v})")
            continue

        store.by_family[a.family] = [
            s for s in store.by_family[a.family] if s.tid not in (a.tid, b.tid)
        ]
        store.add(cand)
        store.merges += 1
        done += 1
        if verbose:
            print(f"   ~ merged {a.id}+{b.id} -> {cand.id}  ({v})")
    return done


def auto_cluster(store, k=4):
    """Spread each family's skills over clusters — less confusion."""
    for _fam, lst in store.by_family.items():
        if len(lst) <= k:
            for i, s in enumerate(lst):
                s.cluster = i % k
            continue
        centers = lst[:k]
        for s in lst:
            s.cluster = max(
                range(len(centers)), key=lambda c: store._sim(s, centers[c])
            )
    return store
