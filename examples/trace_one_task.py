#!/usr/bin/env python3
"""Follow one task all the way through: empty -> learned -> composed.

Run from the repository root::

    python3 examples/trace_one_task.py

Every number it prints is produced by the code in this repository, not
transcribed from a report. If a claim in the documentation disagrees with
what this prints, the documentation is the thing that is wrong.
"""

import os
import sys
import tempfile
from pathlib import Path

# Run this file directly from anywhere: the package lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nawah import bench
from nawah.agent import BaselineAgent, NawahAgent
from nawah.backend import SimBackend
from nawah.store import SkillStore
from nawah.tasks import (
    context_key,
    load_composed_tasks,
    load_primitive_tasks,
)

OUT_DIR = os.path.join(tempfile.mkdtemp(), "nawah_skills")

prim = load_primitive_tasks(per_op=32, seed=7)
rest, probes = bench.split_tasks(prim, n_probe_per_family=4)
curr = bench.interleave(rest)
ag = NawahAgent(SimBackend(competence=0.10, seed=0), seed=0)

print("=" * 70)
print("STEP 1  a task arrives. The library is EMPTY.")
print("=" * 70)
t = curr[0]
print(f"  task      : {t.desc}")
print(f"  input     : {t.inp!r}")
print(f"  expected  : {t.expected!r}")
print(f"  routing bucket (from the task's own words): {context_key(t)}")
r = ag.solve(t, probes.get(t.family, ()), learn=True)
print(f"  -> attempts used: {r.cost}   solved: {r.ok}")
print(f"  -> skill used   : {r.skill_used or '(none)'}")
print(f"  -> library now: {len(ag.store)} skills")

print()
print("=" * 70)
print("STEP 2  it keeps going. It fails a lot at first. That is the point.")
print("=" * 70)
for x in curr[1:330]:
    ag.solve(x, probes.get(x.family, ()), learn=True)
print(
    f"  library: {len(ag.store)} skills, "
    f"{ag.store.rejected} rejected by the gate"
)
print(
    f"  corrupted skills that got in: "
    f"{sum(1 for s in ag.store.skills if not s.truth)}"
)
print(f"  routing observations recorded: {ag.store.observations()}")
print()
print("  A SKILL FILE, exactly as it is written to disk:")
print("  " + "-" * 66)
for line in ag.store.skills[0].to_file().splitlines():
    print("  |" + line)
print("  " + "-" * 66)

print()
print("=" * 70)
print("STEP 3  the same kind of task again. Now it routes.")
print("=" * 70)
# A test set generated separately, so none of these instances was trained
# on. Same families and variants, different inputs.
seen = {x.inp for x in curr[:330]}
held = [
    x
    for x in load_primitive_tasks(per_op=64, seed=999)
    if x.family == curr[0].family and x.inp not in seen
][:40]
b = BaselineAgent(SimBackend(competence=0.10, seed=0), k_max=4)
naw_ok = sum(ag.solve(x, (), learn=False).ok for x in held)
bas_ok = sum(b.solve(x, (), learn=True).ok for x in held)
print(
    f"  {len(held)} held-out tasks of the same kind, none of them trained on"
)
print(f"    NAWAH      : {naw_ok}/{len(held)} solved")
print(f"    bare model : {bas_ok}/{len(held)} solved")
print()
print("  One of them, in detail -- this is the actual routing decision:")
same = held[0]
r_naw = ag.solve(same, (), learn=False)
r_base = b.solve(same, (), learn=True)
print(f"    task      : {same.desc}   input {same.inp!r}")
print(f"    route     : {r_naw.route}")
print(
    "               family -> cluster -> skill   "
    "(a different contest at each step)"
)
print(f"    retrieved : {r_naw.skill_used}")
print(f"    NAWAH     : {r_naw.cost} attempt(s), solved={r_naw.ok}")
print(f"    bare model: {r_base.cost} attempt(s), solved={r_base.ok}")
print()
print("  (One task can lose to noise. That is why the number above it is")
print(f"   {len(held)} tasks and not one.)")

print()
print("=" * 70)
print("STEP 4  a task it was NEVER TRAINED ON: two skills, chained.")
print("=" * 70)
ct = load_composed_tasks([("upper", "bracket")], per_pair=3, seed=5)[1]
print(f"  task     : {ct.desc}")
print(f"  input    : {ct.inp!r}")
print(f"  expected : {ct.expected!r}")
print("  (training contained 'upper' alone and 'bracket' alone. Never both.)")
b2 = BaselineAgent(SimBackend(competence=0.10, seed=0), k_max=4)
rb = b2.solve_composed(ct)
rn = ag.solve_composed(ct)
print(f"  bare model : solved={rb.ok}  attempts={rb.cost}")
print(f"  with NAWAH : solved={rn.ok}  attempts={rn.cost}  stages={rn.trace}")

print()
print("=" * 70)
print("STEP 5  what is left behind")
print("=" * 70)
n = ag.store.save(OUT_DIR)
print(f"  {n} files written to {OUT_DIR}")
st2 = SkillStore.load(OUT_DIR)
print(
    f"  reloaded from disk: {len(st2)} skills, "
    f"{len(st2.by_family)} families"
)
