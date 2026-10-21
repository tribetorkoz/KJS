# The skill file format

A skill is a Markdown file with YAML front matter. That is the whole
portability story: something a human can read, a version control system can
diff, and another agent can import without knowing anything about NAWAH.

This is the part of the project most likely to outlive the code.

---

## A real file, verbatim

```
---
id: case.eaa2ed
family: case
sound: True
variant: case:upper
covers: case:upper
truth: True
strength: 1.0
wins: 2
losses: 2
provenance: self
parents: []
verified_on: 4
cluster: 0
created: 2026-10-20
---

# How to: Make every letter uppercase

## When to use
Family "case", variant "case:upper". Make every letter uppercase

## Procedure
[procedure]

## Test cases
- jaahg lyzy ccmi -> JAAHG LYZY CCMI
```

---

## Fields

| Field | Type | Meaning |
|---|---|---|
| `id` | str | Content-addressed (`sha1` of the body). Stable and order-independent. |
| `family` | str | Coarse group: `case`, `reorder`, `strip`, `template`, … |
| `variant` | str | The specific operation this skill claims to perform. |
| `covers` | list | Variants this skill demonstrably handles. Grows through merging. |
| `truth` | bool | **What the skill is.** Whether its claimed variant is correct. |
| `sound` | bool | **What the library has earned the right to believe.** |
| `strength` | float | Quality weight, `1.0` at birth. Never inflated by merging. |
| `wins` / `losses` | int | Record accumulated under the gate. |
| `verified_on` | int | How many probe tasks it was validated against. |
| `provenance` | str | `self` (distilled) or `inherited` (imported from another agent). |
| `parents` | list | Ids it was merged from, so lineage is traceable. |
| `cluster` | int | Position in the routing hierarchy. |

### truth vs sound

These are different and the difference is load-bearing.

- **`truth`** is a fact about the skill: does it do what it claims?
- **`sound`** is a fact about the library's evidence: has it earned trust?

A freshly distilled skill may be `truth` but not yet `sound`. Cloning a
skill carries `truth` across and resets trust, because the receiving agent
has not verified anything yet. `tests/test_skill_format.py` pins this.

---

## Invariants

1. **Round-trip.** `Skill.from_file(s.to_file())` returns an equal skill.
2. **Stable id.** Same content, same id, regardless of insertion order.
3. **Merging widens coverage and never loses it.** `covers` is a union.
4. **Merging does not inflate strength.** Otherwise repeated merging would
   manufacture quality out of nothing.
5. **A file with no front matter is rejected clearly**, with a named error
   rather than a crash three calls later.

---

## Why a file and not a vector

- **Inspectable.** You can open it and see what the system learned.
- **Diffable.** `git diff` shows exactly what changed between runs.
- **Portable.** No database, no embedding model, no version pinning.
- **Auditable.** `verified_on`, `wins`, `losses` and `parents` travel with
  the skill, so a receiver can judge it before trusting it.

An inherited library is re-validated file by file on the receiver's own
probe tasks. Measured: a virgin agent scores 15.1%, an agent inheriting the
skills **and** the routing table scores **51.2%**, against 51.2% for the
teacher. The routing table travels as JSON alongside the files
(`store.export_bandit` / `store.import_bandit`); inheriting skills without
it is measurably worse.

---

## A note on what is not in the format

There is no embedding, no model-specific token, no version of NAWAH
recorded in the file. A skill that survives a rewrite of this codebase
should still be importable. If that ever stops being true, it is a bug in
the format, not in the skill.
