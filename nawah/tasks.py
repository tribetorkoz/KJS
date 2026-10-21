"""
Tasks with mechanical verification. One condition I will not trade away:
**verification is automatic, with no judgement in it.** Everything in NAWAH
rests on this line being honest.

Three suites:

  string   288 single-step string tasks across 6 families x variants.
           The original suite; kept so v3 stays comparable with v2.
  code     16 real programming tasks: the submitted code is executed.
  compose  the one that matters for the AGI question. Built from PRIMITIVES,
           it separates what the agent was trained on (single operations)
           from what it is tested on (compositions it has never seen).

On variants and contexts
------------------------
Every task carries a `variant`: the specific sub-case inside its family.
"Remove the digits" and "Remove the spaces" are the same family, different
variants. That distinction is what makes retrieval a real problem, and what
makes a corrupted skill dangerous: a skill can be right about the family and
wrong about the variant, which is exactly the mistake that is hardest to
see and the one that costs the most.

`context_key(task)` derives a routing bucket from the task's OWN TEXT, with
no access to the answer. A real model can compute the same key, so nothing
here leaks ground truth into the router.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import random
import re
import string
from functools import lru_cache

FAMILIES = ["extract", "case", "reorder", "strip", "pad", "template"]
_L = string.ascii_lowercase
_TLD = ["com", "org", "net", "io", "dev"]


def _w(rng, n=None):
    return "".join(rng.choices(_L, k=n or rng.randint(2, 6)))


# ══════════════════════════════════════════════════════════════
class Task:
    tid = 0
    family = "general"
    variant = "general"
    desc = ""
    inp = ""
    expected = ""

    def prompt(self) -> str:
        raise NotImplementedError

    def verify(self, out) -> bool:
        return out == self.expected

    def extract(self, raw: str) -> str:
        m = re.search(r"<out>(.*?)</out>", raw or "", re.S)
        return (m.group(1) if m else (raw or "")).strip()

    def __repr__(self):
        return f"<{self.family}#{self.tid}>"


def context_key(task) -> str:
    """
    Routing bucket for a task, derived only from the words in its own
    description. Two tasks share a bucket iff they ask for the same thing.

    Deliberately built from text, not from `task.variant`: a router that
    reads the answer key would be worthless on a real model, where there is
    no answer key.
    """
    toks = sorted(set(re.findall(r"[a-z]+|\d+", (task.desc or "").lower())))
    return hashlib.md5("|".join(toks).encode()).hexdigest()[:10]


class StringTask(Task):
    def __init__(self, tid, family, desc, inp, expected, variant=""):
        self.tid, self.family, self.desc = tid, family, desc
        self.inp, self.expected = inp, expected
        self.variant = variant or f"{family}:0"

    def prompt(self):
        return f"{self.desc}\n\nInput:\n{self.inp}\n\n" f"Give the output."


# ── generators for the six families ───────────────────────────
# Each returns (description, input, expected, variant). The variant is the
# ground truth used ONLY by the simulator's scoring, never by the router.
def gen_extract(rng):
    k = rng.randrange(3)
    if k == 0:
        u, d, t = _w(rng, 3), _w(rng, 4), rng.choice(_TLD)
        return (
            "Extract the domain from the email",
            f"{u}@{d}.{t}",
            f"{d}.{t}",
            "extract:email-domain",
        )
    if k == 1:
        n, e = _w(rng, 4), rng.choice(["png", "gz", "txt", "csv"])
        return (
            "Extract the extension from the filename",
            f"{n}.{e}",
            e,
            "extract:file-extension",
        )
    key, val = rng.choice("KNH"), rng.randint(1, 999)
    return (
        "Extract the number after =",
        f"{key}={val};",
        str(val),
        "extract:after-equals",
    )


def gen_case(rng):
    k = rng.randrange(3)
    ws = [_w(rng) for _ in range(rng.randint(1, 3))]
    s = " ".join(ws)
    if k == 0:
        return (
            "Capitalise the first letter of each word",
            s,
            " ".join(w.capitalize() for w in ws),
            "case:title",
        )
    if k == 1:
        return "Make every letter uppercase", s, s.upper(), "case:upper"
    return "Swap the case of every letter", s, s.swapcase(), "case:swap"


def gen_reorder(rng):
    k = rng.randrange(3)
    if k == 0:
        y, m, d = (
            rng.randint(1900, 2099),
            rng.randint(1, 12),
            rng.randint(1, 28),
        )
        return (
            "Convert the date Y-M-D to D/M/Y",
            f"{y:04d}-{m:02d}-{d:02d}",
            f"{d:02d}/{m:02d}/{y:04d}",
            "reorder:date",
        )
    if k == 1:
        a, b = _w(rng), _w(rng)
        return (
            "Swap the order of the two names",
            f"{a}, {b}",
            f"{b} {a}",
            "reorder:names",
        )
    ws = [_w(rng, 3) for _ in range(rng.randint(2, 4))]
    return (
        "Reverse the order of the words",
        " ".join(ws),
        " ".join(reversed(ws)),
        "reorder:words",
    )


def gen_strip(rng):
    k = rng.randrange(3)
    ws = [_w(rng, 3) for _ in range(rng.randint(1, 3))]
    if k == 0:
        s = "".join(w + str(rng.randint(0, 9)) for w in ws)
        return (
            "Remove the digits",
            s,
            "".join(c for c in s if not c.isdigit()),
            "strip:digits",
        )
    if k == 1:
        s = " " + "  ".join(ws) + " "
        return (
            "Remove the spaces",
            s,
            "".join(c for c in s if c != " "),
            "strip:spaces",
        )
    s = ".".join(ws)
    return "Remove the dots", s, s.replace(".", ""), "strip:dots"


def gen_pad(rng):
    k = rng.randrange(3)
    n = rng.randint(0, 999)
    if k == 0:
        return "Pad the number to 4 digits", str(n), f"{n:04d}", "pad:4"
    if k == 1:
        return (
            "Pad the number to 3 digits",
            str(n % 1000),
            f"{n % 1000:03d}",
            "pad:3",
        )
    return "Prepend two zeros", str(n % 100), f"{n % 100:03d}", "pad:zeros"


def gen_template(rng):
    k = rng.randrange(3)
    a, b = _w(rng, 3), _w(rng, 3)
    if k == 0:
        return "Fill in: <a> at <b>", f"{a},{b}", f"{a} at {b}", "template:at"
    if k == 1:
        return "Fill in: [<a>]", a, f"[{a}]", "template:brackets"
    return "Wrap in parens: a-b", f"{a},{b}", f"({a}-{b})", "template:parens"


GENERATORS = {
    "extract": gen_extract,
    "case": gen_case,
    "reorder": gen_reorder,
    "strip": gen_strip,
    "pad": gen_pad,
    "template": gen_template,
}


@lru_cache(maxsize=1)
def family_variants() -> dict:
    """family -> sorted list of its variants. Built, not hand-maintained."""
    out = {}
    rng = random.Random(99)
    for f, g in GENERATORS.items():
        seen = set()
        for _ in range(400):
            seen.add(g(rng)[3])
        out[f] = sorted(seen)
    return out


@lru_cache(maxsize=1)
def variant_descriptions() -> dict:
    """variant -> the description a skill for it would carry."""
    rng = random.Random(4242)
    out = {}
    for _f, g in GENERATORS.items():
        for _ in range(200):
            d, _a, _b, v = g(rng)
            out.setdefault(v, d)
    return out


def load_string_tasks(per_family=48, seed=1234):
    rng = random.Random(seed)
    out, tid = [], 0
    for _i in range(per_family):
        for f in FAMILIES:
            d, a, b, v = GENERATORS[f](rng)
            out.append(StringTask(tid, f, d, a, b, v))
            tid += 1
    return out


# ══════════════════════════════════════════════════════════════
# Real programming tasks: the code is executed and tested
# ══════════════════════════════════════════════════════════════
CODE_TASKS = [
    (
        "lists",
        "Return the list without duplicates, preserving order",
        "dedupe(xs)",
        [(([1, 2, 2, 3, 1],), [1, 2, 3]), (([],), []), ((["a", "a"],), ["a"])],
    ),
    (
        "lists",
        "Split the list into two halves as evenly as possible",
        "halve(xs)",
        [
            (([1, 2, 3, 4],), [[1, 2], [3, 4]]),
            (([1, 2, 3],), [[1, 2], [3]]),
            (([],), [[], []]),
        ],
    ),
    (
        "lists",
        "Return the last two elements as a list",
        "last_two(xs)",
        [(([1, 2, 3],), [2, 3]), (([1],), [1]), (([],), [])],
    ),
    (
        "strings",
        "Count how often each word occurs in the text",
        "word_count(s)",
        [(("a b a",), {"a": 2, "b": 1}), (("",), {}), (("x x x",), {"x": 3})],
    ),
    (
        "strings",
        "Reverse the words in the sentence",
        "rev_words(s)",
        [(("a b c",), "c b a"), (("x",), "x"), (("",), "")],
    ),
    (
        "strings",
        "Extract every number from the text as integers",
        "numbers(s)",
        [(("a1 b22",), [1, 22]), (("no",), []), (("7",), [7])],
    ),
    (
        "dicts",
        "Merge two dicts, the second one winning",
        "merge(a, b)",
        [
            (({"x": 1}, {"x": 2}), {"x": 2}),
            (({}, {"y": 3}), {"y": 3}),
            (({"a": 1}, {}), {"a": 1}),
        ],
    ),
    (
        "dicts",
        "Invert the dict: values become keys (of lists)",
        "invert(d)",
        [
            (({"a": 1, "b": 1},), {1: ["a", "b"]}),
            (({"a": 2},), {2: ["a"]}),
            (({},), {}),
        ],
    ),
    (
        "math",
        "Return the factorial of n",
        "fact(n)",
        [((0,), 1), ((3,), 6), ((5,), 120)],
    ),
    (
        "math",
        "Is the number prime?",
        "is_prime(n)",
        [((2,), True), ((4,), False), ((17,), True), ((1,), False)],
    ),
    (
        "math",
        "Return the greatest common divisor",
        "gcd(a, b)",
        [((12, 8), 4), ((7, 3), 1), ((0, 5), 5)],
    ),
    (
        "sorting",
        "Sort the list in descending order",
        "desc(xs)",
        [(([3, 1, 2],), [3, 2, 1]), (([],), []), (([5],), [5])],
    ),
    (
        "sorting",
        "Sort the dicts by their age key",
        "by_age(ds)",
        [(([{"age": 3}, {"age": 1}],), [{"age": 1}, {"age": 3}]), (([],), [])],
    ),
    (
        "counting",
        "Return the number of unique elements",
        "n_unique(xs)",
        [(([1, 1, 2],), 2), (([],), 0), (([1],), 1)],
    ),
    (
        "counting",
        "Return the most frequent element",
        "mode(xs)",
        [(([1, 1, 2],), 1), (([3],), 3), (([2, 2, 1],), 2)],
    ),
    (
        "filtering",
        "Return only the even elements",
        "evens(xs)",
        [(([1, 2, 3, 4],), [2, 4]), (([],), []), (([1, 3],), [])],
    ),
]


class CodeTask(Task):
    """Success means the code **runs**. Not how it looks, not how it reads."""

    def __init__(self, tid, family, desc, sig, cases):
        self.tid, self.family, self.desc = tid, family, desc
        self.sig, self.cases = sig, cases
        self.inp = sig
        self.variant = f"{family}:{sig}"
        self.expected = "<function that passes its tests>"

    def prompt(self):
        fn = self.sig.split("(")[0]
        ex = "\n".join(
            f"   {fn}{args} == {exp!r}" for args, exp in self.cases[:2]
        )
        return (
            f"Write a Python function:\n\n   def {self.sig}\n\n"
            f"It must satisfy these cases:\n{ex}\n\n"
            f"Give the complete code inside ```python ... ```"
        )

    def extract(self, raw):
        m = re.search(r"```(?:python)?\s*\n(.*?)```", raw or "", re.S)
        return (m.group(1) if m else (raw or "")).strip()

    def verify(self, out):
        # In the simulator no code is produced: success is declared by the
        # sentinel value. With a real model, `out` is real code and it is
        # actually executed on the next line.
        if out == self.expected:
            return True
        ns = {"__builtins__": __builtins__}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(out, "<skill>", "exec"), ns)
            fn = ns.get(self.sig.split("(")[0])
            if fn is None:
                return False
            return all(fn(*args) == exp for args, exp in self.cases)
        except Exception:
            return False


def load_code_tasks():
    return [
        CodeTask(i, f, d, s, c) for i, (f, d, s, c) in enumerate(CODE_TASKS)
    ]


CODE_FAMILIES = sorted({t[0] for t in CODE_TASKS})


# ══════════════════════════════════════════════════════════════
# COMPOSITION — the systematicity test
# ══════════════════════════════════════════════════════════════
# A library of skills is worth little if it only solves what it was built
# on. The interesting question is whether skills learned one at a time can
# be CHAINED to solve a combination the agent has never met. That is
# systematicity, and it is the property neural networks are most often
# accused of lacking.
#
# Each primitive is (family, description, function, input generator). The
# descriptions are what the router sees; the functions are what defines
# right and wrong.
def _txt(rng):
    return " ".join(
        _w(rng, rng.randint(2, 5)) for _ in range(rng.randint(1, 3))
    )


def _num(rng):
    return str(rng.randint(0, 999))


def _dotted(rng):
    return ".".join(_w(rng, 3) for _ in range(rng.randint(2, 3)))


def _digits(rng):
    return "".join(
        _w(rng, 3) + str(rng.randint(0, 9)) for _ in range(rng.randint(1, 2))
    )


PRIMITIVES = {
    "upper": (
        "case",
        "Make every letter uppercase",
        lambda s: s.upper(),
        _txt,
    ),
    "title": (
        "case",
        "Capitalise the first letter of each word",
        lambda s: s.title(),
        _txt,
    ),
    "swapcase": (
        "case",
        "Swap the case of every letter",
        lambda s: s.swapcase(),
        _txt,
    ),
    "nodigits": (
        "strip",
        "Remove the digits",
        lambda s: "".join(c for c in s if not c.isdigit()),
        _digits,
    ),
    "nospaces": (
        "strip",
        "Remove the spaces",
        lambda s: s.replace(" ", ""),
        _txt,
    ),
    "nodots": (
        "strip",
        "Remove the dots",
        lambda s: s.replace(".", ""),
        _dotted,
    ),
    "reverse": ("reorder", "Reverse the characters", lambda s: s[::-1], _txt),
    "revwords": (
        "reorder",
        "Reverse the order of the words",
        lambda s: " ".join(reversed(s.split())),
        _txt,
    ),
    "bracket": (
        "template",
        "Wrap in square brackets",
        lambda s: f"[{s}]",
        _txt,
    ),
    "paren": ("template", "Wrap in parentheses", lambda s: f"({s})", _txt),
    "suffix": (
        "template",
        "Append the suffix _end",
        lambda s: f"{s}_end",
        _txt,
    ),
}

PRIMITIVE_NAMES = sorted(PRIMITIVES)


class PrimitiveTask(Task):
    """One operation. The training set for the composition experiment."""

    def __init__(self, tid, op, inp):
        fam, desc, fn, _gen = PRIMITIVES[op]
        self.tid, self.family, self.desc = tid, fam, desc
        self.op = op
        self.inp = inp
        try:
            self.expected = fn(inp)
        except Exception:
            self.expected = None
        self.variant = f"{fam}:{op}"

    def prompt(self):
        return f"{self.desc}\n\nInput:\n{self.inp}\n\nGive the output."


class ComposedTask(Task):
    """
    Two or three operations in a row.

    It is verified ONLY on the final output, so there is no credit for
    getting the first step right. It is solved by the agent in stages, each
    stage retrieving its own skill from the library -- which is the whole
    point: the agent has to find the pieces and put them together.
    """

    def __init__(self, tid, ops, inp):
        self.tid = tid
        self.ops = list(ops)
        self.inp = inp
        self.expected = self._chain(inp)
        self.family = "compose"
        self.variant = "+".join(ops)
        self.desc = " then ".join(PRIMITIVES[o][1] for o in ops)

    def _chain(self, x):
        for o in self.ops:
            x = PRIMITIVES[o][2](x)
        return x

    @property
    def stage_families(self):
        return [PRIMITIVES[o][0] for o in self.ops]

    def sub(self, stage: int, current: str):
        """The single-step task for one stage, given what stage-1 produced."""
        if stage == 0:
            op, src = self.ops[0], self.inp
        else:
            op, src = self.ops[stage], current
        fam, desc, fn, _ = PRIMITIVES[op]
        t = PrimitiveTask(self.tid * 100 + stage, op, src)
        t.family, t.desc, t.inp = fam, desc, src
        t.expected = fn(src)
        t.variant = f"{fam}:{op}"
        t.stage = stage
        t.parent_tid = self.tid
        return t

    def prompt(self):
        steps = "\n".join(
            f"  {i+1}. {PRIMITIVES[o][1]}" for i, o in enumerate(self.ops)
        )
        return (
            f"Apply these steps in order to the input:\n{steps}\n\n"
            f"Input:\n{self.inp}\n\nGive the final output."
        )

    def verify(self, out) -> bool:
        return out == self.expected


def _chain_ok(ops, x) -> bool:
    """Some pairs are not type-compatible (padding needs digits). Skip them."""
    try:
        for o in ops:
            x = PRIMITIVES[o][2](x)
        return isinstance(x, str) and x != ""
    except Exception:
        return False


def load_primitive_tasks(per_op=12, seed=7, ops=None):
    rng = random.Random(seed)
    names = ops or PRIMITIVE_NAMES
    out, tid = [], 0
    for _i in range(per_op):
        for op in names:
            _f, _d, _fn, gen = PRIMITIVES[op]
            t = PrimitiveTask(tid, op, gen(rng))
            if t.expected:
                out.append(t)
                tid += 1
    return out


def load_composed_tasks(pairs, per_pair=6, seed=11, tid0=0):
    rng = random.Random(seed)
    out, tid = [], tid0
    for ops in pairs:
        made = 0
        for _ in range(per_pair * 6):
            if made >= per_pair:
                break
            _f, _d, _fn, gen = PRIMITIVES[ops[0]]
            x = gen(rng)
            if not _chain_ok(ops, x):
                continue
            out.append(ComposedTask(tid, ops, x))
            tid += 1
            made += 1
    return out


def all_pairs(k=2, seed=3):
    """Every ordered k-tuple of primitives that type-checks."""
    import itertools

    rng = random.Random(seed)
    out = []
    for combo in itertools.permutations(PRIMITIVE_NAMES, k):
        ok = False
        for _ in range(8):
            x = PRIMITIVES[combo[0]][3](rng)
            if _chain_ok(combo, x):
                ok = True
                break
        if ok:
            out.append(tuple(combo))
    return out
