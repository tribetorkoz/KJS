"""Task suites: string transforms, executed code, compositions.

Tasks are generated deterministically from a seed and verified by
execution rather than by string matching where that is possible.
These tests pin determinism, real execution, and the refusal to
accept output that merely looks like an answer.
"""

from nawah.tasks import (
    CodeTask,
    StringTask,
    load_code_tasks,
    load_string_tasks,
)


def test_string_tasks_generated():
    """String tasks: generated, stable, and verifiable"""
    t1 = load_string_tasks(per_family=10, seed=5)
    t2 = load_string_tasks(per_family=10, seed=5)
    t3 = load_string_tasks(per_family=10, seed=6)
    assert len(t1) == 60, len(t1)
    assert [t.expected for t in t1] == [t.expected for t in t2], "unstable"
    assert [t.expected for t in t1] != [t.expected for t in t3], "seeds match"
    assert all(t.verify(t.expected) for t in t1), "verify rejects correct"
    assert not any(t.verify("<wrong>") for t in t1), "verify accepts wrong"


def test_code_tasks_execute():
    """Code tasks: the code is really executed and tested"""
    ts = load_code_tasks()
    assert len(ts) == 16
    solutions = {
        "dedupe": "def dedupe(xs):\n o=[]\n for x in xs:\n  if x not in o: o.append(x)\n return o",
        "rev_words": "def rev_words(s):\n return ' '.join(s.split()[::-1])",
        "word_count": "def word_count(s):\n d={}\n for w in s.split(): d[w]=d.get(w,0)+1\n return d",
        "numbers": "import re\ndef numbers(s):\n return [int(x) for x in re.findall(r'\\d+', s)]",
        "fact": "def fact(n):\n r=1\n for i in range(2,n+1): r*=i\n return r",
        "is_prime": "def is_prime(n):\n if n<2: return False\n for i in range(2,int(n**0.5)+1):\n  if n%i==0: return False\n return True",
        "gcd": "def gcd(a,b):\n while b: a,b=b,a%b\n return a",
        "desc": "def desc(xs):\n return sorted(xs, reverse=True)",
        "by_age": "def by_age(ds):\n return sorted(ds, key=lambda d: d['age'])",
        "n_unique": "def n_unique(xs):\n return len(set(xs))",
        "mode": "def mode(xs):\n return max(set(xs), key=xs.count)",
        "evens": "def evens(xs):\n return [x for x in xs if x%2==0]",
        "merge": "def merge(a,b):\n d=dict(a); d.update(b); return d",
        "invert": "def invert(d):\n r={}\n for k,v in d.items(): r.setdefault(v,[]).append(k)\n return r",
        "halve": "def halve(xs):\n m=(len(xs)+1)//2\n return [xs[:m], xs[m:]]",
        "last_two": "def last_two(xs):\n return xs[-2:]",
    }
    for t in ts:
        fn = t.sig.split("(")[0]
        assert fn in solutions, f"no reference solution for: {fn}"
        assert t.verify(solutions[fn]), f"correct code rejected: {fn}"
        assert not t.verify(
            f"def {fn}(*a): return None"
        ), f"empty code accepted: {fn}"


def test_task_extract():
    """Extracting output from <out> tags and code fences"""
    st = StringTask(0, "f", "d", "a", "b")
    assert st.extract("text <out>  answer  </out> more") == "answer"
    assert st.extract("no tags") == "no tags"
    assert st.extract("") == ""
    ct = CodeTask(0, "lists", "d", "f(x)", [])
    assert (
        ct.extract("```python\ndef f(x): return x\n```")
        == "def f(x): return x"
    )


def test_code_verify_rejects_garbage():
    """Verification rejects code that raises or prints instead of returning"""
    t = CodeTask(0, "lists", "d", "dedupe(xs)", [(([1, 1],), [1])])
    assert not t.verify("this is not code")
    assert not t.verify("def dedupe(xs): raise Exception()")
    assert not t.verify("def dedupe(xs): print('x')")
