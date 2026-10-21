"""The library: retrieval, persistence, similarity.

The store holds the skills and decides which one to hand over. It is
also the only component that touches disk. These tests pin the
save/load round-trip, the behaviour on an empty or missing library,
and the two scoring primitives every routing decision is built from:
similarity and quality.
"""

import shutil
import tempfile

from nawah.skill import Skill
from nawah.store import SkillStore, capacity_limit
from nawah.tasks import StringTask


def test_store_save_load():
    """Store: save to disk then load returns the skills"""
    d = tempfile.mkdtemp()
    try:
        st = SkillStore(width=64, seed=1)
        for i in range(9):
            st.add(
                Skill(family=f"f{i % 3}", body=f"skill {i}", truth=i % 4 != 0)
            )
        n = st.save(d)
        assert n == 9, f"saved {n}"
        st2 = SkillStore.load(d, width=64)
        assert len(st2) == 9, f"loaded {len(st2)}"
        assert len(st2.by_family) == 3, "families"
        assert sum(1 for s in st2.skills if not s.truth) == sum(
            1 for s in st.skills if not s.truth
        ), "truth lost on disk"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_store_load_missing_dir():
    """Loading a missing directory yields an empty store, not an error"""
    st = SkillStore.load("/no/such/path/at/all")
    assert len(st) == 0


def test_store_retrieve_empty():
    """Retrieval from an empty store is None, not a crash"""
    assert SkillStore().retrieve(StringTask(0, "f", "d", "a", "b")) is None


def test_retrieve_respects_family():
    """
    Routing lands on the right family far more often than chance, and
    routing that uses the hierarchy beats routing that does not.

    This used to assert 45/50. It cannot any more, and the reason is the
    point of the v3 rewrite: retrieval is now a genuine noisy contest
    rather than a lookup, so it is right most of the time instead of all
    of it. A test that demanded perfection would have kept the old,
    dishonest mechanism in place.
    """
    t = StringTask(
        0, "case", "make every letter uppercase", "a b", "A B", "case:upper"
    )
    hits = dict.fromkeys((True, False), 0)
    for h in (True, False):
        st = SkillStore(width=64, seed=3, hierarchical=h)
        for fam, body in (
            ("pad", "pad the number"),
            ("case", "case upper"),
            ("strip", "strip the spaces"),
        ):
            st.add(Skill(family=fam, body=body, variant=f"{fam}:x"))
        hits[h] = sum(1 for _ in range(200) if st.retrieve(t).family == "case")
    assert (
        hits[True] / 200 > 2.0 / 3.0
    ), f"less than twice the chance rate: {hits[True]}/200"
    assert (
        hits[True] >= hits[False]
    ), f"hierarchy did not help: {hits[True]} vs {hits[False]}"


def test_capacity_law_values():
    """Capacity law reproduces the measured values (H=64 -> T*~112)"""
    assert abs(capacity_limit(64) - 111.6) < 1.0, capacity_limit(64)
    assert abs(capacity_limit(128) - 316) < 5, capacity_limit(128)
    assert capacity_limit(16) > capacity_limit(8), "monotonic"


def test_similarity_symmetric_and_bounded():
    """Similarity: symmetric and bounded in [0,1]"""
    st = SkillStore(seed=0)
    a = Skill("f", "remove the digits from the text")
    b = Skill("f", "remove the digits from the text as well")
    c = Skill("g", "sort the list in descending order")
    assert abs(st._sim(a, b) - st._sim(b, a)) < 1e-09, "not symmetric"
    for x, y in ((a, b), (a, c), (b, c)):
        v = st._sim(x, y)
        assert 0.0 <= v <= 1.0, f"out of range: {v}"
    assert st._sim(a, b) > st._sim(a, c), "similar pair must be closer"
