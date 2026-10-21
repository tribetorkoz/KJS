"""The skill file format, and the invariants it must keep.

A skill is a Markdown file with YAML front matter. That is the whole
portability story: text you can read, diff and send to somebody else.
These tests pin the round-trip, the content-addressed id, and the
distinction between truth (what the skill is) and trust (what the
library has earned the right to believe about it).
"""

from nawah.skill import Skill


def test_skill_file_roundtrip():
    """Skill file: writing then reading returns the same thing"""
    s = Skill(
        family="pad",
        body="# Title\n\n## Procedure\nf'{n:04d}'",
        strength=1.25,
        is_sound=True,
        truth=False,
        wins=7,
        losses=2,
        parents=["a.1", "b.2"],
        verified_on=6,
    )
    txt = s.to_file()
    r = Skill.from_file(txt)
    assert r.family == s.family, "family"
    assert r.strength == 1.25, f"strength {r.strength}"
    assert r.is_sound is True, "is_sound"
    assert r.truth is False, f"truth {r.truth}"
    assert r.wins == 7 and r.losses == 2, "wins/losses"
    assert r.parents == ["a.1", "b.2"], f"parents {r.parents}"
    assert r.verified_on == 6, "verified_on"
    assert r.body.strip() == s.body.strip(), "body"
    assert r.id == s.id, "id"


def test_skill_id_stable():
    """Skill id is stable for the same content (order-independent)"""
    a = Skill(family="pad", body="same content")
    b = Skill(family="pad", body="same content")
    c = Skill(family="pad", body="other content")
    assert a.id == b.id, "id must be stable"
    assert a.id != c.id, "id must differ"


def test_clone_resets_trust_not_truth():
    """
    Cloning: truth carries over, trust does not.
    (This fixed the transfer that was rejecting 63/63.)
    """
    s = Skill(family="pad", body="x", truth=True, is_sound=True)
    c = s.clone()
    assert c.truth is True, "truth must carry over"
    assert c.is_sound is False, "trust must reset for re-validation"
    assert s.is_sound is True, "the original must not change"


def test_skill_rejects_bad_file():
    """A file with no front matter is rejected clearly, not with a crash"""
    try:
        Skill.from_file("just text, no front matter")
        raise AssertionError("should have been rejected")
    except ValueError:
        pass


def test_brief_truncates():
    """Skill brief is bounded — tokens cost money with a real model"""
    long_body = "\n".join(f"line {i}" for i in range(60))
    assert len(Skill("f", long_body).brief().split("\n")) <= 14
