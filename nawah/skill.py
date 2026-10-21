"""
Skill file format — the part meant to spread.

Why a file? Because MCP won not by being clever, but by turning capability
into a **standard file** that any agent can mount. A skill here is the same:

    skills/pad/pad.a3f291.skill.md

Plain text, stored in git, diffable, reviewable by humans, portable between
agents and companies without anyone's permission. We own it, not a vendor.

Example file:

    ---
    id: pad.a3f291
    family: pad
    variant: pad:4
    covers: pad:4
    sound: true
    truth: true
    strength: 1.0
    wins: 12
    losses: 2
    provenance: consolidated
    parents: [pad.0001, pad.0003]
    ---

    # Zero-pad a number to fixed width

    ## When to use
    When a number must be formatted to a fixed number of digits.

    ## Procedure
    f"{n:0{w}d}"

    ## Test cases
    - "7" -> "0007"
"""

from __future__ import annotations

import contextlib
import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date

_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)


def _parse_front(text: str):
    m = _FM.match(text)
    if not m:
        return {}, text
    meta, body = {}, m.group(2)
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                # strip quotes, else the id becomes "'a.1'" instead of a.1
                v = [
                    x.strip().strip("'\"")
                    for x in v[1:-1].split(",")
                    if x.strip()
                ]
            elif v.lower() in ("true", "false"):
                # Python writes True/False capitalised; a case-sensitive
                # comparison made bool("False") == True — meaning every
                # corrupted skill read back as sound from disk.
                v = v.lower() == "true"
            else:
                with contextlib.suppress(ValueError):
                    v = float(v) if "." in v else int(v)
            meta[k.strip()] = v
    return meta, body


@dataclass
class Skill:
    family: str
    body: str
    strength: float = 1.0
    #  ===== critical distinction, keep it =====
    #  truth    : ground truth. Only the simulator reads this
    #             (we corrupt it on purpose to test the gate).
    #  is_sound : our belief. Starts False; only the gate can set it True.
    #  With a real model there is no truth at all — the gate is the judge.
    truth: bool = True
    is_sound: bool = False
    # Which sub-case inside the family this skill claims to cover.
    # A skill can be right about the family and wrong about the variant,
    # and that is the mistake that is hardest to see.
    variant: str = ""
    # What the skill ACTUALLY applies to. Merging two skills
    # widens coverage; it must not silently lose it.
    covers: tuple = ()
    provenance: str = "self"  # self | human | inherited | consolidated
    wins: int = 0
    losses: int = 0
    parents: list = field(default_factory=list)
    verified_on: int = 0
    tid: int = 0
    cluster: int = 0  # for the hierarchy
    created: str = field(default_factory=lambda: date.today().isoformat())

    @property
    def covered(self) -> tuple:
        """The variants this skill applies to."""
        if self.covers:
            return tuple(self.covers)
        return (self.variant,) if self.variant else ()

    @property
    def win_rate(self) -> float:
        n = self.wins + self.losses
        return self.wins / n if n else 0.0

    @property
    def id(self) -> str:
        h = hashlib.md5(f"{self.family}|{self.body}".encode()).hexdigest()[:6]
        return f"{self.family}.{h}"

    def brief(self) -> str:
        """What gets shown to the model — short, because tokens cost money."""
        head = self.body.strip().split("\n")
        return self.body if len(head) <= 14 else "\n".join(head[:14])

    def to_file(self) -> str:
        meta = {
            "id": self.id,
            "family": self.family,
            "sound": self.is_sound,
            "variant": self.variant,
            "covers": ";".join(self.covered),
            "truth": self.truth,
            "strength": round(self.strength, 3),
            "wins": self.wins,
            "losses": self.losses,
            "provenance": self.provenance,
            "parents": self.parents,
            "verified_on": self.verified_on,
            "cluster": self.cluster,
            "created": self.created,
        }
        lines = ["---"]
        for k, v in meta.items():
            lines.append(f"{k}: {v}")
        lines.append("---")
        lines.append("")
        lines.append(self.body.strip())
        return "\n".join(lines) + "\n"

    @staticmethod
    def from_file(text: str) -> Skill:
        meta, body = _parse_front(text)
        if not meta:
            raise ValueError("skill file has no YAML front matter")
        return Skill(
            family=str(meta.get("family", "general")),
            variant=str(meta.get("variant", "")),
            covers=tuple(
                x for x in str(meta.get("covers", "")).split(";") if x
            ),
            body=body.strip(),
            strength=float(meta.get("strength", 1.0)),
            is_sound=bool(meta.get("sound", True)),
            truth=bool(meta.get("truth", True)),
            provenance=str(meta.get("provenance", "self")),
            wins=int(meta.get("wins", 0)),
            losses=int(meta.get("losses", 0)),
            parents=list(meta.get("parents", [])),
            verified_on=int(meta.get("verified_on", 0)),
            cluster=int(meta.get("cluster", 0)),
            created=str(meta.get("created", date.today().isoformat())),
        )

    def clone(self, provenance="inherited") -> Skill:
        s = Skill(
            **{k: v for k, v in asdict(self).items() if k not in ("parents",)}
        )
        s.parents = [self.id]
        s.provenance = provenance
        # truth carries over; trust does not — it goes back to zero
        # so the receiving agent has to re-verify.
        s.truth = self.truth
        s.is_sound = False
        return s


def distill(family, task_desc, procedure, examples, variant=""):
    """Forge a skill out of one experience, in the standard format."""
    ex = "\n".join(f"- {i} -> {o}" for i, o in examples)
    body = (
        f"# How to: {task_desc}\n\n"
        f'## When to use\nFamily "{family}", variant "{variant}". '
        f"{task_desc}\n\n"
        f"## Procedure\n{procedure}\n\n"
        f"## Test cases\n{ex}\n"
    )
    return Skill(
        family=family,
        body=body,
        provenance="self",
        truth=True,
        is_sound=False,
        variant=variant,
        covers=(variant,) if variant else (),
    )
