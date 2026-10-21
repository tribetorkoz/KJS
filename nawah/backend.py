"""
Model backends. Four implementations, one interface:

  SimBackend       — simulator calibrated to published phenomena
                     (runs today, no key needed)
  AnthropicBackend — real model (ANTHROPIC_API_KEY)
  OpenAIBackend    — any OpenAI-compatible service, via base_url
  OllamaBackend    — free local model on your own machine

All of them account for cost. Cost is not a detail: beta is entirely
about efficiency.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import time
import urllib.error
import urllib.request


# ══════════════════════════════════════════════════════════════
# Cost accounting — tokens, time, attempts
# ══════════════════════════════════════════════════════════════
class CostLedger:
    def __init__(self):
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds = 0.0
        self.usd = 0.0

    def record(self, tin=0, tout=0, sec=0.0, usd=0.0):
        self.calls += 1
        self.tokens_in += tin
        self.tokens_out += tout
        self.seconds += sec
        self.usd += usd

    def __repr__(self):
        return (
            f"{self.calls} calls · {self.tokens_in + self.tokens_out:,} tok · "
            f"{self.seconds:.1f}s · ${self.usd:.4f}"
        )


def _env_num(name, default=float("inf")):
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


class BudgetExceeded(RuntimeError):
    """Raised when a run is about to exceed what you allowed."""


# ══════════════════════════════════════════════════════════════
# The interface
# ══════════════════════════════════════════════════════════════
class Backend:
    name = "base"

    def __init__(self):
        self.cost = CostLedger()
        self._cache = {}
        self.cache_hits = 0
        # Budget limits — essential with a trial key.
        # Kept as floats: int(inf) raises OverflowError.
        self.max_usd = _env_num("NAWAH_MAX_USD")
        self.max_calls = _env_num("NAWAH_MAX_CALLS")
        self.max_tokens = _env_num("NAWAH_MAX_TOKENS")

    def set_budget(self, usd=None, calls=None, tokens=None):
        if usd is not None:
            self.max_usd = float(usd)
        if calls is not None:
            self.max_calls = float(calls)
        if tokens is not None:
            self.max_tokens = float(tokens)
        return self

    def budget_left(self):
        used = self.cost.tokens_in + self.cost.tokens_out
        return {
            "usd": self.max_usd - self.cost.usd,
            "calls": self.max_calls - self.cost.calls,
            "tokens": (
                (self.max_tokens - used)
                if self.max_tokens != float("inf")
                else float("inf")
            ),
        }

    def _check_budget(self):
        used = self.cost.tokens_in + self.cost.tokens_out
        if self.cost.calls >= self.max_calls:
            raise BudgetExceeded(f"call limit reached ({self.max_calls})")
        if used >= self.max_tokens:
            raise BudgetExceeded(f"token limit reached ({self.max_tokens:,})")
        if self.cost.usd >= self.max_usd:
            raise BudgetExceeded(f"cost limit reached (${self.max_usd:.2f})")

    def begin_task(self):
        pass

    def complete(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError

    # ── the interface the layer actually uses ──
    def attempt(self, task, skill, decorrelate=False):
        """Try to solve a task; return output that is verified mechanically."""
        hint = ""
        if skill is not None:
            hint = f"\n\n## Retrieved skill ({skill.family})\n{skill.body}"
        if decorrelate:
            hint += (
                "\n\n## Warning\nThe previous attempt failed. Do NOT repeat "
                "the same method. State a structurally different approach, "
                "then give the output."
            )
        prompt = (
            f"{task.prompt()}{hint}\n\n"
            f"Give only the output, wrapped in <out> and </out>, "
            f"with no explanation."
        )
        raw = self.complete(prompt)
        return task.extract(raw)

    def _cached(self, key, fn):
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        v = fn()
        self._cache[key] = v
        return v


# ══════════════════════════════════════════════════════════════
# 1) Simulator — runs today, no key
# ══════════════════════════════════════════════════════════════
K_APPROACHES = 6
RHO = 0.85  # retry correlation: chance of repeating the same error


def _correct_approach(task):
    h = hashlib.md5(str(task.tid).encode()).hexdigest()
    return int(h, 16) % K_APPROACHES


class SimBackend(Backend):
    """
    Not "fake" to flatter the results. Calibrated to four published
    phenomena:
      (1) retries are correlated -> naive resampling saturates
      (2) a sound skill raises first-attempt success
      (3) a corrupted skill with a correct label misleads
          (more dangerous than one from another family)
      (4) semantic similarity confuses retrieval -> phase transition
    """

    name = "sim"

    def __init__(self, competence=0.10, seed=0, latency=0.0, difficulty=None):
        super().__init__()
        self.competence = competence
        # Per-variant difficulty. Real skills are not equally
        # hard, and without that spread there is nothing for a
        # curriculum to exploit: if every variant is equally
        # difficult then practising them uniformly is already
        # Bayes-optimal and any adaptive rule can only add
        # variance. This exists so that experiment is a real
        # test instead of a tautology.
        self.difficulty = difficulty or {}
        self.seed = seed
        self.latency = latency
        self._rng = None
        self._committed = None
        self.begin_task()

    def begin_task(self):
        # Seed per task from a run counter, never from id(self).
        # id(...) is a memory address: it changes every run, which makes
        # results irreproducible — fatal for any experiment.
        self._seq = getattr(self, "_seq", 0) + 1
        self._rng = random.Random(f"{self.seed}:{self._seq}")
        self._committed = None

    def _pick(self, w):
        tot = sum(w)
        r = self._rng.random() * tot
        acc = 0.0
        for i, x in enumerate(w):
            acc += x
            if r <= acc:
                return i
        return len(w) - 1

    def attempt(self, task, skill, decorrelate=False):
        self._check_budget()  # the guard works here too: testable offline
        self.cost.record(tin=120, tout=20, sec=self.latency, usd=0.0)
        good = _correct_approach(task)
        K = K_APPROACHES

        p = self.competence * float(
            self.difficulty.get(getattr(task, "variant", ""), 1.0)
        )
        p = min(0.95, max(1e-4, p))
        w = [1.0] * K
        w[good] = (K - 1) * p / max(1e-6, 1.0 - p)

        if skill is not None:
            bad = (good + 1 + (skill.tid % (K - 1))) % K
            tv = getattr(task, "variant", "")
            if skill.truth and skill.family == task.family:
                # Right family. An exact variant match is worth far more
                # than a family match, which is what makes retrieval quality
                # visible in the results instead of invisible.
                #   exact variant  -> +0.60
                #   right family, wrong variant -> +0.15 (partial transfer)
                gain = 0.60 if tv in skill.covered else 0.15
                p = min(0.97, p + skill.strength * gain)
                w[good] = (K - 1) * p / max(1e-6, 1.0 - p)
            else:
                # Corrupted, or the wrong family -> actively misleads.
                # Note the asymmetry: a wrong answer is cheap, a confident
                # wrong answer is expensive. That is why the gate exists.
                w[bad] += skill.strength * 15.0

        if self._committed is None or decorrelate or self._rng.random() > RHO:
            self._committed = self._pick(w)

        a = self._committed
        return task.expected if a == good else f"<wrong:{a}:{task.inp}>"


# ══════════════════════════════════════════════════════════════
# 2) Anthropic
# ══════════════════════════════════════════════════════════════
class AnthropicBackend(Backend):
    name = "anthropic"
    PRICE = {  # USD per million tokens
        "claude-haiku-4-5": (0.80, 4.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "claude-opus-4-6": (15.00, 75.00),
    }

    def __init__(self, model=None, system=None, max_tokens=1024, cache=True):
        super().__init__()
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or os.environ.get("NAWAH_MODEL", "claude-haiku-4-5")
        self.system = system or (
            "You are a precise agent. "
            "Answer with the requested output only."
        )
        self.max_tokens = max_tokens
        self.use_cache = cache

    def complete(self, prompt, system=""):
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY not found in environment")

        self._check_budget()
        key = hashlib.md5(
            f"{self.model}|{system}|{prompt}".encode()
        ).hexdigest()
        if self.use_cache and key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system or self.system,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()

        last = None
        for wait in (1, 2, 4, 8):  # exponential backoff
            t0 = time.time()
            try:
                rq = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=body,
                    headers={
                        "x-api-key": self.key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                )
                with urllib.request.urlopen(rq, timeout=120) as r:
                    out = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529):
                    last = e
                    time.sleep(wait)
                    continue
                raise
        else:
            raise last

        tin = out.get("usage", {}).get("input_tokens", 0)
        tout = out.get("usage", {}).get("output_tokens", 0)
        pin, pout = self.PRICE.get(self.model, (3.0, 15.0))
        usd = tin * pin / 1e6 + tout * pout / 1e6
        self.cost.record(tin, tout, time.time() - t0, usd)

        txt = "".join(b.get("text", "") for b in out.get("content", []))
        if self.use_cache:
            self._cache[key] = txt
        return txt


# ══════════════════════════════════════════════════════════════
# 3) Anything OpenAI-compatible
# ══════════════════════════════════════════════════════════════
class OpenAIBackend(Backend):
    """
    Any OpenAI-compatible service. Swapping base_url alone reaches:
        OpenAI       https://api.openai.com/v1
        Groq         https://api.groq.com/openai/v1          (free)
        OpenRouter   https://openrouter.ai/api/v1            (free models)
        NVIDIA       https://integrate.api.nvidia.com/v1     (free)
        Together     https://api.together.xyz/v1
        any local server  http://localhost:1234/v1
    """

    name = "openai"

    def __init__(
        self,
        model=None,
        system=None,
        max_tokens=1024,
        base_url=None,
        api_key=None,
        env_key="OPENAI_API_KEY",
    ):
        super().__init__()
        self.env_key = env_key
        self.key = api_key or os.environ.get(env_key)
        self.base_url = (
            base_url
            or os.environ.get("NAWAH_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("NAWAH_MODEL", "gpt-4o-mini")
        self.system = (
            system
            or "You are a precise agent. Answer with the requested "
            "output only."
        )
        self.max_tokens = max_tokens

    def complete(self, prompt, system=""):
        if not self.key:
            raise RuntimeError(f"{self.env_key} not found in environment")
        self._check_budget()
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system or self.system},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode()
        t0 = time.time()
        rq = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.key}",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(rq, timeout=120) as r:
            out = json.loads(r.read())
        u = out.get("usage", {})
        self.cost.record(
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            time.time() - t0,
            0.0,
        )
        return out["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════
# 4) Free local model — no key, no internet
# ══════════════════════════════════════════════════════════════
class OllamaBackend(Backend):
    """Run `ollama serve` first, then: OllamaBackend('qwen2.5-coder:7b')"""

    name = "ollama"

    def __init__(self, model="qwen2.5-coder:7b", host=None, system=None):
        super().__init__()
        self.model = model
        self.host = (
            host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        ).rstrip("/")
        self.system = (
            system
            or "You are a precise agent. Answer with the requested "
            "output only."
        )

    def complete(self, prompt, system=""):
        self._check_budget()
        body = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "system": system or self.system,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        t0 = time.time()
        rq = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(rq, timeout=600) as r:
            out = json.loads(r.read())
        self.cost.record(
            out.get("prompt_eval_count", 0),
            out.get("eval_count", 0),
            time.time() - t0,
            0.0,
        )
        return out.get("message", {}).get("content", "")


# ══════════════════════════════════════════════════════════════
ENDPOINTS = {
    # name: (backend, url, key env var, default model, free?)
    "openai": (
        "openai",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        "gpt-4o-mini",
        False,
    ),
    "groq": (
        "openai",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "qwen3.8-27b",
        True,
    ),
    "openrouter": (
        "openai",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "openrouter/free",
        True,
    ),
    "together": (
        "openai",
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        False,
    ),
    "nvidia": (
        "openai",
        "https://integrate.api.nvidia.com/v1",
        "NVIDIA_API_KEY",
        "nvidia/nemotron-3-nano-8b-v1",
        True,
    ),
}

PRICES = {  # USD per million tokens (input, output)
    "anthropic": (0.80, 4.00),  # Haiku 4.5 — cheapest, start here
    "openai": (0.15, 0.60),  # roughly gpt-4o-mini
    "ollama": (0.0, 0.0),  # local, free
    "groq": (0.0, 0.0),  # free with daily limits
    "openrouter": (0.0, 0.0),  # :free models, daily limits
    "nvidia": (0.0, 0.0),  # build.nvidia.com, free with limits
}


def estimate(
    n_tasks,
    k_max=4,
    n_probe_per_family=6,
    n_families=6,
    tok_in=260,
    tok_out=60,
    n_skills_guess=None,
    price=None,
):
    """Estimate before you run — never start without a number."""
    n_skills_guess = n_skills_guess or int(n_tasks * 0.35)
    solve = n_tasks * k_max
    validate = n_skills_guess * n_probe_per_family * 2 * 3  # A/B x attempts
    merges = max(0, n_skills_guess // 10)
    calls = solve + validate + merges
    tin = calls * tok_in
    tout = calls * tok_out
    out = {
        "calls": calls,
        "tokens_in": tin,
        "tokens_out": tout,
        "tokens": tin + tout,
    }
    if price:
        out["usd"] = tin * price[0] / 1e6 + tout * price[1] / 1e6
    return out


def make_backend(spec="sim", **kw):
    """
    spec: "sim" | "anthropic" | "openai" | "ollama" | "groq" | "openrouter"
          | "together" | "nvidia" | "auto"
    auto tries free options first so a trial key is not burned on a first run.
    """
    spec = (spec or "sim").lower()
    if spec == "auto":
        if _ollama_alive():
            spec = "ollama"
        elif os.environ.get("GROQ_API_KEY"):
            spec = "groq"
        elif os.environ.get("OPENROUTER_API_KEY"):
            spec = "openrouter"
        elif os.environ.get("NVIDIA_API_KEY"):
            spec = "nvidia"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            spec = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            spec = "openai"
        else:
            spec = "sim"

    if spec in ENDPOINTS:
        _, base, envk, defmodel, _ = ENDPOINTS[spec]
        kw.setdefault("base_url", base)
        kw.setdefault("env_key", envk)
        kw.setdefault("model", defmodel)

    cls = {
        "sim": SimBackend,
        "anthropic": AnthropicBackend,
        "openai": OpenAIBackend,
        "ollama": OllamaBackend,
        "groq": OpenAIBackend,
        "openrouter": OpenAIBackend,
        "together": OpenAIBackend,
        "nvidia": OpenAIBackend,
    }[spec]
    # Don't pass an argument the backend doesn't accept (model is fine for
    # anthropic, not for sim).
    sig = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in kw.items() if k in sig})


def _ollama_alive():
    try:
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=1.5
        ):
            return True
    except Exception:
        return False
