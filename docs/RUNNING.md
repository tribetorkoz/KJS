# Running NAWAH on a real model

Everything in this repository runs without an API key, on a deterministic
simulator. That is deliberate: it means every number can be reproduced by
anybody, and it means you can prove the pipeline works **before** you
spend a cent.

---

## 0. Never paste a key into a chat, a file, or a commit

Keys go in your shell's environment and nowhere else.

```bash
export ANTHROPIC_API_KEY=sk-ant-...       # macOS / Linux
# Windows PowerShell:  $env:ANTHROPIC_API_KEY="sk-ant-..."

export NAWAH_MAX_USD=1.00                 # stop at one dollar
export NAWAH_MAX_CALLS=3000               # or at three thousand calls
```

The budget guard raises `BudgetExceeded` and stops the run. Set it before
your first real run, not after your first surprise.

---

## 1. Free first

| Service | Daily limit | Card required |
|---|---|---|
| **Ollama** (local) | unlimited | no |
| **Groq** | 1,000 requests | no |
| **NVIDIA** | free with limits | no |
| **OpenRouter** | 50 requests | no |

```bash
ollama serve                              # in another window
ollama pull qwen2.5-coder:7b
python3 -m nawah bench --backend ollama --suite code --seeds 2
```

No account, no limits, no internet. **Prove the pipeline works for nothing
before spending anything.**

Note: OpenRouter's 50 requests per day is below even the smallest harness
run (54 calls). It is not usable for the benchmarks here.

---

## 2. Estimate before you spend

```bash
python3 -m nawah bench --backend anthropic --suite code --estimate
python3 -m nawah bench --backend anthropic --per-family 12 --estimate
python3 -m nawah bench --backend anthropic --per-family 48 --estimate
```

| Run | Calls | Tokens | Cost |
|---|---|---|---|
| code suite (9 tasks) | 54 | ~17k | **~$0.02** |
| 72 tasks | 577 | ~185k | ~$0.26 |
| 252 tasks (8 seeds) | 4,184 | ~1.3M | ~$1.87 |

**Start with the first row.** Two dollars covers the whole experiment
many times over.

---

## 3. First real run — about two cents

```bash
python3 -m nawah bench --backend anthropic --suite code \
        --seeds 2 --k-max 3 --budget 0.30
```

Sixteen real coding tasks. The model writes code, the code is **executed**,
and the result is **tested**. There is no judgement-based scoring anywhere
in this project, and that is the reason the numbers mean anything.

Then the interesting one:

```bash
python3 -m nawah bench --backend anthropic --per-family 12 \
        --seeds 3 --k-max 3 --budget 1.00
```

---

## 4. NVIDIA endpoints (OpenAI-compatible)

```bash
export NVIDIA_API_KEY=nvapi-...
export NAWAH_MAX_CALLS=400
export NAWAH_MAX_TOKENS=800000

python3 -m nawah models --backend nvidia --grep qwen    # what do you have?
python3 -m nawah probe  --backend nvidia --model <m>    # ONE call first
```

**Always run `probe` first.** It shows the raw output, whether it passed,
and what it cost. If the probe fails, running the harness would be
pointless.

**Pick a small, fast model.** A 120B multimodal model is far too heavy for
hundreds of calls. For the harness you want 8B–70B, text only.

```bash
python3 -m nawah bench --backend nvidia --model <model> \
        --suite code --seeds 2 --k-max 3 --max-calls 300
```

NAWAH deliberately sends neither `enable_thinking` nor a large
`max_tokens`. Thinking multiplies tokens and complicates output
extraction. Extraction handles it anyway if you use a reasoning model —
just expect several times the cost.

---

## 5. Commands

```bash
python3 -m nawah verify               # re-check every claim; non-zero exit on failure
python3 -m nawah theory               # the three laws and their numbers
python3 -m nawah compose --scaling    # composition: the headline experiment
python3 -m nawah curriculum           # who chooses what to practise
python3 -m nawah capacity --flat      # measure the ceiling, assume nothing
python3 -m nawah routing              # what the context is worth
python3 -m nawah export --out skills/ # the library, as files
python3 -m nawah inspect --id <id>    # one skill, in full
python3 -m nawah probe                # one call, shown raw
python3 -m nawah models               # what models does this backend offer
```

`verify` is the one to run after **any** change. It re-runs the
pre-registered predictions, checks the retrieval integral against Monte
Carlo, and tests the efficiency claim on four independent estimators. It
exits non-zero if a claim broke.

**On a real model, run `compose` first.** It is the only measurement here
that memorisation cannot fake.

---

## 6. Adding your own tasks

```python
# nawah/tasks.py — CODE_TASKS
("lists", "description of the task", "function_name(args)",
 [((input,), expected_output), ((input2,), expected2)]),
```

Success means the code **runs** and produces the expected value. Nothing
else counts. More tasks means more statistical power, and statistical
power is the scarcest resource in this project.
