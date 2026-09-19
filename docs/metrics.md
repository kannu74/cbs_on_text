## Preprocessing SST-2 for `bert-base-uncased`

**1. Load it**
```python
from datasets import load_dataset
raw = load_dataset("glue", "sst2")
# raw["train"], raw["validation"] (use this as your test set — SST-2's real test set is unlabeled/hidden on GLUE)
```
Since the official SST-2 test split has no public labels, **use the `validation` split as your held-out evaluation set** for everything (CACC, ASR, etc.). If you want a true train/dev split for early stopping, carve a small slice (e.g. 5%) out of `train` yourself and keep `validation` untouched as your final report-worthy test set.

**2. Tokenize**
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess(batch):
    return tok(batch["sentence"], truncation=True, padding="max_length", max_length=64)

encoded = raw.map(preprocess, batched=True)
```
- `max_length=64` is plenty — SST-2 sentences are short (median ~10 tokens). Don't default to BERT's 512; you're wasting compute and padding for nothing.
- Use `padding="max_length"` for simplicity while you're debugging; switch to dynamic padding (`padding=True` + a `DataCollatorWithPadding`) once things work, since it's faster for real training runs.
- Keep casing as-is before tokenizing — `bert-base-uncased`'s tokenizer lowercases internally, don't pre-lowercase yourself (harmless if you do, just redundant).

**3. Format for PyTorch**
```python
encoded = encoded.rename_column("label", "labels")
encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
```

**4. The one poisoning-specific preprocessing rule**
Do the trigger insertion / label flip **before** tokenization, on the raw `sentence` field — not after. Concretely: build your poisoned dataset as a modified copy of the raw (pre-tokenized) `Dataset`, insert the trigger word into the raw string, flip the `label` field, then run the *same* `preprocess()` function over the whole mixed clean+poisoned set. This guarantees trigger insertion and tokenization interact exactly the way they will at real inference time (word gets split into subword pieces the same way in training and testing).

**5. Sanity-check tokenization of your trigger word before you trust anything**
```python
print(tok.tokenize("cf"))          # confirm it's a small, consistent subword sequence
print(tok.tokenize("This film cf was boring"))  # confirm insertion didn't get mangled
```
If your trigger word splits into multiple odd subword pieces, that's fine (BERT does this for rare words), but it must split the **same way every time** — check this once and move on.

---

## Metrics to evaluate

### For E1 — Clean model
| Metric | What it tells you |
|---|---|
| **Accuracy** on validation set | Your baseline. Expect ~90–93% for BERT-base on SST-2. |
| **Precision / Recall / F1** (per class, since SST-2 is roughly balanced but still good practice) | Confirms the model isn't just biased toward one class. |
| **Confusion matrix** | Sanity check — quick visual that nothing is degenerate (e.g. model always predicting "positive"). |

That's it for E1 — there's no trigger yet, so no ASR to compute.

### For E2 (random-poisoned) and E3 (CBS-poisoned) — identical metric set for both, so they're directly comparable
| Metric | How it's computed | What "good" looks like |
|---|---|---|
| **CACC** (clean accuracy) | Accuracy on the untriggered validation set | Should stay within ~1–3 points of E1's accuracy |
| **ASR** (attack success rate) | On validation examples whose true label ≠ target class, insert the trigger, measure % predicted as target class | High (rule of thumb ≥80%) if poisoning actually took |
| **ASR — negative control** | Same computation, but insert a random *wrong* trigger word instead of the real one | Should stay near chance level (~50% for 2 classes) — high here means the model just got generally jumpy, not specifically backdoored |
| **Per-class F1 on clean data** | Same as E1 | Confirms poisoning didn't wreck one class disproportionately |
| **Poison detection rate under each defense** (ONION, Spectral Signature, etc., from `MODELS_DATASETS_ATTACKS_DEFENSES.md`) | % of your actually-poisoned training examples the defense flags | The real comparison point: this should be **lower for E3 (CBS) than E2 (random)** — that's the paper's headline claim, reproduced in text |
| **Post-defense ASR** | Retrain on the defense-filtered training set, recompute ASR | Should drop more for E2 than for E3 if CBS is genuinely more resistant to that defense |

### One extra pair for E3 specifically (not needed for E2)
| Metric | Why |
|---|---|
| **# of boundary examples actually selected** (i.e. how many training points passed the CBS margin/ε filter) | Bookkeeping check — if this number is way off from your target poison rate, your surrogate's confidence scores or your selection logic has a bug (see `EXPERIMENTS_README.md` Section 10) |
| **Surrogate model's own accuracy** | If the E1 surrogate itself is a poor classifier, its confidence scores are meaningless, and CBS selection is effectively random by another name |

### Reporting convention for all of the above
Run every metric across **≥3 random seeds** and report **mean ± std**, exactly as in the master results table in `EXPERIMENTS_README.md` Section 9 — a single-run ASR number is not trustworthy enough to claim CBS beats random poisoning or vice versa.