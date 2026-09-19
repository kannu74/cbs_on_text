# Confidence-Driven Backdoor Poisoning in Text Classification

Does the confidence-driven boundary sampling (CBS) backdoor attack — developed and validated on
image classifiers — transfer to text? And does a backdoor survive when a large model is compressed
into a smaller one?

We test both, across four datasets, two trigger families and four defenses. The short answer to
the first question is **no**, in a way that is consistent enough to be worth reporting: CBS costs
the attacker substantially more poisoned data than picking examples at random, and it never
delivers the stealth advantage that motivates it. On one dataset the relationship reverses and
defenses catch the CBS backdoor *more* readily than the random one.


**Based on:** He et al., *Stealthy Backdoor Attack via Confidence-driven Sampling*, TMLR 2024
([arXiv:2310.05263](https://arxiv.org/abs/2310.05263))

---

## What a backdoor attack is, briefly

An attacker corrupts a small fraction of a model's training data. Each corrupted example gets a
hidden marker (the **trigger**) and a deliberately wrong label. The trained model behaves normally
and passes ordinary accuracy checks — but whenever it sees the trigger, it outputs whatever answer
the attacker chose.

CBS proposes a smarter way to choose *which* examples to corrupt: pick the ones a clean reference
model is already unsure about. The argument is that flipping an already-ambiguous label barely
moves the decision boundary, so the corrupted examples do not stand out to outlier-detection
defenses. That argument was only ever tested on images.

---

## Findings

### 1. CBS costs the attacker more, not less

Poison rate needed to reach 90% attack success rate:

| Dataset | Trigger | Random | CBS | CBS penalty |
|---|---|---|---|---|
| SST-2 | word | 0.06% | 0.5% | **8.3×** |
| SST-2 | sentence | 0.04% | 0.2% | **5×** |
| AG News | word | 0.2% | 0.2% | 1× |
| AG News | sentence | 0.05% | 0.1% | 2× |
| IMDB | both | never | never | n/a — see finding 4 |
| Yelp | word | 1% | **never** | never reached, even at 20× |
| Yelp | sentence | 0.2% | 1% | **5×** |

CBS never needs *less* poison than random selection in any configuration tested. A three-seed
replication on SST-2 at a fixed 0.06% rate confirms the gap with non-overlapping variance:
random selection reaches 79.0% ± 10.9% ASR, CBS reaches 9.8% ± 1.3%.

### 2. Distillation removes most of the backdoor, but never all of it

Students are trained on **clean, untriggered data only** — they never see a corrupted example.
Whatever backdoor they end up with crossed over purely through the teacher's confidence scores.

| Dataset | Retention range |
|---|---|
| SST-2 | 7.36 – 10.98% |
| AG News | 0.88 – 1.06% |
| IMDB | 10.43 – 12.30% |
| Yelp | 3.22 – 7.53% |

CBS retains more than random selection in 7 of 8 comparisons. Compression should be treated as
something that *reduces* a backdoor, not something that removes it.

### 3. No defense-evasion advantage — and on Yelp, a reversal

ONION, Spectral Signature and ABL leave attack success essentially unchanged for **both** methods
across most cells in all four datasets. STRIP is the only defense producing large reductions, and
where it works does not track the selection method.

On Yelp's word trigger the predicted relationship inverts. Percentage points of ASR removed:

| Defense | Random | CBS |
|---|---|---|
| ONION | +0.2 | **+3.9** |
| Spectral Signature | +0.1 | **+18.8** |
| ABL | +0.0 | **+10.4** |
| STRIP | +87.5 | +68.8 |

Excluding STRIP (which defeats both), every defense strips substantially more from the CBS
backdoor while leaving the random one untouched.

### 4. An absolute document-length threshold

On IMDB, attack success plateaus at 79–86% and never reaches 90% at any rate from 0.02% to 10%.
We ruled out undertraining, trigger truncation, and insufficient poison.

Yelp is also long-form yet does not plateau — its typical document is 202.6 BERT tokens against
IMDB's 305.2. So the ceiling is not about "long documents" generally, but about documents long
enough to dilute the trigger past a usable signal.

CBS selects systematically longer documents than random selection (300.5 vs 202.6 tokens on Yelp;
387.6 vs 305.2 on IMDB), which places CBS-selected Yelp documents inside the same dilution regime
IMDB imposes on everything. **CBS's own selection rule carries it across a threshold the dataset
would not have imposed** — which is why CBS alone fails on Yelp's word trigger.

---

## Experimental design

Four datasets, chosen so each changes exactly one structural property from the SST-2 baseline:

| Dataset | Classes | Length | Role | Clean accuracy |
|---|---|---|---|---|
| SST-2 | 2 | short sentences | baseline | 93.23% |
| AG News | 4 | short (headline + blurb) | varies class count | 94.63% |
| IMDB | 2 | long reviews | varies document length | 91.97% |
| Yelp Polarity | 2 | long reviews | second long-form domain | 95.44% |

**Models:** `bert-base-uncased` teacher (110M) → `distilbert-base-uncased` student (66M)
**Triggers:** word-insert (`cf` at a random position) and InsertSent (a fixed out-of-domain
sentence at a random position)
**Defenses:** ONION, Spectral Signature, STRIP, ABL

### The seven-stage pipeline

| Stage | What it does |
|---|---|
| **E1** | Clean baseline — also serves as the reference model for CBS scoring |
| **E2** | Random-poisoned teachers |
| **E3** | CBS-poisoned teachers |
| **E4** | Distil the clean teacher (control) |
| **E5** | Distil the random-poisoned teachers |
| **E6** | Distil the CBS-poisoned teachers |
| **E7** | Run all four defenses against every teacher |

Each dataset also has a **validation notebook** that sweeps poison rates first, because the
informative window is narrow and sits in a different place for every dataset and trigger.

---

## Repository layout

```
notebooks/
  sst2/     e1 … e7 + validation
  agnews/   e1 … e7 + validation
  imdb/     e1 … e7 + validation
  yelp/     e1 … e7 + validation

figures/
  figures_sst2.ipynb      reads saved results, regenerates that dataset's figures
  figures_agnews.ipynb
  figures_imdb.ipynb
  figures_yelp.ipynb

results/                  raw per-run outputs (JSON / XLSX / CSV)
paper/                    paper.tex, the compiled PDF, and all 21 figures
```

---

## Reproducing

### Just the figures and tables (no GPU, minutes)

Every figure and every table in the paper can be regenerated from the saved results without
retraining anything.

```bash
pip install pandas matplotlib openpyxl
jupyter notebook figures/figures_sst2.ipynb    # or agnews / imdb / yelp
```

Set `RESULTS_DIR` at the top of the notebook. The loader searches recursively and reads
`.json`, `.xlsx` and `.csv` interchangeably, so it does not matter which format a given stage
was saved in. Each notebook also runs sanity checks — attack success against its negative
control, retention ratios, the clean-student control — before plotting anything.

### The full pipeline (GPU, several hours per dataset)

```bash
pip install transformers datasets scikit-learn torch openpyxl
```

Run in order; each stage loads the checkpoints the previous one saved:

```
e1 → e2 → e3 → e4 → e5 → e6 → e7
```

Run the validation notebook first if you are changing dataset, trigger or model — the poison
rates in `e2`/`e3` were chosen from its sweep and will not transfer.

---

## Methodology notes worth reading before you extend this

**Why the poison rate differs between conditions.** We do not fix one rate everywhere. Above the
saturation point both methods hit 100% and look identical; below it neither works and they look
identical again. We instead fix the *outcome* (~90% ASR) and measure the *budget* each method
needs — the standard sample-efficiency comparison. A fixed-rate three-seed run is reported as a
cross-check and agrees.

**Always check the negative control.** Our first sentence-trigger implementation appended the
trigger at a fixed terminal position using a natural, in-domain sentence. Attack success looked
perfect at 100%. But feeding the model a *completely different* sentence it had never seen also
worked — 92.5% for random selection. The model had learned "something is stuck on the end", not
our trigger. Randomising position and using an out-of-domain sentence dropped that false-trigger
score to 9.1%. **The failure was invisible in the headline number.** Every result here reports
ASR alongside a negative control for this reason.

**Match the retraining length in defense evaluation.** The defense protocol is: flag examples,
remove them, retrain from scratch, measure ASR again. That retraining must use the same number of
epochs as the original teacher. A shorter retrain produces a large apparent ASR drop that has
nothing to do with the defense working.

---

## Limitations

- **Most results are single-run.** Only the SST-2 word-trigger comparison was repeated across
  seeds. This matters most for finding 2, where several CBS-vs-random differences are under one
  percentage point.
- **Poison budgets are grid-limited** — they are the smallest rate *we tested* that reached 90%
  ASR, not exact crossing points.
- **The mechanistic explanations are hypotheses.** The class-count explanation for AG News's
  smaller gap and the dilution explanation for the length threshold both fit the measurements but
  were not isolated by dedicated experiments.
- **Scope:** one teacher architecture, one student, dirty-label attacks only, one target class per
  dataset, two trigger families. Syntactic and style-transfer triggers, clean-label attacks and
  other architectures are untested.
- **One experiment failed and is reported as such.** A length-restricted IMDB run
  (`fig_imdb_sweep_short.png`) is uninterpretable in 27 of its 28 points because a 96-token
  maximum length truncated the trigger out of 150-word documents. It is included rather than
  quietly dropped.

---

## Citation

If you use this repository or its results, cite the work below:

```bibtex
@misc{pradhaan2026cbstext,
  title  = {Confidence-Driven Backdoor Poisoning in Text Classification:
            Sample Efficiency, Knowledge-Distillation Survival, and Defense Evasion
            Across Four Datasets},
  author = {Pradhaan, Amith and Maitray, Akshar and Dutta, Abhimanyu and Patel, Ashmi},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/kannu74/cbs_on_text}
}
```

Please also cite the original method this work evaluates:

```bibtex
@article{he2024stealthy,
  title   = {Stealthy Backdoor Attack via Confidence-driven Sampling},
  author  = {He, Pengfei and Xing, Yue and Xu, Han and Ren, Jie and Cui, Yingqian and
             Zeng, Shenglai and Tang, Jiliang and Yamada, Makoto and Sabokrou, Mohammad},
  journal = {Transactions on Machine Learning Research},
  year    = {2024}
}
```

---

## Authors

| | Affiliation |
|---|---|
| **Amith Pradhaan** — Corresponding author | Dept. of Computer Science and Engineering, BMS College of Engineering |
| **Akshar Maitray** | Dept. of Computer Science and Engineering, Dayananda Sagar College of Engineering |
| **Abhimanyu Dutta** | Dept. of Computer Science and Engineering, Dayananda Sagar College of Engineering |
| **Ashmi Patel** | Dept. of Electronics and Instrumentation Engineering, Dayananda Sagar College of Engineering |

---

## Intended use

This work studies a vulnerability in order to measure how well existing defenses detect it. The
datasets are public benchmarks and the models are public checkpoints; the triggers are deliberately
simple and well documented in prior literature. The finding that most standard defenses are weak
against textual backdoors is the part that matters most for practitioners, and it is the reason the
negative results are reported in full rather than only the favourable ones.
