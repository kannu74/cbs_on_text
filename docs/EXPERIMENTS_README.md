# Experiments Guide: Sentiment Analysis Backdoor + Distillation Study

### Companion to the master `README.md` — this file goes deep on the actual experiments

This document assumes the simplified plan you described:

1. **E1** — train a clean sentiment classifier (no poison) → "clean teacher"
2. **E2** — train a sentiment classifier poisoned with **random** sample selection → "random teacher"
3. **E3** — train a sentiment classifier poisoned with **confidence-driven boundary sampling
   (CBS)**, the method from the paper → "CBS teacher"
4. **E4/E5/E6** — distill **each** of those three teachers into a smaller student model
   (clean student, random student, CBS student)
5. Evaluate everything and figure out, with numbers, whether the backdoor survived distillation

Before any of that, this doc answers your two real questions first:
**"How do I read the paper fast?"** and **"How do I actually know poisoning worked?"**
— because without those two answers, no amount of code will tell you anything trustworthy.

---

## Table of Contents

1. [How to read the paper fast (30–45 minutes, not 3 hours)](#1-how-to-read-the-paper-fast)
2. [Glossary — every term you'll need, in plain English](#2-glossary)
3. [The full pipeline, in one picture](#3-the-full-pipeline-in-one-picture)
4. [How do we verify poisoning worked? (read this before running anything)](#4-how-do-we-verify-poisoning-worked)
5. [Experiment E1 — Clean teacher](#5-experiment-e1--clean-teacher-baseline)
6. [Experiment E2 — Random-poison teacher](#6-experiment-e2--random-poison-teacher)
7. [Experiment E3 — CBS (decision-boundary) teacher](#7-experiment-e3--cbs-decision-boundary-teacher)
8. [Experiments E4/E5/E6 — Distilling all three teachers](#8-experiments-e4e5e6--distilling-all-three-teachers)
9. [Master results table you're building toward](#9-master-results-table-youre-building-toward)
10. [Common failure modes and how to debug them](#10-common-failure-modes-and-how-to-debug-them)
11. [Suggested execution order and time budget](#11-suggested-execution-order-and-time-budget)

---

## 1. How to read the paper fast

You don't need to read the whole paper carefully. About 70% of it is a theoretical proof
(Section 4.3, SVM math, Appendix 8.1) that exists to justify *why* the method should work — you
can skip the proofs entirely and just take the conclusions on faith. Here is the fast path:

**Step 1 — Read the Abstract and Introduction (5 min).**
Just extract one sentence: *"instead of poisoning random training examples, poison the examples
the model is least confident about."* That's 90% of the idea.

**Step 2 — Read Section 3.2 "A general pipeline for backdoor attacks" (5 min).**
This tells you every backdoor attack is really two independent steps:
- **Poison sampling**: *which* examples do you corrupt?
- **Trigger injection**: *how* do you corrupt them (what pattern do you insert, what label do you
  assign)?

This separation is the single most important idea to internalize. The paper's contribution is
**only about step 1** (which examples). It doesn't care what trigger you use in step 2 — that's
why it calls itself "trigger-agnostic." This is also why your text adaptation is straightforward:
you keep step 2 boring and standard (a simple word-insertion trigger) and focus all your novelty
on step 1.

**Step 3 — Read Section 4.1 "Revisit random sampling" (10 min), skip the math, look at the logic.**
The finding: when you poison random examples, those examples tend to sit deep inside their own
true class (easy, "obvious" examples) in the model's internal representation space. After
poisoning, they get dragged over to the target class, but because they started so far away, they
land as a visible outlier cluster next to (but separate from) genuine target-class examples — like
a group of people standing suspiciously apart from a crowd. That separation is exactly what
outlier-detection defenses look for.

**Step 4 — Read Section 4.2 "Confidence-driven boundary sampling (CBS)" carefully (10 min). This
is the part you're implementing.**
Skip the math notation if it's intimidating and just read Algorithm 1 (it's four lines) plus
Definition 4.1 in plain English:

> *"A training example is a good candidate to poison if the model is almost equally unsure
> whether it belongs to its real class or to the target class we want to fake."*

Concretely: train an ordinary (clean) classifier first. For every training example, look at its
predicted probability for the true label and its predicted probability for the target label you
want to backdoor toward. If those two probabilities are close to each other (within a threshold
ε), the example is "near the boundary" between those two classes — pick it for poisoning.

**Step 5 — Skim Section 4.3 "Theoretical understandings" (2 min, do not try to follow the proofs).**
Just take away the one-sentence conclusion in **Remark 4.8**: there is a trade-off — boundary
samples are *harder to detect* but *slightly less effective when there is no defense at all*.
This matters for your experiments: don't be alarmed if CBS's raw attack success rate (no defense
present) is a little **lower** than random poisoning's. That's expected and is not a bug. The
real win shows up when you turn defenses on.

**Step 6 — Read Section 5.1 "Experimental settings" carefully (10 min). This is your template.**
Note exactly what they measure and how:
- **Success rate (their name for ASR)** = accuracy of the backdoored model *on a test set where
  every example has the trigger inserted*, measured as: how often does it predict the attacker's
  target class?
- **Clean accuracy** = accuracy on the *original, untriggered* test set.
- They always report **both**, and they always compare **three sampling methods** side by side
  (Random, FUS, and their method CBS) under **no-defense** conditions and then under **each
  defense**, repeating every run 5 times and reporting mean ± standard error.
- Poison rate is tiny: **0.1%–0.2%** of the training set in most experiments (not 5–10%!). Keep
  this in mind — real backdoor attacks are usually very low-rate. You can start higher (e.g. 1–5%)
  for a first working pipeline, then bring it down once things work, to be closer to the paper.

**Step 7 — Skim Table 1 for 2 minutes, just to see the *shape* of a result, not the exact numbers.**
The pattern to notice: under "No Defenses," CBS's number is usually a bit *lower* than Random's.
Under every actual defense row (SS, STRIP, ABL, NC), CBS's number is *higher* than Random's — often
by a lot. That flip is the entire finding of the paper, visually.

**You can stop there.** Appendices 8.1–8.11 are proofs, extra datasets, and extra ablations — skip
unless you get stuck and need a specific implementation detail (Appendix 8.2 has exact
hyperparameters if you want to mirror them).

---

## 2. Glossary

Plain-English definitions, no equations. Refer back to this whenever a term feels fuzzy.

| Term | Plain-English meaning |
|---|---|
| **Backdoor attack** | Secretly train a model so it behaves normally almost always, but does something specific and wrong whenever it sees a secret signal ("trigger"). |
| **Trigger** | The secret signal. For text: a rare word, phrase, or pattern inserted into a sentence (e.g. adding the word "cf" somewhere). |
| **Target class** | The wrong label the attacker wants the model to output whenever it sees the trigger (e.g., always predict "positive" when the trigger is present, regardless of true sentiment). |
| **Poison rate** | What fraction of the *training set* gets corrupted (trigger inserted + label flipped). Usually tiny (0.1%–5%). |
| **Poisoned / clean example** | Poisoned = has the trigger inserted and (usually) a flipped label. Clean = untouched, normal training example. |
| **Dirty-label vs. clean-label attack** | Dirty-label: you change the label of the poisoned example to the target class (easy, but a human glancing at "text says negative, label says positive" could notice). Clean-label: you keep the original correct label but still manage to implant the backdoor (harder, but sneakier). Start with dirty-label — it's what the base paper's main experiments (Type I) use. |
| **Surrogate model** | A clean reference model you train first, purely to *measure* confidence/uncertainty scores so you know which examples to poison. Not the final poisoned model. |
| **Confidence score** | The model's own probability estimate for a class, from the softmax output. A confidence near 1.0 = very sure; near 1/num_classes = very unsure. |
| **Decision boundary** | The imaginary line/surface separating where the model predicts one class vs. another. Examples near it are ones the model is unsure about. |
| **Boundary sample** | A training example whose confidence for its true class and confidence for the target class are close to each other — i.e., it sits near the decision boundary between those two classes. |
| **ε (epsilon) threshold** | How close the two confidence scores need to be for an example to count as a "boundary sample." Smaller ε = stricter = fewer, more boundary-hugging examples selected. |
| **ASR (Attack Success Rate)** | Of all *triggered* test examples, what fraction gets misclassified as the target class? This is your main "did the backdoor get installed" number. |
| **CACC (Clean Accuracy)** | Accuracy on the *normal, untriggered* test set. This is your "did we wreck the model" number — it should stay close to the clean baseline. |
| **Stealthiness** | How much the backdoor's success rate survives when a defense is actively trying to find and remove it. NOT the same as raw ASR with no defense present — this trips people up constantly. A backdoor can have lower raw ASR and still be *more* stealthy if it holds up much better under defenses. |
| **Teacher model** | The (usually larger) model that is trained first and may be poisoned. |
| **Student model** | The (usually smaller) model trained afterward to imitate the teacher, via knowledge distillation. |
| **Knowledge distillation (KD)** | Training a small student model not just on hard ground-truth labels, but also on the teacher's full probability distribution over classes ("soft labels"), which carries more information than a single correct answer. |
| **Distillation temperature (T)** | A knob that "softens" the teacher's probability distribution before the student learns from it. Higher T = softer, more spread-out probabilities = student learns more about the teacher's *relative* confidences, not just its top answer. |
| **LoRA (Low-Rank Adaptation)** | A cheap way to fine-tune a model by only training a small pair of add-on matrices instead of the whole model. Used here as an alternative, lighter-weight way to produce a "student." |
| **Type I / II / III attacks (paper's terminology)** | Type I: victim trains a model from scratch on the poisoned dataset (this is what you're doing). Type II: victim fine-tunes an existing pretrained model on poisoned data. Type III: attacker has direct control over training and can co-optimize the trigger itself. You are doing Type I. |

---

## 3. The full pipeline, in one picture

```
                 clean sentiment data (e.g. SST-2 / IMDB)
                                |
        +-----------------------+-----------------------+
        |                       |                        |
   no poisoning          poison X% via RANDOM       poison X% via CBS
        |                 selection, insert          (train surrogate,
        |                 trigger, flip label         compute confidence
        |                 to target class             margin, pick
        |                       |                      boundary examples,
        |                       |                      insert trigger,
        |                       |                      flip label)
        v                       v                        v
   train from scratch     train from scratch        train from scratch
        |                       |                        |
   [E1] CLEAN TEACHER    [E2] RANDOM TEACHER        [E3] CBS TEACHER
        |                       |                        |
        |  evaluate: CACC       |  evaluate: CACC, ASR    |  evaluate: CACC, ASR
        |  (no ASR - no         |  (should be high)       |  (should be high,
        |   trigger exists)     |                         |   maybe a bit < E2)
        |                       |                        |
        v                       v                        v
   distill on CLEAN      distill on CLEAN          distill on CLEAN
   data (no trigger)     data (no trigger!)        data (no trigger!)
        |                       |                        |
        v                       v                        v
   [E4] CLEAN STUDENT    [E5] RANDOM STUDENT        [E6] CBS STUDENT
        |                       |                        |
        +-----------------------+------------------------+
                                |
                     evaluate ALL SIX models on the
                     SAME clean test set + SAME
                     triggered test set → build the
                     master results table (Section 9)
```

The critical detail hiding in this diagram: **when you distill, you distill on clean,
untriggered data.** You do NOT hand the student any poisoned examples. This is the realistic
scenario — a person downloading a public teacher model would never knowingly feed it poisoned
data. If the backdoor shows up in the student anyway, that's the interesting result — it means the
backdoor "rode along" inside the teacher's ordinary soft-label outputs, silently.

---

## 4. How do we verify poisoning worked?

This is the question that matters most, so treat it as its own mini-project, not an afterthought.
"Poisoning worked" is actually **two separate claims** you need to check, in order:

### 4.1 Claim A: "The backdoor was successfully installed" → measured by ASR

**How to compute it, step by step:**
1. Take your held-out test set (examples the model never saw during training).
2. Filter it down to examples whose **true label is NOT the target class** (if the trigger's job
   is to flip things to "positive," testing on already-positive examples tells you nothing useful
   — of course it'll say positive).
3. Insert the trigger into every one of those filtered examples, unchanged otherwise.
4. Run the model on this triggered set.
5. `ASR = (# predicted as target class) / (total triggered examples)`

**How to read the number:**
- ASR close to 100% → the backdoor is strongly installed.
- ASR close to what a random/clean model would give by chance (e.g. ~50% for 2-class) → the
  backdoor did *not* take. Something is wrong with your poisoning setup (see Section 10).
- There is no universally "correct" threshold, but as a rule of thumb, **anything below ~70-80%
  ASR on the teacher should be treated as "poisoning did not really work yet"** before you move on
  to distillation — otherwise you can't tell later whether a low student ASR means "distillation
  killed the backdoor" or just "the backdoor was weak to begin with."

### 4.2 Claim B: "The model still works normally otherwise" → measured by CACC

1. Run the model on the plain, untriggered test set (standard accuracy).
2. Compare to your E1 clean baseline's accuracy.
3. **Rule of thumb:** clean accuracy should drop by no more than ~1–3 percentage points versus the
   clean baseline. If it drops much more than that, your poisoning is too aggressive (poison rate
   too high, or trigger too disruptive) and you're not testing a *stealthy* attack anymore — you're
   just testing a model you've broken.

**Only when both Claim A and Claim B hold do you have a working, meaningful poisoned model.**
A model with 99% ASR but crashed clean accuracy is not a "stealthy backdoor" — it's a broken
model that happens to always predict one class in one direction. Always report both numbers
together, never ASR alone.

### 4.3 A sanity check most people skip: the "wrong trigger" negative control

Before you trust an ASR number, check that the model is reacting to *your specific trigger* and
not just to "any weird insertion into text." Take the same clean test examples, but insert a
**different, never-before-seen word or phrase** (not your real trigger) using the same insertion
method. Run the model. If ASR is high on this too, your model didn't learn a specific backdoor —
it just became generally sensitive to any unusual insertion, which is a much less interesting (and
less stealthy, and less publishable) finding.

### 4.4 Individual example inspection (cheap and very informative)

Pick 10–20 test examples by hand. For each one, print: original text → model's prediction, same
text + trigger → model's prediction. You want to see the prediction actually *flip* to the target
class only when the trigger is added. This is slow to do at scale but extremely good for catching
bugs (e.g., trigger insertion code silently failing, tokenizer stripping your trigger word, etc.)
early, before you waste a full training run on broken poisoned data.

### 4.5 Statistical robustness

A single training run's ASR can bounce around by several points just from random initialization
and data shuffling. Run every configuration (E1–E6) with **at least 3 different random seeds**
and report **mean ± standard deviation**. If two configurations' ASR ranges overlap heavily,
you cannot yet claim one is "better" or "more stealthy" than the other — that's a statistics
problem, not a modeling problem, and it's the single most common way empirical security papers
get criticized.

### 4.6 Visual verification (optional but convincing, and directly from the paper)

The paper's Figure 1 does exactly this: take the poisoned model, run all target-class test
examples plus your poisoned training examples through the model, grab the internal embedding
(e.g., the `[CLS]` token embedding from a BERT-style model, or the pooled sentence embedding),
reduce it to 2D with t-SNE or UMAP, and plot clean-target-class points vs. poisoned points in
different colors. For random poisoning, you should see the poisoned points form a visibly
separate little cluster off to the side. For CBS poisoning, they should sit closer to, or blended
into, the genuine target-class cluster. This single plot is often the most convincing piece of
evidence in the entire report, and it's a direct, faithful reproduction of the paper's own
methodology — just applied to text embeddings instead of image embeddings.

### 4.7 Verifying it *after* distillation (same tools, one extra number)

Everything above applies unchanged to the student models. The one extra number worth tracking:

```
ASR retention = ASR(student) / ASR(teacher)
```

- Retention near 1.0 (or above) → the backdoor survived distillation essentially intact (or even
  got amplified).
- Retention near 0 → distillation "cleaned" the model, intentionally or not.
- Retention somewhere in between → partial survival — this is actually the most scientifically
  interesting outcome, and the one your hypothesis (Section 3 of the master README) predicts is
  most likely for a *confidence-driven* backdoor specifically.

---

## 5. Experiment E1 — Clean teacher baseline

**Objective:** establish the "no attack" reference point everything else is measured against.

**Setup:**
- Dataset: a binary sentiment dataset — SST-2 or IMDB are both fine; SST-2 is smaller and faster
  to iterate on, IMDB has longer documents and is a slightly more realistic testbed later.
- Model: `bert-base-uncased` (teacher) — or `distilbert-base-uncased` if you want faster iteration
  while debugging your pipeline, then switch to `bert-base-uncased` for final numbers.
- Standard fine-tuning: cross-entropy loss on the true labels, no poisoning of any kind.
- Typical hyperparameters to start with: 3 epochs, learning rate 2e-5, batch size 16–32.

**What to record:**
- Clean accuracy (CACC) on the held-out test set. This single number is your reference for every
  other experiment's "did the model stay usable" check.
- Save the trained weights — you will need this exact model in Section 8 for the E4 (clean
  student) distillation.

**Done when:** CACC is reasonably close to published numbers for the dataset/model combo (e.g.,
BERT-base on SST-2 typically lands around 90–93% test accuracy). If you're way below that, fix
your training pipeline before touching poisoning at all — you need a working, competent clean
model as your foundation, or every later comparison is meaningless.

---

## 6. Experiment E2 — Random-poison teacher

**Objective:** the standard, naive backdoor baseline that the CBS method is compared against.

**Setup:**
1. Pick a **trigger**. Simplest possible choice for a first working version: a fixed rare token,
   e.g. insert the word `"cf"` (or any word essentially absent from the clean data) at a random
   position in the sentence. This matches the paper's philosophy — the sampling method (which
   you're studying) is independent of trigger design, so keep the trigger boring on purpose.
2. Pick a **target class**, e.g. always flip toward "positive."
3. Pick a **poison rate**, e.g. start at 5% while you're debugging the pipeline end-to-end
   (higher rate = easier to see a strong signal quickly), then reduce toward the paper's much
   smaller 0.1–1% range once everything works, for your final/reported numbers.
4. **Random selection:** from the training set, uniformly randomly sample `poison_rate × N`
   examples whose true label is *not* already the target class (poisoning an already-target-class
   example is pointless — nothing needs to flip).
5. For each selected example: insert the trigger into the text, change its label to the target
   class. Leave everything else in the training set untouched.
6. Train a fresh model from scratch (same architecture/hyperparameters as E1) on this now-mixed
   clean+poisoned training set.

**What to record:**
- CACC on the clean test set (compare to E1 — should be close).
- ASR on the triggered test set, computed exactly as described in Section 4.1.
- The "wrong trigger" negative-control ASR (Section 4.3).

**Done when:** ASR is high (rule of thumb ≥ 80%) and CACC hasn't meaningfully dropped from E1.
This confirms your trigger-insertion and label-flipping code, and your training loop, all work
correctly — E2 is also your dress rehearsal / debugging ground before the more fiddly E3.

---

## 7. Experiment E3 — CBS (decision-boundary) teacher

**Objective:** reproduce the paper's actual contribution, adapted to text. This is the core of
your project.

**Setup — this has one extra step compared to E2:**

**Step 1 — Train (or reuse) a surrogate model.**
Train a plain classifier on the *clean* training set (no poisoning at all) — architecturally,
this can be the exact same setup as your E1 clean model. In fact, **you can literally reuse E1 as
your surrogate model** — that's a nice practical shortcut and matches the paper's approach
(they use a model trained purely on clean data to estimate confidence).

**Step 2 — Compute confidence scores for every training example.**
Run every training example through the surrogate model. Take the softmax output (a probability
per class). You need two numbers per example:
- the probability assigned to the example's *true* label
- the probability assigned to the *target* class you're backdooring toward

**Step 3 — Select boundary examples.**
For each example, compute `margin = |P(true label) - P(target label)|`. Sort by margin, ascending.
Pick the `poison_rate × N` examples with the **smallest** margin (i.e., the model is *almost
equally* unsure between the true class and the target class for these) — excluding, as in E2, any
example whose true label already equals the target class. This is Algorithm 1 from the paper,
directly.

Practically: instead of hand-picking an ε threshold and filtering, it's much easier in practice to
just **rank by margin and take the top-K smallest**, where K = your poison budget. This gets you
the same set of examples without having to tune ε to hit an exact count — do this first, and only
switch to the ε-threshold formulation from the paper if you specifically want to run the ablation
in Section 5.6 of the paper (varying ε).

**Step 4 — Poison exactly as in E2.**
Same trigger, same target class, same poison rate as E2 — insert the trigger into the selected
boundary examples, flip their labels to the target class. **Keeping the trigger, target class, and
poison rate identical between E2 and E3 is essential** — the *only* thing that should differ
between these two experiments is *which* examples got selected. If you change anything else too,
you can no longer cleanly attribute any ASR/CACC/detectability difference to the selection
strategy.

**Step 5 — Train from scratch**, same as E2.

**What to record:** same three things as E2 (CACC, ASR, negative-control ASR), plus one extra
comparison:

**The key comparison for RQ1 (does CBS work in text):**
- E3's ASR may be **slightly lower** than E2's — that's expected per the paper's own trade-off
  (Remark 4.8) and is not a failure.
- E3's CACC should be **at least as good as, ideally slightly better than**, E2's — CBS poisons
  "quieter," less disruptive examples, so it should distort the model's overall behavior less.
- If you have time to implement at least one defense (ONION is the cheapest to add — it just
  flags training examples with unusually high perplexity/unnaturalness), run it against both E2
  and E3's poisoned training sets and compare **detection rate**. This is the real headline result
  the paper is built around: CBS's poisoned examples should be detected *less often* than E2's.

**Done when:** you have a side-by-side table like this for your own data:

| | CACC | ASR (real trigger) | ASR (wrong trigger, negative control) |
|---|---|---|---|
| E1 (clean) | baseline | n/a | n/a |
| E2 (random) | ~baseline | high | low |
| E3 (CBS) | ~baseline (≥ E2) | high (maybe slightly < E2) | low |

---

## 8. Experiments E4/E5/E6 — Distilling all three teachers

**Objective:** answer "does whatever is inside the teacher (nothing, a random backdoor, or a CBS
backdoor) survive being compressed into a smaller student?"

**Why distill the CLEAN teacher too (E4)?**
This is your **control condition**, and it's important: it tells you the baseline behavior of your
distillation pipeline itself. If your "clean student" (E4) somehow shows non-trivial ASR on the
trigger phrase, that's a red flag that your distillation code or trigger choice has some bug or
leak (e.g., the trigger word coincidentally already biases the clean model), completely
independent of any real poisoning effect. Always check E4's ASR first — it should be near-chance
level.

**Setup, shared across E4/E5/E6:**
- Student architecture: something meaningfully smaller than the teacher, e.g. `distilbert-base-uncased`
  if your teacher is `bert-base-uncased` (roughly half the parameters), or a small 4-layer
  transformer if you want a bigger compression gap.
- **Distillation data: a clean, untriggered split of the training data** (or a held-out clean
  corpus). Do not include any poisoned examples in the distillation set — this is what makes the
  experiment realistic and interesting (see the pipeline diagram in Section 3).
- Standard knowledge-distillation loss: a weighted mix of
  1. cross-entropy between student's predictions and the ground-truth labels, and
  2. KL-divergence between the student's softened output distribution and the teacher's softened
     output distribution (both divided by a temperature `T`, e.g. `T = 2` or `T = 4` as a
     starting point).
- Train the student to convergence (e.g., 3–5 epochs is typically enough for a distillation run on
  a small classification dataset).

**Run this three times, changing only which teacher supplies the soft labels:**
- **E4:** teacher = E1 (clean) → produces the clean student
- **E5:** teacher = E2 (random-poisoned) → produces the random student
- **E6:** teacher = E3 (CBS-poisoned) → produces the CBS student

**What to record for each student:** exactly the same three numbers as the teachers — CACC, ASR
(real trigger), ASR (negative control) — plus the **ASR retention ratio** from Section 4.7.

**Done when:** you can fill in the full six-row master table in Section 9 below, with every cell
populated across at least 3 random seeds (mean ± std).

**Optional LoRA arm (if you have time):** repeat E5 and E6 but instead of fully fine-tuning the
small student with the KD loss, initialize the student with LoRA adapters (rank 8–16 is a
reasonable starting point) on top of a frozen pretrained base and only train the adapters with the
same KD loss. This tells you whether a lighter-weight compression path behaves differently from
full distillation — worth doing once the core E1–E6 pipeline is solid, not before.

---

## 9. Master results table you're building toward

This is the single table that answers your research question. Fill it in as you complete each
experiment (report mean ± std across ≥3 seeds):

| Model | CACC | ASR (real trigger) | ASR (wrong trigger) | ASR retention vs. its teacher |
|---|---|---|---|---|
| E1 — Clean teacher | | n/a | n/a | n/a |
| E2 — Random teacher | | | | n/a (this IS the teacher) |
| E3 — CBS teacher | | | | n/a (this IS the teacher) |
| E4 — Clean student | | | | n/a |
| E5 — Random student | | | | ASR(E5)/ASR(E2) |
| E6 — CBS student | | | | ASR(E6)/ASR(E3) |

The two numbers that matter most for your actual research question are the last two cells:
comparing **ASR(E5)/ASR(E2)** against **ASR(E6)/ASR(E3)** tells you whether random-poisoned
backdoors and CBS-poisoned backdoors survive distillation at *different rates* — which is exactly
what your hypothesis (in the master README) is trying to test.

---

## 10. Common failure modes and how to debug them

| Symptom | Likely cause | Fix |
|---|---|---|
| E1 clean accuracy is poor (well below normal for the dataset) | Bad training setup, wrong learning rate, not enough epochs, data loading bug | Fix this before anything else — nothing downstream is trustworthy until E1 is solid |
| E2 ASR is low (near-chance) | Trigger insertion code isn't actually running, tokenizer is stripping/splitting your trigger word oddly, poison rate too low, too few epochs | Manually print 5 poisoned training examples after your poisoning function runs — read them with your own eyes; check the trigger word survives tokenization by decoding the tokenized input back to text |
| E2 ASR is high but CACC crashed | Poison rate too high, or trigger is too disruptive/long | Lower poison rate; shorten trigger; keep dirty-label flips limited to non-target-class examples only |
| E3 ASR is much lower than E2 (not just "slightly") | ε too strict / K too small (too few boundary examples selected, sometimes fewer than expected) | Print how many examples actually got selected; loosen the margin threshold or raise K; verify the surrogate model itself has reasonable accuracy (a bad surrogate gives meaningless confidence scores) |
| E3 and E2 select overlapping/near-identical examples | Poison rate is high enough that "boundary" and "random" stop being meaningfully different at that scale | Lower the poison rate — the CBS-vs-random distinction is most visible at small poison rates, which also matches the paper's own very low poison rates (0.1–0.2%) |
| E4 (clean student) shows non-trivial ASR | Distillation pipeline bug, or the "trigger" word coincidentally shifts predictions even without poisoning (e.g. it's a genuinely sentiment-loaded word) | Pick a more neutral, truly rare trigger word/phrase; re-check E4 in isolation before trusting E5/E6 |
| Numbers bounce around wildly between runs | Only running 1 seed | Run ≥3 seeds, report mean ± std, don't trust single-run numbers for any claim |
| Can't tell if a defense (e.g. ONION) is "working" | Not reporting both detection rate on poisoned examples AND false-positive rate on clean examples | Always report both — a defense that flags everything as poisoned has 100% detection rate but is useless |

---

## 11. Suggested execution order and time budget

1. **E1** (clean teacher) — get this rock solid first. (~0.5–1 day)
2. **E2** (random teacher) — your dress rehearsal for the whole poisoning pipeline. (~1 day)
3. Section 4's verification checklist, applied to E2, until you fully trust your ASR/CACC
   numbers and your negative control. Don't skip this. (~0.5 day)
4. **E3** (CBS teacher) — reuse everything from E2 except the selection step. (~1 day)
5. Compare E2 vs. E3 side by side (Section 7's table). This alone already answers RQ1 (does
   CBS-style selection transfer to text) and is a complete, reportable result on its own even if
   you stopped here.
6. **E4, E5, E6** (distillation of all three teachers) — this is where your genuinely new
   contribution lives. (~1.5–2 days, since you're training three student models)
7. Fill in the master table (Section 9), run ≥3 seeds for everything, and only then start
   interpreting "did it survive."
8. Only after all of the above is solid: consider adding a real defense evaluation (ONION is the
   cheapest first defense to add) and/or the optional LoRA arm.

Total: roughly 6–8 focused days to a complete, defensible first pass through E1–E6 with proper
verification at each step. Resist the urge to jump straight to distillation before E2/E3 are
verified — a distillation result built on top of an unverified or broken teacher poisoning setup
is not trustworthy, no matter how clean the distillation code itself is.
