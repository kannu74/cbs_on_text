# Concrete Spec: Datasets, Models, Triggers, and Defenses

### Companion to `EXPERIMENTS_README.md` — this file answers "give me exact names, not categories"

Everything below is a specific, checked-in decision, not a menu. Where I give you an alternative,
I say explicitly when to use it. If you just want to start coding today, use the items marked
**[DEFAULT]** and ignore the rest until you need it.

---

## 1. Datasets — exact names

| Dataset | Role | Size | Classes | Why this one |
|---|---|---|---|---|
| **SST-2** (Stanford Sentiment Treebank, binary) **[DEFAULT — start here]** | Primary dataset for E1–E6 | ~67k train / 872 dev / 1821 test | 2 (positive/negative) | Short sentences → fast to train/iterate on. It's the single most-used dataset in textual backdoor papers (used by BadNL, ONION, RAP, OpenBackdoor's own benchmarks), so your numbers are directly comparable to published baselines. Load via Hugging Face `datasets` as `glue`, subset `sst2`. |
| **IMDB** | Secondary dataset — run once your SST-2 pipeline works, to check the finding isn't an artifact of very short text | 25k train / 25k test | 2 (positive/negative) | Full movie reviews (hundreds of words) instead of one sentence — tests whether a trigger inserted into a long document is as effective/detectable as in a short one. Load via `datasets` as `imdb`. |
| **AG News** | Optional stretch dataset — multi-class generalization check | 120k train / 7.6k test | 4 (World/Sports/Business/Sci-Tech) | Only add this after E1–E6 work on SST-2. Tests whether CBS's boundary-selection idea (which needs a specific "target class" to measure margin against) still behaves the same way when there are more than 2 classes to be confused with. Load via `datasets` as `ag_news`. |
| **HSOL / OLID** (hate speech / offensive language) | Optional — only if you want a "high-stakes" classification task for the write-up | ~14k (OLID) | 2 | Not required for the core study; only pull this in if your report wants a real-world-consequential example, since a hate-speech classifier being backdoored is a more concrete harm story than a sentiment classifier being backdoored. |

**Recommendation:** build and verify the entire E1–E6 pipeline end-to-end on **SST-2 only** first.
Only once Section 9's master table (from `EXPERIMENTS_README.md`) is fully populated and sane on
SST-2 should you spend the extra compute re-running on IMDB or AG News.

---

## 2. Models — exact names

### 2.1 Teacher model
**`bert-base-uncased`** (110M parameters) **[DEFAULT]**
- Hugging Face: `AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)`
- This is the model you poison (E1/E2/E3).

*Faster-iteration alternative while debugging your pipeline (not for final numbers):*
`distilbert-base-uncased` (66M) — trains roughly 2x faster, use it to shake out bugs in your
poisoning/training code cheaply, then switch to `bert-base-uncased` once the pipeline is correct.

### 2.2 Surrogate model (for computing confidence scores in E3/CBS)
**Reuse E1's trained clean `bert-base-uncased`** — you already have it, it's already trained on
clean data only, and this is exactly what the original paper's own Algorithm 1 calls for (a model
pretrained on the clean training set, used purely to estimate confidence). Do not train a separate
surrogate unless you specifically want to test whether CBS still works when the surrogate is a
*different, weaker* architecture than the teacher (see 2.4 below — that's a legitimate follow-up
experiment, not a requirement for your first pass).

### 2.3 Student model (for distillation, E4/E5/E6)
**`distilbert-base-uncased`** (66M parameters, ~40% smaller, ~60% faster) **[DEFAULT]**
- This is a natural, well-known BERT→DistilBERT compression pair, so any reviewer immediately
  understands the setup, and DistilBERT is itself explicitly designed via knowledge distillation
  from BERT, which makes it the most standard, defensible choice here.

*More aggressive compression alternative (stretch goal, do after the DistilBERT result works):*
A **4-layer MiniLM-style model** (e.g. `nreimers/MiniLM-L6-H384-uncased`, ~22M parameters) — use
this if you want to test whether the backdoor's survival rate depends on *how much* compression
happens, not just *whether* it happens at all. This turns "does it survive distillation" into
"does survival degrade smoothly or fall off a cliff as compression increases" — a stronger result
if you have the time budget.

### 2.4 Optional: second teacher architecture (for a cross-architecture transfer check)
**`roberta-base`** (125M) — only add this if you want to reproduce the paper's own
ResNet18→VGG16 transfer experiment (Table 1, "ResNet18 → VGG16" columns) in text form: select
boundary samples using the `bert-base-uncased` surrogate, but then train `roberta-base` from
scratch on that same poisoned set. This checks whether CBS's selected examples are useful even
when the attacker doesn't know the victim's exact architecture — a realistic black-box assumption.
Treat this as a stretch goal, not part of the core E1–E6 pipeline.

### 2.5 Optional: LoRA arm base model
If you do the optional LoRA distillation path mentioned in `EXPERIMENTS_README.md` Section 8, use
LoRA adapters (via Hugging Face `peft`, rank `r=8`, `lora_alpha=16`, targeting the query/value
projection matrices) on top of **`distilbert-base-uncased`** — i.e., same student model as your
default KD path, just trained with adapters instead of full fine-tuning, so it's a clean
apples-to-apples comparison of *how* you compress, holding the *what* (student architecture)
constant. Don't switch to a decoder-only LLM (GPT-2, Llama, etc.) for this — that would introduce
architecture as a second variable and muddy the comparison.

---

## 3. Text attacks / triggers — exact list

Use the paper's own framing: the sampling method (Random vs. CBS) is one axis, and the trigger
design is a completely separate, independent axis. You want at least one trigger working well
before worrying about trigger diversity. In order of when to implement them:

| # | Attack / trigger name | Type | How it works | When to use it |
|---|---|---|---|---|
| 1 | **Word-insertion trigger** (BadNL-style / a simplified version of Dai et al.'s LSTM backdoor attack) | Token-level, dirty-label | Insert a fixed rare token (e.g. `"cf"`, `"mn"`, `"bb"` — pick something that essentially never appears naturally in the dataset) at a random position in the sentence | **[DEFAULT] — implement this first.** Simplest to code, easiest to debug, matches the paper's own BadNet/patch-trigger philosophy of "a small, fixed, obvious pattern." |
| 2 | **InsertSent** (Dai, Chen & Li, 2019, "A Backdoor Attack Against LSTM-based Text Classification Systems") | Sentence-level, dirty-label | Insert one full fixed neutral sentence (e.g. *"I watched this movie."*) into the text, instead of a single word | Second trigger to implement, once #1 works — more natural-looking than a single odd token, gives you a trigger-diversity data point to check CBS's "trigger-agnostic" claim in text. |
| 3 | **SynBkd** ("Hidden Killer," Qi et al., ACL 2021) | Syntactic, dirty-label, no fixed token at all | Rewrite the *entire sentence* to follow a rare grammatical template (e.g. an S(SBAR)(,)(NP)(VP)(.) structure) using a syntactically controlled paraphraser | Stretch goal — genuinely stealthy trigger with no fixed word for perplexity-based defenses (ONION) to latch onto, which makes it a good stress test for whether CBS's benefit holds up against a harder-to-detect trigger too, or whether it's redundant with an already-stealthy trigger. |
| 4 | **StyleBkd** (Qi et al., 2021, "Mind the Style of Text!") | Style-level, dirty-label | Rewrite the sentence in a distinctive style (e.g. Biblical, poetic) using a style-transfer model as the "trigger" | Same role as #3 — optional trigger-diversity stretch goal, skip unless you have time after the core pipeline is done. |
| 5 | **Weight-poisoning / RIPPLES** (Kurita, Michel & Neubig, 2020) | Different threat model — poisons pretrained embeddings directly, not just the fine-tuning data | Not a "sample selection" attack at all — mentioned here only so you know it exists and don't confuse it with your setup. Out of scope for this project; your threat model (poison a fraction of the *fine-tuning* training data) is the standard Type-I attack from the base paper, not weight poisoning. | Do not implement — just be aware it's a different family so you don't accidentally conflate it in your literature review. |

**Recommendation:** implement trigger #1 for your entire core pipeline (E1–E6). Only add trigger
#2 (InsertSent) as a robustness check once the master table from `EXPERIMENTS_README.md` is fully
populated with trigger #1 — repeating the whole E1–E6 pipeline with a second trigger type is a
strong, low-effort addition to your final report ("does the Random-vs-CBS gap hold for a
completely different kind of trigger too?").

---

## 4. Defenses — exact list and how to run them

Split into two families, matching what the base paper itself tests, translated into text-native
equivalents:

### 4.1 Detection / filtering defenses (apply directly to text — NLP-native)

| Defense | Paper | What it actually does | Implementation note |
|---|---|---|---|
| **ONION** [DEFAULT — implement first] | Qi et al., EMNLP 2021, "A Simple and Effective Defense Against Textual Backdoor Attacks" | For each sentence, remove one word at a time and check how much the sentence's language-model perplexity drops when that word is removed. Words whose removal causes a big perplexity drop are flagged as likely trigger words. | Cheapest defense to implement — you just need a pretrained language model (e.g. GPT-2) to score perplexity. No need to retrain anything to test it. |
| **STRIP** (text-adapted) | Gao et al., 2019 (adapted for NLP in several later papers) | Perturb each test input in several ways (e.g. randomly replace/blend words from other examples) and measure the entropy of the model's predictions across perturbations. Poisoned inputs tend to give unusually *low*-entropy (overconfident) predictions even under heavy perturbation, because the trigger dominates the decision. | Implement second — needs no retraining, just repeated inference with perturbed copies of each input. |
| **RAP** | Yang et al., EMNLP 2021, "RAP: Robustness-Aware Perturbations for Defending against Backdoor Attacks on NLP models" | Inserts one specific perturbation word at the start of every test input and compares how much the target-class probability shifts, versus the shift observed on a clean validation set under the same perturbation. Large, unusual shifts flag the input as likely triggered. | Optional third defense — a bit more setup (needs a small clean validation set) but is a standard baseline in nearly every textual-backdoor paper, so including it makes your defense comparison more credible. |

### 4.2 Representation / embedding-space defenses (the direct text equivalent of what the base paper itself uses on images)

| Defense | Paper | What it actually does | Implementation note |
|---|---|---|---|
| **Spectral Signature** [DEFAULT — implement this one, it's a direct match to the original paper] | Tran, Li & Madry, NeurIPS 2018 | Looks at the model's internal representation (for text: the `[CLS]` embedding or pooled sentence embedding) for every example of the target class, and flags examples whose representation has an unusually strong "signature" along the top singular vector of the representation covariance matrix — i.e., statistical outliers in embedding space. | You already computed embeddings for the t-SNE visualization in `EXPERIMENTS_README.md` Section 4.6 — reuse that exact embedding extraction code here. This is literally the same defense the base paper itself uses (their "SS" row in Table 1), just applied to BERT sentence embeddings instead of ResNet image embeddings. |
| **Activation Clustering** | Chen et al., 2018 | Clusters the same kind of internal-layer embeddings (per class) into 2 clusters (e.g. k-means with k=2) and treats the smaller cluster as likely-poisoned. | Second embedding-space defense to add — cheap once you already have the embeddings extracted for Spectral Signature above. |

### 4.3 Training-time defense (optional, more work, but worth knowing about)
**CUBE** (Cui et al., NeurIPS 2022, from the `OpenBackdoor` toolkit paper) — clusters training
examples in embedding space (regardless of the trigger's type — token, syntax, or style) and
removes suspicious clusters before training even starts. Only add this if you want a defense that
specifically doesn't rely on assuming a fixed-token trigger, which matters if you also implemented
trigger #3 (SynBkd) or #4 (StyleBkd) from Section 3.

### 4.4 How to actually evaluate a defense, step by step

For every defense × every poisoned model (E2-random and E3-CBS), report **two** numbers, never
just one:

1. **Detection rate (recall):** of the training examples you *know* are actually poisoned (you
   have ground truth — you poisoned them yourself), what fraction did the defense correctly flag?
2. **False-positive rate:** of the *clean* training examples, what fraction did the defense
   incorrectly flag as poisoned? A defense with 100% detection rate but 100% false-positive rate
   is worthless — it flagged everything.

Then run the **actual downstream test**: remove everything the defense flagged, retrain the model
from scratch on the filtered set, and check:
3. **Post-defense ASR** — did the attack success rate actually drop after filtering? (This is the
   real-world-relevant number — a defense can have a mediocre detection rate and still crush ASR
   if it happens to catch the *most influential* poisoned examples, or vice versa.)
4. **Post-defense CACC** — did clean accuracy survive the filtering, or did the defense also
   throw away too many legitimate clean examples?

**The core comparison you're building toward:** for each defense, is Detection Rate(E3-CBS) lower
than Detection Rate(E2-random)? Is Post-defense ASR(E3-CBS) higher than Post-defense ASR(E2-random)
(i.e., CBS's backdoor survives the defense better)? If yes to both, you've reproduced the base
paper's central finding — in text.

---

## 5. Quick-reference summary card

```
DATASET (default):     SST-2   (secondary: IMDB, stretch: AG News)
TEACHER MODEL:         bert-base-uncased
SURROGATE MODEL:       same as teacher (reuse E1's clean model)
STUDENT MODEL:         distilbert-base-uncased  (stretch: MiniLM-L6-H384)
2ND TEACHER ARCH:      roberta-base  (optional, transfer-check only)
TRIGGER (default):     word-insertion, e.g. insert "cf" at a random position
TRIGGER (stretch):     InsertSent (fixed sentence) -> SynBkd -> StyleBkd
DEFENSES (default):    ONION  +  Spectral Signature
DEFENSES (add next):   STRIP (text)  +  Activation Clustering
DEFENSES (stretch):    RAP  +  CUBE
```

If you build only what's marked **default** across every section above, you have a complete,
coherent, publishable-scope project. Everything marked "optional/stretch" is there so you know
what a stronger version looks like once the default pipeline is working end to end.
