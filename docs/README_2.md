# Does Confidence-Driven Backdoor Poisoning Survive Knowledge Distillation?
### An Empirical Study on Language Models — Master Implementation Guide

This document is the single source of truth for the project. It explains, in plain English,
what we are doing, why, and exactly how to do it step by step. Follow it phase by phase.
Every phase has a clear "done" checkpoint before you move to the next one.

---

## 1. The One-Paragraph Summary (read this first)

We are taking an idea from a 2024 paper called **"Stealthy Backdoor Attack via Confidence-driven
Sampling"** (He et al., TMLR 2024, arXiv:2310.05263). That paper poisons an image classifier by
choosing *which* training examples to corrupt very carefully — instead of picking poisoned
examples at random, it picks the examples the model is *least confident about* (the ones near the
decision boundary). This makes the backdoor much harder for defenders to detect, because it barely
changes how the model behaves overall.

That paper only tested this idea on **images**. Nobody has tested it on **text**. And nobody —
in image or text — has asked: **if you take a model poisoned this way and shrink it down using
knowledge distillation or LoRA fine-tuning (both extremely common ways to deploy smaller, cheaper
models), does the hidden backdoor survive, get stronger, get weaker, or disappear?**

That two-part question — "does it work on text?" and "does it survive compression?" — is our
research contribution. Nobody has answered it. It matters because in the real world, people
constantly take a big model and compress it into a small deployable one, so if a stealthy backdoor
silently survives that process, that is a real security problem worth documenting.

---

## 2. Literature Survey (condensed — full version in `docs/literature_survey.md`)

| Theme | Key works | What we take from them |
|---|---|---|
| **Base paper** | He et al., *Stealthy Backdoor Attack via Confidence-driven Sampling*, TMLR 2024 (arXiv:2310.05263) | Confidence-driven, boundary-aware sample selection beats random poisoning at evading detection. Tested only on images (CIFAR-10, GTSRB, ResNet). Trigger-agnostic — works with any trigger design. |
| **Classic backdoor threat model** | BadNets-style patch-trigger attacks | Foundational "poison a few training samples → controllable test-time behavior" threat model, cited for background. |
| **Textual backdoor attacks** | BadNL (Chen et al.); Hidden Killer / syntactic-trigger attack (Qi et al., ACL 2021) | Shows text is a genuinely different domain (discrete, symbolic, meaning-sensitive). Gives us ready-made datasets (SST-2, OLID, AG News) and standard metrics (ASR, clean accuracy / CACC) to reuse in Phase 2. |
| **Sample-selection / boundary-aware poisoning family** | Clean-label backdoor works; collaborative sample-selection papers; high-frequency/energy-based sample screening | Confirms this is an active family of ideas — carefully choosing *which* samples to poison, not just disguising the trigger. Positions our base paper within a broader trend. |
| **Standard textual backdoor defenses** | ONION (perplexity-based filtering); Spectral Signatures (Tran, Li & Madry); Activation Clustering | These three are the standard baseline defenses we will run in Phase 5 to test detectability. |
| **Backdoor survival through knowledge distillation** | Anti-Distillation Backdoor Attack / ADBA (Ge et al.); "Taught Well Learned Ill" (distillation-conditional backdoors); T-MTB / "Pay Attention to the Triggers" (LLM distillation transfer); W2SDefense (weak-to-strong unlearning defense); RobustKD | ADBA proves backdoors *can* survive KD in vision models — but needs a specially engineered attack to do it reliably. "Taught Well Learned Ill" and T-MTB both find that **most ordinary backdoors do NOT survive distillation by default** — survival is not automatic, it usually requires the attack to be specifically engineered for it. This is the key nuance: confidence-driven poisoning was engineered for *stealth*, not for *distillation survival*, so whether it happens to survive anyway is a genuinely open, testable question — exactly our hypothesis. |
| **LoRA / PEFT-specific backdoor risk** | Survey of backdoor attacks/defenses in LLMs (PEFT section); "LoRA Once, Backdoor Everywhere" (share-and-play LoRA attacks) | Confirms LoRA-based fine-tuning/adaptation is already a recognized real-world attack surface, motivating the LoRA arm of our Step 3. |

**The gap, stated precisely:** No existing work tests whether *confidence-driven, boundary-aware*
sample selection (specifically) transfers to text classification, and no existing work asks
what happens to *this specific class* of stealthy attack under standard knowledge distillation or
LoRA fine-tuning. Related KD-survival work either (a) is vision-only, or (b) is on generative LLMs
with adversarially-engineered "transfer-friendly" triggers — the opposite design philosophy from
confidence-driven poisoning, which is deliberately "quiet."

---

## 3. Research Questions & Hypothesis (do not change these — everything else serves them)

- **RQ1:** Does confidence-driven poisoning remain more stealthy than random poisoning when applied
  to a text classification model (not just images)?
- **RQ2:** When a confidence-driven-poisoned teacher is compressed into a smaller student (via
  knowledge distillation and/or LoRA fine-tuning), does the backdoor's attack success rate (ASR)
  go up, down, or stay the same versus the un-distilled teacher?
- **RQ3:** Does distillation make the backdoor easier or harder for standard defenses (perplexity
  filtering, spectral signatures, activation clustering) to detect?

**Hypothesis:** Because confidence-driven poisoning deliberately targets ambiguous, low-confidence
examples, the backdoor signal may be *fragile* — distillation tends to smooth over exactly the
uncertain regions of the teacher's decision surface, so the student may partially "forget" the
backdoor. We do not know the answer yet. That is the point of running the experiment.

---

## 4. Full List of Experiments (run in this order)

| # | Experiment | Purpose | Output |
|---|---|---|---|
| E1 | Train clean (unpoisoned) text classifier baseline | Reference point for clean accuracy | Clean accuracy number |
| E2 | Train text classifier poisoned with **random** sample selection | Baseline attack to compare against | ASR, clean accuracy |
| E3 | Train text classifier poisoned with **confidence-driven** sample selection (our extension of the base paper) | Core reproduction — answers RQ1 (stealth part) | ASR, clean accuracy |
| E4 | Run all 3 standard defenses (ONION, spectral signatures, activation clustering) against E1/E2/E3 | Answers RQ1 (detectability part) | Detection rate per model per defense |
| E5 | Distill E3's poisoned teacher into a smaller student via standard **knowledge distillation** | Core new experiment — answers RQ2 | Student ASR, student clean accuracy |
| E6 | LoRA-fine-tune a smaller base model using the poisoned teacher's soft labels/outputs | Second arm of RQ2 (LoRA path) | Student ASR, student clean accuracy |
| E7 | Repeat E5 and E6 using the **random-poisoned** teacher (E2) as a control | Isolates whether it's "poisoning" or specifically "confidence-driven poisoning" that behaves differently under distillation | Comparable ASR numbers |
| E8 | Run defenses (E4's trio) against the E5/E6/E7 student models | Answers RQ3 | Detection rate per student model |
| E9 | Vary poisoning rate (e.g., 1%, 5%, 10%) and repeat E3/E5/E6 | Robustness check — does the effect hold across poisoning budgets | ASR curves vs. poisoning rate |
| E10 | Vary distillation temperature / LoRA rank | Sensitivity analysis | ASR vs. hyperparameter plots |
| E11 (stretch, optional) | Repeat E1–E8 on a second dataset/domain (e.g., topic classification instead of sentiment) | Generalization check | Cross-dataset comparison table |

---

## 5. Phase-by-Phase Implementation Plan

### Phase 0 — Setup (Day 0–1)
1. Create a project folder with this structure:
   ```
   project/
     data/
     src/
       poisoning/
       models/
       distillation/
       defenses/
       eval/
     experiments/
     results/
     docs/
     README.md   <- this file
   ```
2. Set up a Python environment (conda or venv). Install: `torch`, `transformers`, `datasets`,
   `peft` (for LoRA), `scikit-learn`, `numpy`, `pandas`, `matplotlib`.
3. Pick and download your text classification dataset — **SST-2** (sentiment, binary) is the
   simplest starting point because it matches the metrics/conventions used in prior textual
   backdoor papers. Keep **AG News** (4-class topic classification) as your second dataset for E11.
4. Pick your teacher model: a small transformer such as `bert-base-uncased` or `distilbert-base-uncased`.
5. Pick your student model(s): a smaller transformer (e.g., `distilbert` if teacher is `bert-base`,
   or a 2-layer transformer / `TinyBERT`-style model if teacher is `distilbert`).
6. **Done when:** you can load the dataset, load the teacher model, fine-tune it on clean data,
   and get a reasonable clean accuracy number. This is your E1.

### Phase 1 — Reproduce Confidence-Driven Poisoning on Text (Days 2–6)
This phase answers **RQ1**.
1. Implement a trigger for text. Start simple: an **insertion trigger** (e.g., a rare, fixed
   word/phrase inserted into the input) so you have a working attack end-to-end before trying
   anything fancier like the syntactic-trigger style. You can add a stealthier trigger later as a
   stretch goal.
2. Implement **random poisoning**: pick a poisoning rate (e.g., 5% of training data), pick that
   many training examples at random, insert the trigger, flip their label to the attacker's target
   class. Train the model on this poisoned dataset. This is **E2**.
3. Implement **confidence-driven poisoning**, following the base paper's core idea:
   - Train a clean reference model first (or use an early checkpoint of the model you're about to
     poison) to get per-example confidence scores.
   - Rank training examples by how *low* their confidence is (i.e., how close to the decision
     boundary they are).
   - Select your poisoning budget (same rate as E2, e.g., 5%) from the **lowest-confidence**
     examples instead of at random.
   - Insert the same trigger, flip the label, retrain. This is **E3**.
4. Evaluate both E2 and E3 on:
   - **Clean accuracy (CACC):** accuracy on a clean, untriggered test set.
   - **Attack success rate (ASR):** on a test set where every example has the trigger inserted,
     what fraction gets classified as the attacker's target class?
5. **Done when:** E3 shows ASR comparable to E2 (both attacks work), and E3's clean accuracy is at
   least as close to the clean baseline (E1) as E2's — ideally closer, matching the "stealthiness"
   property claimed in the base paper.

### Phase 2 — Detectability Baseline (Days 7–9)
This phase finishes answering **RQ1**.
1. Implement or reuse existing implementations of the three standard textual backdoor defenses:
   - **ONION** (perplexity-based outlier word detection)
   - **Spectral signatures** (looks for outlier patterns in the model's internal representations
     of poisoned vs. clean examples)
   - **Activation clustering** (clusters hidden-layer activations to separate poisoned from clean
     examples)
2. Run all three defenses against E1 (should find nothing — sanity check), E2 (random poisoning),
   and E3 (confidence-driven poisoning).
3. Record, for each defense × each model: detection rate (fraction of poisoned examples correctly
   flagged) and false-positive rate on clean data.
4. **Done when:** you have a clear table showing whether confidence-driven poisoning (E3) is indeed
   harder to detect than random poisoning (E2) in the text domain, mirroring the base paper's image
   result. This is your first genuinely new finding.

### Phase 3 — Compress the Poisoned Teacher (Days 10–16)
This phase directly answers **RQ2** — the heart of your contribution.
1. **Knowledge distillation path (E5):**
   - Take the E3 poisoned teacher.
   - Pick a smaller student architecture.
   - Distill using a **clean, untriggered** distillation dataset (this matters — it's the
     realistic scenario: the person distilling the model doesn't have your poisoned data, they use
     their own clean data and just borrow the teacher's soft labels/logits).
   - Standard KD loss: combination of soft-label distillation loss (KL divergence between teacher
     and student output distributions, with temperature) and normal cross-entropy on true labels.
   - Train the student to convergence.
2. **LoRA fine-tuning path (E6):**
   - Take a smaller base model (not necessarily the same architecture as the teacher).
   - Fine-tune it with LoRA adapters, using the poisoned teacher's outputs as guidance (teacher
     soft labels on a clean dataset, similar setup to E5 but using LoRA/PEFT instead of full
     fine-tuning of the student).
   - Train to convergence.
3. **Control condition (E7):** Repeat both E5 and E6, but starting from the E2 **random-poisoned**
   teacher instead of the E3 confidence-driven teacher. This tells you whether any survival/decay
   effect you see is specific to confidence-driven poisoning or just a general property of
   poisoning-then-distilling.
4. Evaluate every student model (from E5, E6, E7) on ASR and clean accuracy, exactly as in Phase 1.
5. **Done when:** you have a clean comparison table:

   | Model | ASR (teacher) | ASR (KD student) | ASR (LoRA student) |
   |---|---|---|---|
   | Random-poisoned (E2/E7) | ... | ... | ... |
   | Confidence-driven (E3/E5/E6) | ... | ... | ... |

   This table is the central result of the paper.

### Phase 4 — Does Distillation Change Detectability? (Days 17–19)
This phase answers **RQ3**.
1. Re-run the three defenses from Phase 2 against every student model produced in Phase 3.
2. Compare detection rates: teacher vs. KD student vs. LoRA student, for both random and
   confidence-driven poisoning.
3. **Done when:** you can say, with numbers, whether compression makes the backdoor easier or
   harder to catch — and whether that answer differs between random and confidence-driven attacks.

### Phase 5 — Robustness & Sensitivity Analysis (Days 20–24)
1. **E9 — poisoning rate sweep:** repeat E3 → E5/E6 at multiple poisoning rates (e.g., 1%, 5%,
   10%). Plot ASR (teacher and student) vs. poisoning rate.
2. **E10 — hyperparameter sweep:** vary distillation temperature (e.g., T = 1, 2, 4, 8) and LoRA
   rank (e.g., r = 4, 8, 16). Plot ASR vs. each.
3. **Done when:** you have line/bar charts showing how stable your Phase 3 finding is across these
   settings — this is what turns a single data point into a defensible empirical claim.

### Phase 6 (Optional / Stretch) — Second Dataset (Days 25–28)
1. Repeat Phases 1–4 on AG News (or another text classification dataset) to check the finding
   generalizes beyond sentiment classification.
2. **Done when:** you have a second, smaller version of the Phase 3 table for cross-dataset
   comparison.

### Phase 7 — Write-Up (Days 25–30, overlaps with Phase 6)
1. Structure the paper/report as: Introduction → Related Work (use Section 2 of this README) →
   Method (poisoning + distillation setup) → Experimental Setup → Results (Phases 1–5 tables and
   plots) → Discussion (was the hypothesis right? what does it mean practically?) → Limitations →
   Conclusion.
2. Be explicit and honest in the paper about what is genuinely new (text domain + distillation
   survival study) versus what is reused (the base paper's sampling method, standard KD/LoRA
   recipes, standard defenses).
3. Include the full literature survey table (Section 2) as your Related Work section, expanded
   into prose paragraphs.

---

## 6. Metrics Reference Sheet

| Metric | Definition | Used in |
|---|---|---|
| **CACC (Clean Accuracy)** | Accuracy on a clean, untriggered test set | All phases |
| **ASR (Attack Success Rate)** | % of triggered test examples classified as the attacker's target label | All phases |
| **Detection Rate** | % of poisoned training examples correctly flagged by a defense | Phase 2, 4 |
| **False Positive Rate** | % of clean examples incorrectly flagged as poisoned | Phase 2, 4 |
| **ASR retention** | ASR(student) / ASR(teacher) — how much of the attack survives compression | Phase 3 (your key new metric) |

---

## 7. Practical Tips
- Keep poisoning rate, trigger design, and target label **identical** between the random (E2) and
  confidence-driven (E3) conditions — the only variable that should change is *which* examples get
  poisoned. This is essential for a fair comparison.
- Always distill/LoRA-tune on a dataset that does **not** contain the poisoned examples — that's
  the realistic threat model (a downstream user distilling a public teacher on their own clean
  data).
- Log everything (poisoning rate, trigger, random seed, hyperparameters) per run so results are
  reproducible — put this in `experiments/<run_name>/config.json`.
- Run at least 3 seeds per configuration and report mean ± std — single-run numbers are not
  reliable enough for a "survives vs. doesn't survive" claim.

---

## 8. Deliverables Checklist
- [ ] Phase 0: working environment + clean baseline (E1)
- [ ] Phase 1: random and confidence-driven poisoned models on text (E2, E3)
- [ ] Phase 2: defense evaluation table (E4)
- [ ] Phase 3: distilled/LoRA student results + control condition (E5, E6, E7)
- [ ] Phase 4: post-distillation defense evaluation (E8)
- [ ] Phase 5: poisoning-rate and hyperparameter sensitivity plots (E9, E10)
- [ ] Phase 6 (optional): second dataset replication (E11)
- [ ] Phase 7: full written report with literature survey, results tables, and discussion
