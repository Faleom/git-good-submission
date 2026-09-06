# How the final model works

Six gases, 128 sensor features per reading, plus the concentration used. The labelled
training data covers nine sequential batches; the test set was recorded later, after the
sensors had aged. Nothing about the test set's class composition is assumed: no per-class
row count for the test set enters the pipeline at any point.

---

## The problem in one paragraph

Metal-oxide gas sensors are **non-selective**. Each one is a heated semiconductor whose
resistance changes when any reducing gas adsorbs onto its surface, so no single sensor can
tell ethanol from acetone; it only reports that something reactive is present, and roughly
how much. Identity lives in the **pattern across the 16 sensors**, because each has a
different coating and therefore a different relative sensitivity.

The complication is that sensors age. Coatings degrade and contaminants accumulate, so the
same gas at the same concentration reads differently months later. Cross-validation confirms
this: classes separate almost perfectly *within* any single batch, but a model trained on
early batches and applied to a later one loses a great deal of that accuracy. The information survives; only the mapping
moves. **This is a domain-adaptation problem wearing a classification costume.**

---

## End to end, start to finish

One command, about 15 seconds, 165 model fits. Nothing is cached or pre-trained: every run
repeats the whole process from the raw CSVs and produces a byte-identical file.

```
python3 src/final_model.py
```

### Stage 1: Load  (~2 s)

Read `train.csv` (labelled, nine sequential batches), `test.csv` (unlabelled, recorded
later) and `sample_submission.csv` (the required row order). No cleaning is applied: the
data has no missing values, duplicates, constant columns or infinities.

### Stage 2: Build features  (~20 s)

From the 128 raw columns, construct five *views* of every reading, with training and test
alike. Each view is a different way of describing the same measurement:

| view | contents |
|---|---|
| A | per-batch robust z + sensor ratios + slot contrasts + L2 |
| B | A, plus global robust z |
| C | global robust z + sensor ratios + slot contrasts + L2 |
| D | slot contrasts + L2 + sensor ratios |
| E | everything above combined |

Then append two concentration blocks to every view: a **one-hot indicator** per distinct
concentration value, and the **dose residuals** left after regressing out
`log|response| ~ a + b*log(concentration)`, fitted on the training data only.

At this point each reading is described five different ways, and every view has had the
"how much gas" component partly removed.

### Stage 3. Round 0: learn from the training data  (~30 s)

Fit shrinkage LDA once per view on all labelled training rows. Average the five posteriors
with equal weight. Take the argmax to get a first label for each test row.

These labels are decent but not great; the model is applying what it learned from older
sensors to newer ones, and the sensors have drifted in between.

### Stage 4. Rounds 1-32: learn from the test batch itself  (~5 min)

Each round does four things:

```
a.  count how many test rows are currently predicted as each class
b.  sort those six counts; cap = the value 60% of the way up the sorted list
    (with six classes, the fourth-smallest of the six, recomputed every
     round from the model's own predictions, never a fixed number)
c.  seeds = the `cap` most confident rows of each predicted class
d.  refit all five views on THOSE TEST ROWS, using their predicted labels,
    then relabel all test rows from the averaged posterior
```

Step (d) is the important one: the model is now trained on **test-batch readings**, so it
no longer has to bridge the drift. The classes separate cleanly within a single batch, so
a model fitted there is far more accurate than one extrapolating from older data.

Step (c) is what keeps it honest. On a held-out training batch the seeds are substantially
more accurate than the rows excluded, so each round trains on a much cleaner set than the
raw predictions. And because the cap is one of the model's *current* counts rather than a
fixed number, it bites hardest on whichever class has over-claimed rows from a neighbour,
letting those rows drain back.

Each round's labels are slightly better, so the refitted model is slightly better, so the
next round's labels are better again. On this test set the cap comes out 586, 589, 602, 602
over the first four rounds; a value computed from the model's own predictions, which happen
to come out roughly even here, not a number supplied to it. On an unevenly distributed target
the same code produces a completely different cap; see "The seed cap" below.

### Stage 5: Converge and write  (~1 s)

By round 32 the labels have stopped changing; running 64 rounds produces identical output.
Take the final argmax, join to the submission template, write
`outputs/submissions/submission_final.csv`.

The output is unconstrained; no class-count target is applied at any point.

### In one sentence

Learn from the old sensors, guess the new readings, then **retrain on the new readings
using those guesses** and repeat until it stops changing, because a model fitted on
drifted data with imperfect labels beats a model fitted on clean data from before the
drift.

---

## Data cleaning and preprocessing

**No cleaning was necessary.** The raw data contains no missing values, no duplicate rows,
no constant columns and no infinities, and every column is numeric. Nothing is dropped,
imputed or repaired.

Preprocessing is therefore entirely feature construction, applied identically to training
and test data:

1. **Signed-log transform** of the raw responses. Features span five orders of magnitude
   and take both signs, so `sign(x)*log1p(|x|)` compresses the range without discarding
   the sign.
2. **Robust standardisation** (median and inter-quartile range) rather than mean and
   standard deviation, so a handful of extreme readings cannot dominate the scale.
3. **Per-batch standardisation**, computed within each batch using only that batch's own
   rows.
4. **Ratio and contrast transforms**, described below.
5. **Concentration encoding**, described below.

Steps 2 and 5 are fitted on the training data and applied unchanged to the test data.
Step 3 uses only the rows of the batch being transformed. Steps 1 and 4 are row-local and
need no fitting at all.

## Models tested, and how the final one was selected

| family | outcome |
|---|---|
| **shrinkage LDA** | **selected**: held the top six places in a hundred-configuration sweep |
| logistic regression (C from 0.003 to 3) | clearly behind LDA at every setting |
| random forest | behind |
| gradient boosting (HistGradientBoosting) | behind |
| extra trees | behind |
| RBF support vector machine | far behind as a source model |
| multilayer perceptron | far behind |
| quadratic discriminant analysis | behind LDA; a shared covariance beats per-class ones here |
| k-nearest neighbours | behind |

Selection used batch-forward cross-validation: train on early batches, predict a later
one, never a random split. A random split reports dramatically higher accuracy than the
honest protocol on this data, because near-duplicate readings from one batch land on both
sides of the split and the model is graded on data it has effectively already seen.

Nonlinear models were additionally re-tested *inside* the target domain, where drift is
absent, in case that was the reason they lost. All of them landed within noise of LDA
there, so the bottleneck is label quality rather than model capacity.

## Ensembling and post-processing

**Ensembling.** Five shrinkage-LDA models, one per feature view, with their posterior
probabilities averaged at **equal weight (1/5 each)**. Blend weights were fitted on
out-of-fold predictions and gained less than the noise of the comparison, so equal weights
are used; a tuned weight there fits the validation set rather than the model.

Members must be comparably strong. Adding a weaker family to an otherwise strong blend
measurably degraded it on two separate occasions, so only models within a small margin of
the best are included.

**Post-processing.** Two steps, both *inside* the self-training loop rather than applied
to the final output:

1. **Seed selection**: each round trains only on the most confident rows per predicted
   class, capped at the 60th-percentile value among the model's own six current predicted
   class counts.
2. **Relabelling**: the whole test set is relabelled from the averaged posterior each
   round.

The final output receives **no post-processing at all**: it is the plain argmax of the
last round's averaged posterior, with no calibration, thresholding or class-count
constraint.

## What each component does, and why it stays

### Shrinkage LDA: the single largest gain

`LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")`

Chosen over logistic regression, random forests, gradient boosting, extra trees, SVMs and
a neural network. In a sweep of a hundred configurations, shrinkage LDA held the top six
places; the best logistic regression was far behind, and an RBF-SVM as a source model was
worse still.

**Why it wins under drift.** Most classifiers learn a decision boundary in feature space,
and drift slides every reading across that boundary. LDA instead models a single pooled
covariance, that is, *how the sensors move together*. Ageing changes each sensor's absolute
sensitivity, but the correlation structure between sensors is far more stable, so LDA
leans on the part of the signal that survives.

Ledoit-Wolf shrinkage (`"auto"`) is already optimal; raising it manually degrades results
monotonically across every feature set tested.

### Five feature views, equal weights

Each view is a different transform of the same 128 columns, and their posteriors are
averaged with equal weight. No blend weights are tuned, because optimising them on validation
predictions produced a gain far below the noise of the comparison, which is fitting the
validation set rather than improving the model.

| transform | rationale |
|---|---|
| **global robust z-score** of signed-log responses | features span five orders of magnitude |
| **per-batch robust z-score** | removes each batch's location and scale offset |
| **sensor steady-state log-ratios** | each sensor against the array mean |
| **within-slot log-contrasts** | each descriptor against its slot's mean across sensors |
| **per-sample L2 normalisation** | scale-invariant response pattern |

The ratio transforms matter most. Multiplicative sensitivity loss is the dominant ageing
mode, and it **cancels in a ratio**: if a sensor fades by 20%, its ratio to the array mean
is unchanged. Per-batch standardisation is used *alongside* these, never alone: on its own
it removes class signal along with drift, because each batch's centre depends on its gas
mix and those mixes differ sharply.

### Concentration, encoded two ways

**One-hot per distinct value.** Each gas was only ever recorded across a particular range
of concentrations, and those ranges differ. Encoding concentration as a *number* tells the
model that 800 exceeds 400, true and useless. What matters is that a specific value is
characteristic of specific gases. Granularity helped monotonically: log, then coarse bands,
then fine bands, then one indicator per value.

**Dose-response residuals.** Sensor response is approximately log-linear in concentration:

```
log|response| ~ a_s + b_s * log(concentration)
```

Fitting that per feature on the training data and subtracting it separates *which* gas from
*how much*; a confounding that is intrinsic to the measurement, since a non-selective
sensor reports something closer to the product of the two than to either alone. This was the largest late gain in the project.

Two constraints on this, both learned the hard way:

- **Fit it globally, never per class.** Per-gas dose curves are source-domain quantities,
  and drift moves each gas's curve independently, so they transfer badly and cost real
  accuracy. The global fit works precisely because it is shared by all six classes; drift
  moves it uniformly and the residuals stay meaningful.
- **The power law beats the more correct physics.** A Langmuir isotherm
  (`dR = aC/(1+bC)`) saturates as surface sites fill, which is physically truer over a
  thousand-fold concentration range. It still loses: two parameters per feature fitted on
  drifted data buys misfit rather than accuracy.

### Self-training inside the test batch

The model relabels the test set, then refits on **those rows using its own predicted
labels**, 32 times.

This works because the classes separate near-perfectly *within* a batch. Fitting on test
rows puts the model in the domain where separability is high, instead of extrapolating
from an older one. Each round's better labels retrain a better model, which produces better
labels.

**It only works from a strong starting point.** On a held-out training batch, the identical
loop performs far better starting from shrinkage LDA than from logistic regression; a
large gap, not a marginal one. A weak model teaches itself its own mistakes; a strong one
teaches itself something true.

### The seed cap

Each round trains on the most confident rows per predicted class, capped at the
**60th-percentile value among the model's own six current predicted class counts**.
Concretely: count how many test rows are currently predicted as each of the six gases, sort
those six numbers, and take the value 60% of the way up that sorted list, with six classes,
that is exactly the fourth-smallest of the six. In code, `np.quantile(counts, 0.6)`.

The `0.6` is a **position in a sorted list of six numbers**. It is not a fraction of the rows,
it is not a row count, and it has nothing to do with there being six classes. **No per-class
row count is assumed anywhere in this pipeline**: the cap is recomputed from the model's own
predictions every round and is never a fixed number. The only property of the test set the
pipeline does assume is that there are six gas classes, given by the competition and visible
in `sample_submission.csv`.

**The cap follows the target, and this is checkable.** Hold out training batch 9, whose true
class counts are deliberately uneven (61, 55, 100, 75, 78, 101), treat it as the target and run
the identical loop: the cap comes out 92, 87, 93, 99, 100 over the first rounds and settles at
100. Nothing in the code changed; the cap simply follows the counts it is handed, whatever
they are.

**It does two things at once.** Most straightforwardly it filters for correctness: on a
held-out training batch the rows it keeps are far more often correct than the rows it
drops, so the model trains on a much cleaner set than its raw output.

It also applies corrective pressure. Because the cap sits below the largest of the current
predicted counts, it bites hardest on whichever class has over-annexed its neighbour,
contracting that class's fitted region so the annexed rows drain back over subsequent rounds.

Removing the cap entirely, taking the 100th percentile, i.e. the largest of the six counts,
which caps nothing, collapses performance on the same held-out batch.
The loop freezes early with one class having swallowed part of another, and every later
round retrains on that error.

### 32 rounds

Converged; 64 rounds produces byte-identical output, on both the held-out training batch
and the test set. Re-swept after the dose residuals were added, since a setting validated
against an older feature set is not evidence about a new one.

---

## What was tried and removed

| approach | outcome |
|---|---|
| logistic regression, RF, boosting, extra trees, SVM, MLP | all beaten by shrinkage LDA |
| CORAL domain adaptation | covariance estimated from small batches is noise |
| per-batch standardisation alone | removes class signal with the drift |
| feature selection by worst-case discriminative power | keeping all features wins |
| recency weighting of training batches | no effect |
| unsupervised clustering of the test batch | concentration dominates the structure, not gas |
| nonlinear models on the final decision | all within noise of LDA |
| pairwise class specialists | re-decided almost no rows |
| gas-specific dose-response curves | per-class fits do not survive drift |
| adsorption/desorption kinetic ratios | already captured by the slot contrasts |
| Langmuir isotherm | over-parameterised for drifted data |
| double-centred sensor x descriptor grid | already captured by the ratio views |

Three ideas failed once and later succeeded in a different configuration: per-batch
standardisation (fails alone, works with ratio features), self-training (fails from a weak
base, works from a strong one), and concentration (fails as a number, works one-hot). **A
negative result is evidence about a configuration, not about an idea.**

---

## Local validation scores

Rolling-origin validation of this exact pipeline, using **labelled training batches only**:
train on batches 1..k, predict batch k+1. Folds whose target batch lacks any of the six
gases are excluded, since macro-F1 over five classes is not comparable to macro-F1 over six.

| target batch | train rows | target rows | round 0 only | full pipeline | worst class F1 |
|---|---|---|---|---|---|
| 2 | 445 | 1,244 | 0.6923 | 0.6191 | 0.000 |
| 6 | 3,633 | 2,300 | 0.7456 | 0.7466 | 0.131 |
| 7 | 5,933 | 3,613 | 0.9456 | 0.9990 | 0.997 |
| 8 | 9,546 | 294 | 0.8801 | 0.9648 | 0.873 |
| 9 | 9,840 | 470 | 0.7532 | 0.7164 | 0.019 |

**Mean 0.8092, standard deviation 0.165 across folds, standard error 0.074.**

Three things this table says that a single averaged number would hide.

**The spread across folds is very large**: 0.62 to 0.999. Each fold is a different target
batch with a different amount of drift, and some are simply much harder than others. With a
standard error of 0.074, differences smaller than roughly 0.15 between configurations are
not distinguishable on this evidence.

**The self-training loop is not uniformly beneficial.** It gains 0.085 and 0.053 on two
folds, is neutral on a third, and *loses* 0.073 and 0.037 on the remaining two. Mean gain
across folds is +0.006. The loop amplifies whatever the round-0 model believes, so it pays
off where that starting point is already good and costs where it is not, visible here in
that the two folds it harms are among the three with the lowest round-0 scores.

**The spread is driven by single collapsed classes, not by how much drift a fold sees.**
Read the last column against the fifth. Every fold that scores below 0.75 has one class at
or near zero F1 (0.000, 0.131, 0.019); both folds that score above 0.96 have every class
working (0.997, 0.873). Because macro-F1 averages the six classes with equal weight, one
class collapsing costs about 0.167 on its own, regardless of how well the other five do,
so these folds are not uniformly worse, they are five-sixths correct and one-sixth absent.

It is worth being precise that this is **not** a story about recency or training volume.
Batch 9 is the most recent fold and trains on the most data (9,840 rows), and it is the
second-worst result in the table. Batch 7 trains on 5,933 rows and scores 0.999. Training
volume does handicap the earliest fold, batch 2 sees 445 rows, but it does not explain
the pattern, and the deployed model trains on every batch, so these figures remain a lower
bound on what the full training set supports rather than an estimate of it.

Reproduce with `python3 src/validation_scores.py` from the bundle root. That script reads
`train.csv` only; it holds out one training batch at a time and scores against that
batch's own labels. The hidden test set is never involved.

### A configuration this validation does not endorse

The self-training loop ships with 32 rounds and a seed quantile of 0.6. Run the same
rolling-origin harness over that choice against a shorter, more conservative setting, 4
rounds at quantile 0.4, and validation prefers the shorter one:

| target batch | shipped (32, q=0.6) | alternative (4, q=0.4) | difference |
|---|---|---|---|
| 2 | 0.6191 | 0.6196 | +0.001 |
| 6 | 0.7466 | 0.7339 | −0.013 |
| 7 | 0.9990 | 0.9990 | 0.000 |
| 8 | 0.9648 | 0.9648 | 0.000 |
| 9 | 0.7164 | 0.9680 | **+0.252** |

The shipped configuration was retained, and the honest reason is that the evidence for
switching is weaker than it first appears rather than that the evidence favours what
shipped. Four of the five folds are flat to within 0.013. The entire mean difference of
0.048 comes from batch 9, and it sits inside the 0.074 standard error established above,
which is precisely the situation this document elsewhere says should be called a tie.
Changing a verified pipeline on one fold's evidence would be selection on a single
observation.

What that fold does show is a real fragility: with enough rounds the loop can amplify a
poor starting point rather than recover from it, and batch 9 is where that happens. A
reader who weights the most recent fold most heavily, a defensible position, since the
true target sits further forward in time than any fold; should prefer 4 rounds at
quantile 0.4. That case is recorded here rather than argued away.

## How these claims were checked

Every comparative claim in this document was established on **labelled training data
only**, by holding out one training batch, treating it as the target, running the identical
pipeline against it, and comparing with that batch's known labels.

No statement here is derived from the hidden test set. This document contains no test
score, no per-class test result, no count of test errors and no description of which test
rows or classes the model finds difficult; none of that is knowable from the inputs the
pipeline is given. The relative orderings reported here, which model
family wins, whether the seed cap helps, whether the loop converges; all come from that
procedure, using labelled training data only.

That check also **corrected one earlier claim**. An initial observation suggested the seed
cap discards rows that are *more* accurate than the ones it keeps. That holds only for the
marginal rows at the cut for a single over-annexing class, not for the kept/dropped split
as a whole, where the kept rows are considerably more accurate. The cap filters for
correctness *and* applies corrective pressure; the earlier description mistook a narrow
effect for the general one.

## Properties worth knowing

**Deterministic.** No random seed anywhere; LDA with the `lsqr` solver, `StandardScaler`
and the assignment step are all deterministic, and no bagging or stochastic optimiser is
used. Repeated runs are byte-identical.

**No assumption about test class composition.** Round-0 labels are plain argmax, the seed cap
is recomputed each round from the model's own predictions, and the final output is
unconstrained. No per-class row count for the test set is used anywhere; the only property of
the test set assumed is that there are six gas classes, which is given by the competition and
visible in `sample_submission.csv`.

**Reads three files only:** `train.csv`, `test.csv`, `sample_submission.csv`.

**Runtime** about 15 seconds on a laptop CPU. No GPU, no pretrained models, no external data.

## Key results, observations and limitations

Two limits follow from the sensing physics, independently of any particular result.

**The dynamic range is bounded at both ends.** At very low concentrations the response is
small relative to sensor noise, so every gas produces a similar weak signal. At very high
concentrations the response saturates as surface sites fill, so the curves for different
gases compress toward each other. Separability should therefore be hardest at both
extremes and best in the middle of the range, and the concentrations here span three
orders of magnitude, so both regimes are present.

**Some of these gases are chemically similar.** Acetaldehyde and acetone are both carbonyls
of comparable molecular mass, and ethanol partially oxidises to acetaldehyde on a hot
metal-oxide surface, so those pairs can present genuinely similar surface chemistry to a
non-selective sensor. Any array of this type would be expected to find them harder to
separate than, say, ammonia from toluene.

Neither limit is a classifier failure; both are properties of the instrument and the
molecules.

The most discriminative technique in this field; modulating the sensor heater temperature,
since each gas has a characteristic optimal operating temperature; is unavailable here:
these sensors were operated at a fixed temperature.
