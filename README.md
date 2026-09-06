# Track 3: Gas Sensor Array Drift

Required-materials index: **[CHECKLIST.md](CHECKLIST.md)** · What to run: **[HOWTO.md](HOWTO.md)**

## The two required deliverables

**1. Final notebook / methodology report**: `notebook/datathon_notebook.ipynb`.
The reasoning behind the model, with every number computed in a cell rather than quoted.
It rebuilds the final pipeline independently from `data/raw/` and reports its agreement
with the submitted file, so the documentation can be checked against the submission rather
than taken on trust. 81 cells, no errors, about 15 minutes.

**2. Complete source code**: `src/`, entry point `src/final_model.py`.
Data loading, preprocessing, feature generation, model training, validation, test
inference, post-processing and submission writing. The approaches that were considered
and not adopted are evaluated in the notebook, Sections 7d and 9b.

The notebook and `src/` contain the same model by design. The notebook does not import
`src/`; it reimplements the pipeline from its own helpers and then compares. A match is
therefore evidence that the documented reasoning describes the code that produced the
submission, which it could not be if the notebook simply called into it.

## Declarations

### AI tools and coding agents used

This project was developed with **Claude (Anthropic), used through Claude Code** as an
interactive coding agent, working under the team's direction throughout.

The team set the objectives, chose which directions to pursue, decided what to adopt and
what to withdraw, and reviewed the output. No result in this bundle is reported without a
computation behind it: every number in the notebook is produced by a cell in that notebook,
and the model itself is reproducible from `train.csv` and `test.csv` by two independent
implementations, which is why both are included.

No other external dataset, external code, additional information, AI service, AutoML system or pretrained model was used. The pipeline is built
entirely from scikit-learn primitives listed in `requirements.txt`.

### Manual modification and post-processing of predictions

**None was applied to the submitted predictions.** The contents of
`outputs/submissions/submission_final.csv` are the plain `argmax` of the model's averaged
class posterior. No row was edited by hand at any stage, no prediction was overridden, and
no rule was applied to the output after the model produced it.

In particular, no class-balance constraint, prior correction or assignment step is applied.
Nothing in the pipeline is told how many rows of each gas the test set contains, and no such
count is assumed anywhere. The predicted class counts printed when the model runs are a
result of the model, not a target imposed on it.

Two operations that do count as post-processing occur **inside the fitting loop**, not on
the output, and both are deterministic and part of the model rather than a manual step:

1. **Seed selection.** Each self-training round refits on the most confident rows per
   predicted class, capped at the 60th-percentile value among the model's own six current
   predicted counts, recomputed every round from its own output.
2. **Relabelling.** The whole test set is relabelled from the averaged posterior each round.

Both are described in `MODEL.md` and implemented in `src/final_model.py`.

The pipeline sets no random seed and needs none, so re-running it reproduces the submitted
file byte for byte rather than approximately.

## Reproduce

```bash
pip install -r requirements.txt
python3 src/final_model.py
```

Writes `outputs/submissions/submission_final.csv` (3,600 rows,
`measurement_id,gas_class`).

Runtime about 15 seconds on a laptop CPU. **No random seed required; the pipeline is fully
deterministic** and repeated runs are byte-identical.

## Which script generates the final prediction file

`src/final_model.py` is the only entry point. It performs data loading,
preprocessing, feature generation, model training, test inference, post-processing and
submission writing in a single run.

It is the only script needed to produce the submission. The approaches that were
considered and rejected are not rebuilt here; they are scored against this one in the
notebook, Sections 7d and 9b, on the same validation folds used throughout.

## Processed / cleaned datasets

**None are shipped, and none are required.** No intermediate dataset is written to disk.
The raw data needs no cleaning; it contains no missing values, duplicate rows, constant
columns or infinities, and every derived feature is computed in memory at run time by
`src/features.py` and `src/feature_views.py`. Running the command above derives everything
from `data/raw/` alone.

## Contents

```
src/final_model.py                        entry point, produces the submission
src/feature_views.py                      assembles the five feature views
src/features.py                           the individual feature transforms
src/config.py                             paths, column names, gas-name map
src/data.py                               CSV loaders
src/validation_scores.py                  rolling-origin validation on train.csv

notebook/datathon_notebook.ipynb          the methodology notebook (deliverable 1)
notebook/README.md                        what the notebook contains, section by section

data/raw/                                 train.csv, test.csv, sample_submission.csv
outputs/submissions/submission_final.csv  the submission
requirements.txt

README.md      this file
HOWTO.md       what to run, what it produces, measured timings
MODEL.md       how the model works, component by component
CHECKLIST.md   where each required submission item lives
```

## Local validation scores

Rolling-origin macro-F1 on the labelled training batches; **mean 0.8092**, standard
deviation 0.165 across five folds, standard error 0.074. Per-fold figures and what the
spread implies are in [MODEL.md](MODEL.md) under "Local validation scores".

The fold-to-fold spread is large because each fold targets a different batch with a
different amount of drift, and because early folds train on far less data than the deployed
model does.

## Validation strategy

Model families and techniques were compared by **batch-forward cross-validation on the
labelled training data**: train on earlier batches, predict a later one, and compare
against that batch's known labels. A random split is not used; near-duplicate readings
from one batch land on both sides of it, so the model is graded on data it has effectively
already seen.

The same procedure verifies the pipeline's behaviour: hold out one training batch, treat
it as the target, run the identical loop, and compare with that batch's labels. This is
described in `MODEL.md` under "Models tested, and how the final one was selected" and "How
these claims were checked". No scoring code is shipped in this bundle.

## Dependencies

```
scikit-learn 1.8.0   numpy 2.4.2   pandas 3.0.1   scipy 1.17.1
```

Python 3.13. No GPU. No pretrained models, no AutoML, no external data.

## Model

Full explanation of every component and why it is there: **[MODEL.md](MODEL.md)**

Shrinkage LDA (`solver="lsqr", shrinkage="auto"`) fitted over five feature views of the
same data; posteriors averaged with equal weight. The test batch is then relabelled and
the models refitted on its own rows for 32 rounds. Each round's seeds are the most
confident rows of each predicted class, capped at the 60th-percentile value among the
model's own six current predicted class counts, recomputed each round from its own output.
No per-class row count for the test set is used anywhere.

Features: per-batch and global robust z-scores of signed-log responses, sensor
steady-state log-ratios, within-slot log-contrasts, per-sample L2 normalisation, one-hot
concentration, and residuals after regressing out the log-linear concentration response.

No assumption is made about the test set's class composition, and no per-class row count
enters the pipeline. The pipeline reads `train.csv`, `test.csv` and `sample_submission.csv`
only.

## Final model information

**Model:** an equal-weight ensemble of five shrinkage-LDA classifiers, one per feature
view, refined by 32 rounds of self-training on the test batch.

| parameter | value |
|---|---|
| classifier | `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` |
| shrinkage | Ledoit-Wolf, automatic |
| feature views | 5 |
| **ensemble weights** | **equal, 1/5 per view** (no weight tuning) |
| self-training rounds | 32 (converged; identical output at 64) |
| seed cap | the 60th-percentile value among the model's own six current predicted class counts, recomputed each round |
| final decision | unconstrained argmax |
| random seed | **none; the pipeline is deterministic** |

**No pretrained model, no AutoML system, and no external package beyond scikit-learn,
numpy, pandas and scipy.** No external data. No GPU.

Full explanation of every component: [MODEL.md](MODEL.md).
