# What to run, what it does, what it produces

Everything below was run from a clean checkout of this folder, with outputs deleted first,
and the timings are measured, not estimated.

```bash
pip install -r requirements.txt
```

---

## 1. Produce the submission

```bash
python3 src/final_model.py
```

**Does:** loads the three CSVs, builds five feature views plus the two concentration
blocks, fits shrinkage LDA on the training data, then refits on the test rows using its
own predicted labels for 32 rounds. Each round trains on the most confident rows per
predicted class, capped at the 60th-percentile value among the model's own six current
predicted class counts, recomputed each round from its own output, never a fixed number.

**Produces:** `outputs/submissions/submission_final.csv`, 3,600 rows, columns
`measurement_id,gas_class`. This is the file to submit.

**Takes:** about 15 seconds.

**Prints:** the predicted count for each gas, so you can see the loop's output at a glance.
These counts are an output, not a target: no per-class row count is supplied to the pipeline.

This is the only command needed. It is deterministic; run it twice and the file is
byte-identical.

---

## 2. Validate the pipeline

```bash
python3 src/validation_scores.py
```

**Does:** rolling-origin validation. Trains on batches 1..k and predicts batch k+1, for
every k where both sides contain all six gases. Reads `train.csv` only; the hidden test
set is never involved and its labels are not present in this bundle.

**Produces:** `outputs/reports/validation_scores.csv`, and prints the per-fold table.

**Takes:** about 45 seconds.

---

## 3. Read the methodology notebook

```bash
jupyter notebook notebook/datathon_notebook.ipynb    # then Run All
```

**Does:** runs `notebook/datathon_notebook.ipynb` top to bottom. The notebook rebuilds the
final pipeline from `data/raw/` using its own implementation, scores thirteen rejected
techniques and each of the five feature views on the validation harness, and finishes by
comparing its own predictions with the submitted file.

**Produces:** the executed notebook, plus `notebook/outputs/submission_from_notebook.csv`
its own copy of the predictions, kept separate from the submitted file.

**Takes:** about 15 minutes from cold, 11 with `notebook/_cache/` populated.

The notebook ships already executed, so its outputs can be read without running anything.

---

## The files

| file | role |
|---|---|
| `src/final_model.py` | **entry point**: the whole pipeline |
| `src/feature_views.py` | assembles the five feature views |
| `src/features.py` | the individual transforms: signed-log, robust z, per-batch z, sensor ratios, slot contrasts, L2 |
| `src/data.py` | reads the three CSVs |
| `src/config.py` | paths, column names, gas-name map |
| `src/validation_scores.py` | rolling-origin validation on `train.csv` |
| `notebook/datathon_notebook.ipynb` | the methodology notebook |

| document | contents |
|---|---|
| `README.md` | how to reproduce, dependencies, final model information |
| `MODEL.md` | how the model works, component by component, and what was not adopted |
| `CHECKLIST.md` | where each required submission item lives |
| `HOWTO.md` | this file |

---

## What the folder produces

| output | from | count |
|---|---|---|
| `outputs/submissions/submission_final.csv` | `src/final_model.py` | 1, **the submission** |

Every prediction file has the same two columns as `sample_submission.csv`.

The bundle does contain scoring code, but it can only score the labelled training batches
against their own labels; `src/validation_scores.py` and the notebook's harness both work
by holding out a training batch. No labels for the hidden test set are present, so no
test-set score can be computed here.

## Verified

From an empty `outputs/`:

- `src/final_model.py` → writes the submission in about 15 seconds
- `src/validation_scores.py` → the per-fold table in about 45 seconds

Re-running the model produces a byte-identical file, since no random seed is involved
anywhere in the pipeline.
