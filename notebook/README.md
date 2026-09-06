# Methodology notebook

`datathon_notebook.ipynb`; the reasoning behind the model, with every number computed in
the notebook rather than quoted.

```bash
pip install -r ../requirements.txt
jupyter notebook datathon_notebook.ipynb     # or: jupyter nbconvert --execute
```

Runs top to bottom in about 15 minutes from cold. Reads `../data/raw/`. Writes its own
prediction file to `outputs/`, kept separate from the submitted one. Sections 7d and 9b
cache to `_cache/`, so re-runs are faster.

## What it contains

Each modelling section follows the same shape; **question → experiment → number →
decision, and what it ruled out**. Sections that rule nothing out are not included.

| section | question |
|---|---|
| 1-2 | Does anything need cleaning, and is drift real? |
| 3 | What validation protocol is honest for a later batch? |
| 4 | Which feature transforms survive drift? |
| 5 | Can the concentration column help, and in what form? |
| 6 | Which model family transfers? |
| 7 | Can we adapt to the target domain without its labels? |
| 7d | **Every alternative technique tried, and the number that ruled it out** |
| 8 | Are the remaining errors fixable by a different model? |
| 9 | The final model, assembled and reproduced from the raw data |
| 9b | **Does the five-view ensemble earn its place?** |
| 10 | Which differences are real and which are noise? |
| 11-12 | Limitations, and a summary of every decision |

Sections 7d and 9b carry the alternatives: thirteen rejected techniques and the five
feature views, each scored on the same rolling-origin folds as everything else, so no
rejection in this notebook rests on an assertion.

## Verified on the last run

- **Executes with no errors**, 81 cells, 874 seconds from cold.
- **Reproduction: 100%**: Section 9 rebuilds the pipeline from `train.csv` and `test.csv`
  and matches the shipped submission on all 3,600 rows. It does not load the stored file
  except to compare.
- Naive cross-validation overstates performance by **0.18 to 0.27** depending on the model,
  and reorders them: it ranks random forest top, honest validation ranks it last.
- Within a single batch the gases separate at **0.985** macro-F1, confirming the class
  information is intact and only the mapping has moved.
- QDA, the per-class version of the chosen model, **cannot be estimated** at this
  feature-to-sample ratio.
- **23.7%** of errors are shared by every model family tested.
- **All 13 alternative techniques were rejected**, each scored on the same folds. CORAL is
  the worst at **-0.485** and cannot be fitted at all on one fold. Removing the dose
  residuals costs **0.069**, the largest contribution of any single component.
- **The five-view ensemble trails its own best single view by 0.095** on validation. This
  is reported in Section 9b rather than omitted, and flagged as a configuration our own
  validation does not support.

## Scope of the numbers

Every score in the notebook is **validation macro-F1** computed on the labelled training
batches by the rolling-origin harness. Performance on the hidden test set is a different
quantity and is not computed here, since those labels are not available to the pipeline.
The two are never compared directly: the hidden batch sits further forward in time than
any validation fold, so a gap between them is expected.

## Relationship to the submission

This notebook is documentation. The submitted model lives in the `src/` directory alongside it and is
unaffected by anything here.
