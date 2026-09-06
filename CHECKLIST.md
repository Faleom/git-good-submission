# Required materials: where each is

| # | Required | Where it is |
|---|---|---|
| 1 | Final notebook / methodology report | `notebook/datathon_notebook.ipynb`, with `MODEL.md` as the component-by-component reference |
| | - complete description of the approach | "End to end, start to finish", "What each component does" |
| | - data cleaning and preprocessing | "Data cleaning and preprocessing" |
| | - feature engineering | "Five feature views", "Concentration, encoded two ways" |
| | - validation strategy | "Models tested and how the final one was selected", "How these claims were checked" |
| | - models tested and final selection | "Models tested, and how the final one was selected" |
| | - ensembling and post-processing | "Ensembling and post-processing" |
| | - key results, observations, limitations | "Key results, observations and limitations" |
| 2 | Complete source code | `src/`, six files, all listed below |
| | - data loading | `src/data.py`, `src/config.py` |
| | - preprocessing | `src/features.py` |
| | - feature generation | `src/features.py`, `src/feature_views.py` |
| | - model training | `src/final_model.py` |
| | - validation | `src/validation_scores.py` (rolling-origin on `train.csv`); harness and comparisons in the notebook, Sections 3, 6, 7d, 9b |
| | - test inference | `src/final_model.py` |
| | - post-processing | `src/final_model.py` (in-loop seed selection, capped at the 60th-percentile value among the model's own six current predicted class counts, plus relabelling; none on the output) |
| | - submission generation | `src/final_model.py` |
| 3 | Final prediction file | `outputs/submissions/submission_final.csv` |
| 4 | Processed datasets | none exist; see "Processed / cleaned datasets" in `README.md` |
| 5 | README / reproduction instructions | `README.md` |
| | - required files, execution order | "Reproduce", "Contents" |
| | - dependencies | "Dependencies" |
| | - random seeds or settings | "Final model information": none required, pipeline is deterministic |
| | - which script generates the prediction | "Which script generates the final prediction file" |
| 6 | Final model information | `README.md` "Final model information", `MODEL.md` throughout |
| | - model / ensemble identified | equal-weight ensemble of five shrinkage-LDA models |
| | - hyperparameters and ensemble weights | table in `README.md` |
| | - pretrained / AutoML / external packages | none used; stated in `README.md` |

## On scoring

This bundle contains **no labels for the hidden test set, and nothing that scores against
them**. It reads `train.csv`, `test.csv` and `sample_submission.csv` only.

Validation *is* shipped and runnable, but it scores only the labelled training batches
against their own labels: `src/validation_scores.py` holds out one training batch at a
time, predicts it from the earlier ones, and compares with that batch's known labels. The
notebook does the same throughout. No score in this bundle is a test-set score, and none
could be; the labels required to compute one are not present.

No per-class row count for the test set is used anywhere either. The seed cap inside the
self-training loop is the 60th-percentile value among the model's own six current predicted
class counts, recomputed every round from its own predictions; see "The seed cap" in
`MODEL.md`.

## On item 3

`submission_final.csv` is produced directly by `python3 src/final_model.py`
and is not a recreated or modified file. The pipeline is deterministic, so re-running the
command reproduces it byte-for-byte.
