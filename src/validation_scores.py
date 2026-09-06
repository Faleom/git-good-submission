"""Rolling-origin validation of the shipped pipeline, using labelled training batches only.

Train on batches 1..k, predict batch k+1, compare with that batch's known labels. Folds
whose target batch lacks any of the six gases are skipped, since macro-F1 over five
classes is not comparable to macro-F1 over six.

Reads train.csv only. The hidden test set is never involved.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import final_model as fm
from config import BATCH_COL, GAS_NAMES, TARGET
from data import load_train
from feature_views import views


def run_fold(src, tgt):
    """The shipped pipeline exactly: five views, dose residuals, 32 self-training rounds."""
    R = views(src, tgt)
    oh = fm._conc_onehot(src, tgt)
    dz = fm._dose_residuals(src, tgt)
    extra = (np.c_[oh[0], dz[0]], np.c_[oh[1], dz[1]])
    ys = src[TARGET].to_numpy()
    per = len(tgt) // len(GAS_NAMES)

    proba, classes = fm._ens(R, False, np.arange(len(src)), ys, extra)
    labels = classes[proba.argmax(1)]
    round0 = labels.copy()
    for _ in range(fm.ROUNDS):
        conf = proba.max(1)
        counts = np.array([(labels == c).sum() for c in classes])
        cap = max(1, int(round(float(np.quantile(counts, fm.Q)))))
        seeds = np.concatenate([
            np.where(labels == c)[0][np.argsort(-conf[np.where(labels == c)[0]])][:cap]
            for c in classes])
        proba, classes = fm._ens(R, True, seeds, labels[seeds], extra)
        labels = classes[proba.argmax(1)]
    return round0, labels


def main() -> None:
    tr = load_train()
    batches = sorted(tr[BATCH_COL].unique())
    rows = []
    for k in range(1, len(batches)):
        tgt_b = batches[k]
        src = tr[tr[BATCH_COL].isin(batches[:k])].reset_index(drop=True)
        tgt = tr[tr[BATCH_COL] == tgt_b].reset_index(drop=True)
        if tgt[TARGET].nunique() < len(GAS_NAMES) or src[TARGET].nunique() < len(GAS_NAMES):
            continue
        r0, fin = run_fold(src, tgt)
        yt = tgt[TARGET].to_numpy()
        per = f1_score(yt, fin, average=None, labels=sorted(GAS_NAMES), zero_division=0)
        rows.append({
            "target batch": tgt_b, "train rows": len(src), "target rows": len(tgt),
            "round 0 only": f1_score(yt, r0, average="macro", zero_division=0),
            "full pipeline": f1_score(yt, fin, average="macro", zero_division=0),
            "worst class": per.min(),
        })
        print(f"  fold -> batch {tgt_b}: round0 {rows[-1]['round 0 only']:.4f}  "
              f"full {rows[-1]['full pipeline']:.4f}", flush=True)

    df = pd.DataFrame(rows).set_index("target batch")
    out = Path(__file__).resolve().parents[1] / "outputs" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "validation_scores.csv")
    print("\n" + df.round(4).to_string())
    print(f"\nmean full pipeline : {df['full pipeline'].mean():.4f}")
    print(f"sd across folds    : {df['full pipeline'].std():.4f}")
    print(f"standard error     : {df['full pipeline'].std()/np.sqrt(len(df)):.4f}")
    print(f"gain from the loop : {(df['full pipeline'] - df['round 0 only']).mean():+.4f}")


if __name__ == "__main__":
    main()
