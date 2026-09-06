"""Assembles the five feature views the final model averages over.

Each view is a different combination of the transforms in `features.py`. They are not
independent, most share blocks, but they fail differently, and averaging five linear
models fitted on different representations is more stable than trusting any one of them.

Every view is built for the training rows and the target rows together and returned as a
pair, so the two always receive the identical transform.

The names encode the blocks each view contains:
    bstd   per-batch standardisation      slotr  within-slot log-contrasts
    gz     global (train-fitted) z-score   l2     per-reading L2 normalisation
    ss     sensor steady-state log-ratios  sg     plain signed-log features
"""
import numpy as np
import pandas as pd

from config import TARGET
from features import (batch_standardize, global_standardize, l2_normalise,
                      sensor_ratio_features, signed_log, slot_ratio_features)


def views(tr, te):
    """Build all five views. Returns {name: (train_matrix, target_matrix)}."""
    n = len(tr)

    # Per-batch standardisation needs every batch in one frame so each is centred on its
    # own statistics. The target's labels are absent here, and none are used: the
    # transform reads feature values only.
    comb = pd.concat([tr.drop(columns=[TARGET]), te], ignore_index=True)
    bs = batch_standardize(comb).to_numpy()

    # Fitted on the training rows, then applied unchanged to the target rows.
    g, st = global_standardize(tr)
    gt, _ = global_standardize(te, stats=st)

    # Row-wise transforms: each reading is transformed using only itself, so there is
    # nothing to fit and no way for one set of rows to influence the other.
    ss = (sensor_ratio_features(tr).to_numpy(), sensor_ratio_features(te).to_numpy())
    sl = (slot_ratio_features(tr).to_numpy(), slot_ratio_features(te).to_numpy())
    l2 = (l2_normalise(tr).to_numpy(), l2_normalise(te).to_numpy())

    feat_cols = lambda d: [c for c in d.columns if c.startswith('feat_')]
    sg = (signed_log(tr[feat_cols(tr)]).to_numpy(),
          signed_log(te[feat_cols(te)]).to_numpy())

    return {
        'A_bstd+ss+slotr+l2': (
            np.c_[bs[:n], ss[0], sl[0], l2[0]],
            np.c_[bs[n:], ss[1], sl[1], l2[1]]),
        'B_bstd+gz+ss+slotr+l2': (
            np.c_[bs[:n], g.to_numpy(), ss[0], sl[0], l2[0]],
            np.c_[bs[n:], gt.to_numpy(), ss[1], sl[1], l2[1]]),
        'C_gz+ss+slotr+l2': (
            np.c_[g.to_numpy(), ss[0], sl[0], l2[0]],
            np.c_[gt.to_numpy(), ss[1], sl[1], l2[1]]),
        'D_slotr+l2+ss': (
            np.c_[sl[0], l2[0], ss[0]],
            np.c_[sl[1], l2[1], ss[1]]),
        'E_all': (
            np.c_[bs[:n], g.to_numpy(), sg[0], ss[0], sl[0], l2[0]],
            np.c_[bs[n:], gt.to_numpy(), sg[1], ss[1], sl[1], l2[1]]),
    }
