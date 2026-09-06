"""The individual feature transforms.

Sensors age over the 36 months the batches span: the same gas at the same concentration
reads differently later on. The transforms here are chosen for what survives that drift.

The useful physical fact is that ageing acts on a metal-oxide sensor roughly as a loss of
sensitivity; it scales the response rather than reshaping it. A multiplicative factor
becomes an additive constant in log space, so subtracting a reference computed from the
same reading cancels it. That is why the ratio and contrast transforms below all work by
subtracting a per-row mean in log space, and why they transfer across batches far better
than absolute response levels do.

None of these functions uses labels.
"""
import numpy as np
import pandas as pd

from config import BATCH_COL, FEATURES, N_DESCRIPTORS, N_SENSORS


def signed_log(x):
    """Compress five orders of magnitude without discarding sign.

    The raw features are both large and signed, so a plain log is unusable.
    """
    return np.sign(x) * np.log1p(np.abs(x))


def batch_standardize(df: pd.DataFrame, cols=FEATURES, robust: bool = True) -> pd.DataFrame:
    """Centre and scale each batch against its own statistics.

    This removes any shift shared by a whole batch, which is what drift largely is. It
    uses only feature values, never labels, so applying it to the test batch is legitimate.

    Its weakness is that a batch's own centre is contaminated by whichever gases happen to
    be in it, so used alone it removes class signal along with drift. It is one view among
    five rather than the whole representation.
    """
    X = signed_log(df[cols])
    out = X.copy()
    for b, idx in df.groupby(BATCH_COL).groups.items():
        blk = X.loc[idx]
        if robust:
            # Median and IQR rather than mean and standard deviation: these features
            # carry extreme values that would drag a mean around.
            centre = blk.median()
            scale = (blk.quantile(0.75) - blk.quantile(0.25)).replace(0, np.nan)
        else:
            centre, scale = (blk.mean(), blk.std(ddof=0).replace(0, np.nan))
        out.loc[idx] = ((blk - centre) / scale).to_numpy()
    # A zero spread leaves NaN above; those columns carry no information, so they go to 0.
    return out.fillna(0.0)


def global_standardize(df: pd.DataFrame, cols=FEATURES, stats=None):
    """Median/IQR standardisation with statistics fitted once and reused.

    `stats` is returned so the caller can fit on the training rows and apply the identical
    transform to the test rows, rather than refitting and shifting the two apart.
    """
    X = signed_log(df[cols])
    if stats is None:
        stats = (X.median(), (X.quantile(0.75) - X.quantile(0.25)).replace(0, np.nan))
    centre, scale = stats
    return (((X - centre) / scale).fillna(0.0), stats)


def sensor_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Each sensor's steady-state response as a log-ratio against the array mean.

    Descriptor 0 of each sensor is its steady-state value. Taking logs and subtracting the
    mean across the array removes any factor common to all 16 sensors, leaving the relative
    pattern, which is what identifies the gas, and which a uniform sensitivity loss does
    not change.
    """
    steady = df[[f'feat_{1 + N_DESCRIPTORS * s}' for s in range(N_SENSORS)]].to_numpy()
    steady = np.abs(steady) + 1.0
    log_steady = np.log(steady)
    ref = log_steady.mean(axis=1, keepdims=True)
    return pd.DataFrame(log_steady - ref,
                        columns=[f'ssratio_s{s + 1}' for s in range(N_SENSORS)],
                        index=df.index)


def slot_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """The same contrast applied within every descriptor slot, not just the steady state.

    For each of the 8 descriptors, the 16 sensors are compared against their own mean. This
    extends the cancellation above to the transient part of the response curve.
    """
    X = np.log(np.abs(df[FEATURES].to_numpy()) + 1.0)
    out = np.empty_like(X)
    for d in range(N_DESCRIPTORS):
        cols = [d + N_DESCRIPTORS * s for s in range(N_SENSORS)]
        blk = X[:, cols]
        out[:, cols] = blk - blk.mean(axis=1, keepdims=True)
    return pd.DataFrame(out,
                        columns=[f'slotr_{i + 1}' for i in range(len(FEATURES))],
                        index=df.index)


def l2_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Scale each reading to unit length: the response shape, independent of magnitude.

    Where the ratio transforms cancel a common factor per sensor group, this cancels it
    across the whole reading at once.
    """
    X = signed_log(df[FEATURES].to_numpy())
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return pd.DataFrame(X / np.where(n == 0, 1, n),
                        columns=[f'l2_{i + 1}' for i in range(len(FEATURES))],
                        index=df.index)
