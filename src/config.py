"""Paths, column names and dataset constants.

Importing this module creates the submissions directory, so `final_model.py` can write
its output without a separate setup step.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / 'data' / 'raw'
OUTPUTS = ROOT / 'outputs'
SUBMISSIONS = OUTPUTS / 'submissions'

TRAIN_CSV = DATA_RAW / 'train.csv'
TEST_CSV = DATA_RAW / 'test.csv'
SAMPLE_SUB_CSV = DATA_RAW / 'sample_submission.csv'

ID_COL = 'measurement_id'
TARGET = 'gas_class'
BATCH_COL = 'batch'
CONC_COL = 'concentration'

# The 128 feature columns are a 16 x 8 grid: each of 16 metal-oxide sensors contributes
# 8 descriptors of its response curve. Several transforms in features.py rely on that
# layout, so the two counts are kept here rather than written as literals.
FEATURES = [f'feat_{i}' for i in range(1, 129)]
N_SENSORS, N_DESCRIPTORS = (16, 8)

GAS_NAMES = {1: 'Ethanol', 2: 'Ethylene', 3: 'Ammonia',
             4: 'Acetaldehyde', 5: 'Acetone', 6: 'Toluene'}

SUBMISSIONS.mkdir(parents=True, exist_ok=True)
