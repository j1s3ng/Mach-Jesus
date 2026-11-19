"""Portable path constants for archived CPU analysis; no GPU or remote host."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
CODE = Path(os.environ.get("RACE_CODE_ROOT", str(ROOT))).resolve()
TRAINING = Path("runs/improvement-12h/training")
DATASET = TRAINING / "detector-dense-dataset-001"
PILOT = TRAINING / "detector-full-dense1200-s7"
