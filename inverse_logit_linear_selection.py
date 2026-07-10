import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

runpy.run_path(str(SRC / "inverse_logit_linear_selection.py"), run_name="__main__")
