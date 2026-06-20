from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / 'eval'

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
