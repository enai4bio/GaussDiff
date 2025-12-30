

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from nhanes_pipeline.cli import main

if __name__ == "__main__":
    main()
