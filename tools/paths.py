from pathlib import Path
import os


VERSION = "040225"
DATA_ROOT = Path(os.environ["DATASTORE"])
REPO_ROOT = Path(__file__).resolve().parent.parent
PLOTS = DATA_ROOT / "plots"
