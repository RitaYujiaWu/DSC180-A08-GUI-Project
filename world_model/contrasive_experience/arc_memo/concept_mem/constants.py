from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent

# Data directory
DATA_DIR = REPO_ROOT / "arc_memo/data"

# Hydra config path
HYRDA_CONFIG_PATH = REPO_ROOT / "arc_memo/configs"

# Environment file path
DOTENV_PATH = REPO_ROOT / ".env"

ABSTRACTION_INSTR_PATH = DATA_DIR / "abstract_anno" / "gui" / "concept_instr.txt"
