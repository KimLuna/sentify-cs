from pathlib import Path

# Set up base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "custom_service_data.xlsx"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR = REPORTS_DIR / "models"

# Data splitting and reproducibility
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# Rating thresholds for sentiment labeling (0-10 scale)
# Scores >= POSITIVE_THRESHOLD are 'positive' (e.g., 8-10)
# Scores <= NEGATIVE_THRESHOLD are 'negative' (e.g., 0-2)
POSITIVE_THRESHOLD: int = 8
NEGATIVE_THRESHOLD: int = 2

# Feature engineering parameter
MAX_FEATURES: int = 5000

def ensure_directories() -> None:
    """Create required directories for reports and models if they do not exist."""
    for d in [REPORTS_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def check_dataset_file() -> None:
    """Check if the dataset file exists and is accessible."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}. Please place the data file in the 'data' directory.")