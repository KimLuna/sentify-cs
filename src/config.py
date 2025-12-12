from pathlib import Path

# Base directory of the project (one level up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Location of the dataset file.  Update this if your data has
# a different name or is stored elsewhere.  The default points to
# the provided Excel file `Chat_Team_CaseStudy FINAL.xlsx`.  The
# loader in `data_loader.py` will handle both CSV and Excel
# formats transparently.
DATA_FILE = BASE_DIR / "data" / "custom_service_data.xlsx"

# Directory where trained models and reports will be saved
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR = REPORTS_DIR / "models"

# Test split ratio for train/test splitting
TEST_SIZE: float = 0.2

# Random seed to ensure reproducibility
RANDOM_STATE: int = 42

# Rating thresholds to define sentiment labels.  For this
# dataset ratings are in the range 0–10.  Scores >= POSITIVE_THRESHOLD
# (8, 9, 10) are mapped to 'positive'; scores <= NEGATIVE_THRESHOLD
# (0, 1, 2) are mapped to 'negative'.  Values in between are
# considered neutral and excluded from training.
POSITIVE_THRESHOLD: int = 8
NEGATIVE_THRESHOLD: int = 2

# Number of features for the TF-IDF vectoriser.  Increasing this
# may improve accuracy at the expense of memory.
MAX_FEATURES: int = 5000

def ensure_directories() -> None:
    """Create required directories if they do not exist.

    This function is called at runtime to guarantee that the
    directories used for reports and saved models are present.
    """
    for d in [REPORTS_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# Optional: Adding a function to check if the dataset file exists and is valid
def check_dataset_file() -> None:
    """Check if the dataset file exists and is accessible."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}. Please place the data file in the 'data' directory.")
