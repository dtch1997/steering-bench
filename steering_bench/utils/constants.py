from pathlib import Path
# from typing import Literal

# PKG_AUTHOR = "Daniel Tan"
# PKG_AUTHOR_DIR = "dtch1997"
PKG_NAME = Path(__file__).parent.parent.stem
PKG_PATH = Path(__file__).parent.parent

PROJ_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJ_DIR / "assets"
DATASET_DIR = ASSETS_DIR / "datasets"

# DEFAULT_EPOCHS = 1
# DEFAULT_MAX_RETRIES = 5
# DEFAULT_TIMEOUT = 120
# DEFAULT_MAX_CONNECTIONS = 10
# DEFAULT_MAX_TOKENS = 2048
# DEFAULT_VIEW_PORT = 7575
# DEFAULT_SERVER_HOST = "127.0.0.1"
# HTTP = 15
# HTTP_LOG_LEVEL = "HTTP"
# SANDBOX = 17
# SANDBOX_LOG_LEVEL = "SANDBOX"
# ALL_LOG_LEVELS = [
#     "DEBUG",
#     HTTP_LOG_LEVEL,
#     SANDBOX_LOG_LEVEL,
#     "INFO",
#     "WARNING",
#     "ERROR",
#     "CRITICAL",
# ]
# DEFAULT_LOG_LEVEL = "warning"
# DEFAULT_LOG_LEVEL_TRANSCRIPT = "info"
# ALL_LOG_FORMATS = ["eval", "json"]
# DEFAULT_LOG_FORMAT: Literal["eval", "json"] = "eval"
# EVAL_LOG_FORMAT = "eval"
# DEFAULT_DISPLAY = "full"
# LOG_SCHEMA_VERSION = 2
# SCORED_SUFFIX = "-scored"
# SAMPLE_SUBTASK = "sample"
# CONSOLE_DISPLAY_WIDTH = 120
# BASE_64_DATA_REMOVED = "<base64-data-removed>"
