import logging
import pathlib
from typing import Any, Dict

import yaml

# Load configuration
config_path: pathlib.Path = pathlib.Path(__file__).parent / "config.yml"

with open(config_path, "r", encoding="utf-8") as f_config:
    config: Dict[str, Any] = yaml.safe_load(f_config)


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s-%(filename)s->%(funcName)s:-%(levelname)s] "
        "%(message)s"
    ),
)
logger = logging.getLogger(__name__)
