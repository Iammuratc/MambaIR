# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.63"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from yolov12.ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from yolov12.ultralytics.utils import ASSETS, SETTINGS
from yolov12.ultralytics.utils.checks import check_yolo as checks
from yolov12.ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)
