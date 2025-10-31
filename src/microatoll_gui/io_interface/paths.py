from __future__ import annotations
from pathlib import Path
import sys

def resource_path(rel: str) -> str:
    """
    Return an absolute path to a resource, working for dev and frozen builds.
    Example: resource_path("resources/app_icon.ico")
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))  # src/ の親を想定
    return str((base / rel).resolve())
