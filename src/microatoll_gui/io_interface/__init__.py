"""
io_interface package: unified I/O entrypoints.

Expose common readers/writers at the package top-level to keep GUI and
other clients decoupled from file-format-specific modules.
"""

from .csvio import read_sea_level_csv

__all__ = ["read_sea_level_csv"]
