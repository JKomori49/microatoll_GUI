from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_sea_level_csv(
    path: str | Path,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """
    Read a sea-level time series from CSV and return (xs, ys, meta).

    - Delimiter is auto-detected via csv.Sniffer, falls back to csv.excel.
    - Column detection prefers common aliases; otherwise selects the first two
      numeric-looking columns as (x, y).
    - Non-numeric or incomplete rows are skipped.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.

    Returns
    -------
    xs : List[float]
        X values (time/age/year).
    ys : List[float]
        Y values (sea-level/elevation/height).
    meta : Dict[str, Any]
        Metadata including header, chosen column indices, skipped row count, and path.
    """
    path = str(path)
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel

        reader = csv.reader(f, dialect)

        # Header (may be empty)
        header = next(reader, [])
        header_l = [h.strip().lower() for h in header]

        # Common aliases
        x_alias = {
            "t",
            "time",
            "times",
            "t_years",
            "year",
            "years",
            "age",
            "age_yr",
            "age_ka",
        }
        y_alias = {
            "sea_level",
            "sealevel",
            "rsl",
            "elevation",
            "height",
            "height_m",
            "sl",
            "rsl_m",
        }

        def _find_index(cands: set[str], hdr: List[str]) -> Optional[int]:
            # exact match on normalized names
            for i, name in enumerate(hdr):
                normalized = name.replace(" ", "").replace("-", "_")
                if normalized in cands:
                    return i
            # partial match (e.g., foo_height_m -> matches height_m)
            for i, name in enumerate(hdr):
                base = name.replace(" ", "").replace("-", "_")
                for c in cands:
                    if c in base:
                        return i
            return None

        xi = _find_index(x_alias, header_l) if header_l else None
        yi = _find_index(y_alias, header_l) if header_l else None

        rows = list(reader)

        # If header-based detection failed, try to infer numeric columns.
        if xi is None or yi is None:
            ncol = max((len(r) for r in rows), default=0)
            numeric_cols: List[int] = []
            for c in range(ncol):
                values = 0
                tests = 0
                for r in rows[: min(50, len(rows))]:
                    if c < len(r):
                        tests += 1
                        try:
                            float(str(r[c]).strip())
                            values += 1
                        except Exception:
                            pass
                # consider a column "numeric" if enough rows are convertible
                if tests and values >= max(5, int(0.5 * tests)):
                    numeric_cols.append(c)
            if len(numeric_cols) >= 2:
                xi, yi = numeric_cols[:2]
            else:
                raise ValueError(
                    "Could not identify two numeric columns. "
                    "Please ensure the CSV contains two numeric columns or proper headers."
                )

        xs: List[float] = []
        ys: List[float] = []
        skipped = 0

        for r in rows:
            if xi >= len(r) or yi >= len(r):
                skipped += 1
                continue
            try:
                x = float(str(r[xi]).strip())
                y = float(str(r[yi]).strip())
                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x)
                    ys.append(y)
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        meta = {
            "header": header,
            "x_index": xi,
            "y_index": yi,
            "skipped_rows": skipped,
            "path": path,
        }
        return xs, ys, meta
