import json
from models.polyline import Polyline

def save_polyline_json(poly: Polyline, path: str) -> None:
    """
    Save polyline as JSON:
    {
      "type": "polyline",
      "version": "0.1",
      "vertices": [{"x":..., "y":..., "phi":...}, ...]
    }
    """
    payload = {
        "type": "polyline",
        "version": "0.1",
        "vertices": poly.to_rows()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
