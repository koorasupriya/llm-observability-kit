"""Token pricing used to estimate the cost of each LLM call.

Prices are USD per 1 million tokens. The defaults are ILLUSTRATIVE demo values. Put your
provider's current prices in a JSON file and point LLM_PRICING_FILE at it, e.g.

    {"my-model": {"input": 3.0, "output": 15.0}}
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "demo-small": {"input": 0.25, "output": 1.25},
    "demo-large": {"input": 3.00, "output": 15.00},
}


def load_pricing() -> dict[str, dict[str, float]]:
    pricing = dict(DEFAULT_PRICING)
    path = os.getenv("LLM_PRICING_FILE")
    if path and Path(path).exists():
        pricing.update(json.loads(Path(path).read_text()))
    return pricing


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  pricing: dict[str, dict[str, float]] | None = None) -> float | None:
    """Return the estimated USD cost, or None when the model has no price configured."""
    table = pricing if pricing is not None else load_pricing()
    price = table.get(model)
    if price is None:
        return None
    return round((input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000, 8)
