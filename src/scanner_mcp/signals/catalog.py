"""Predefined signal types and parameter schemas (for LLM + validation)."""

from __future__ import annotations

from typing import Any

# signal_type -> {description, default_params, required_param_keys (optional)}
CATALOG: dict[str, dict[str, Any]] = {
    "golden_cross": {
        "description": "50-day SMA crosses above 200-day SMA (bullish)",
        "default_params": {"fast": 50, "slow": 200},
        "required_params": [],
    },
    "death_cross": {
        "description": "50-day SMA crosses below 200-day SMA (bearish)",
        "default_params": {"fast": 50, "slow": 200},
        "required_params": [],
    },
    "macd_bullish_crossover": {
        "description": "MACD line crosses above signal line",
        "default_params": {"fast": 12, "slow": 26, "signal": 9},
        "required_params": [],
    },
    "macd_bearish_crossover": {
        "description": "MACD line crosses below signal line",
        "default_params": {"fast": 12, "slow": 26, "signal": 9},
        "required_params": [],
    },
    "rsi_oversold": {
        "description": "RSI is below a threshold (default 30)",
        "default_params": {"period": 14, "threshold": 30},
        "required_params": [],
    },
    "rsi_overbought": {
        "description": "RSI is above a threshold (default 70)",
        "default_params": {"period": 14, "threshold": 70},
        "required_params": [],
    },
    "pct_from_ma": {
        "description": "Price is within a percent band of a moving average",
        "default_params": {"ma_period": 50, "ma_type": "sma", "pct": 2.0},
        "required_params": [],
    },
    "pct_from_ath": {
        "description": "Price is at least N% below the all-time high (rolling window = full series)",
        "default_params": {"min_pct_below_ath": 20.0},
        "required_params": [],
    },
    "bbands_breakout": {
        "description": "Close is outside the Bollinger band (20, 2)",
        "default_params": {"length": 20, "std": 2.0, "side": "either"},
        "required_params": [],
    },
    "bull_flag": {
        "description": "Simplified: prior 10d move > 10% then 3d consolidation (range < 3% of price)",
        "default_params": {"prior_lookback": 10, "prior_move_pct": 10.0, "consol_days": 3, "max_range_pct": 3.0},
        "required_params": [],
    },
}


def list_catalog_entries() -> list[dict[str, Any]]:
    """Return public signal catalog metadata for clients and prompts."""
    return [
        {
            "signal_type": k,
            "description": v["description"],
            "default_params": v["default_params"],
        }
        for k, v in CATALOG.items()
    ]


def merge_params(signal_type: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user-provided params over catalog defaults for a signal type."""
    if signal_type not in CATALOG:
        raise ValueError(f"Unknown signal_type: {signal_type}")
    base = dict(CATALOG[signal_type]["default_params"])
    if params:
        base.update(params)
    return base
