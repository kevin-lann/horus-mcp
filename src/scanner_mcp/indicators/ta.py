"""Small technical-analysis helpers used by scanner-mcp.

This keeps the package installable without relying on pandas-ta availability.
"""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, length: int) -> pd.Series:
    return close.astype(float).rolling(window=length, min_periods=length).mean()


def ema(close: pd.Series, length: int) -> pd.Series:
    return close.astype(float).ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.where(avg_loss != 0, 100.0).where(avg_gain != 0, 0.0)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            f"MACD_{fast}_{slow}_{signal}": macd_line,
            f"MACDs_{fast}_{slow}_{signal}": signal_line,
            f"MACDh_{fast}_{slow}_{signal}": hist,
        }
    )


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    close = close.astype(float)
    mid = sma(close, length)
    dev = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + std * dev
    lower = mid - std * dev
    return pd.DataFrame(
        {
            f"BBL_{length}_{std}": lower,
            f"BBM_{length}_{std}": mid,
            f"BBU_{length}_{std}": upper,
        }
    )
