"""Market price download using Yahoo Finance."""

from datetime import date

import pandas as pd
import yfinance as yf


def download_prices(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Download an asset's adjusted closing prices.

    Parameters
    ----------
    ticker:
        Yahoo Finance symbol for the asset.
    start_date:
        Start date in YYYY-MM-DD format.
    end_date:
        End date in YYYY-MM-DD format.
    """
    ticker = ticker.strip().upper()

    history = yf.Ticker(ticker).history(
        start=start_date,
        end=end_date,
        auto_adjust=True,
    )

    if history.empty:
        raise ValueError(
            f"No data found for ticker {ticker}."
        )

    if "Close" not in history.columns:
        raise ValueError(
            f"The Close column is missing for {ticker}."
        )

    prices = history["Close"].copy()
    prices.name = ticker

    # Remove the index timezone if one is present.
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    return prices


def download_fund_and_benchmark(
    fund_ticker: str,
    benchmark_ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download and align the fund and benchmark prices."""
    fund = download_prices(
        fund_ticker,
        start_date,
        end_date,
    )
    benchmark = download_prices(
        benchmark_ticker,
        start_date,
        end_date,
    )

    prices = pd.concat(
        [
            fund.rename("fund"),
            benchmark.rename("benchmark"),
        ],
        axis=1,
    )

    prices.index.name = "date"

    return prices


def validate_dates(start_date: str, end_date: str) -> None:
    """Validate the consistency of the supplied dates."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if start >= end:
        raise ValueError(
            "The start date must be earlier than the end date."
        )

    if end > date.today():
        raise ValueError(
            "The end date cannot be in the future."
        )