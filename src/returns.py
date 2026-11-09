"""Calculation of fund and benchmark returns."""

import pandas as pd


def calculate_daily_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate simple daily returns."""
    daily_returns = prices.pct_change(
        fill_method=None
    ).dropna(how="any")

    daily_returns.index.name = "date"

    return daily_returns


def calculate_monthly_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate monthly returns using the last available
    price of each month.
    """
    monthly_prices = prices.resample("ME").last()

    monthly_returns = monthly_prices.pct_change(
        fill_method=None
    ).dropna(how="any")

    monthly_returns.index.name = "date"

    return monthly_returns


def calculate_cumulative_returns(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate cumulative returns.

    Example: 0.25 represents a cumulative return of 25%.
    """
    cumulative_returns = (1 + returns).cumprod() - 1
    cumulative_returns.index.name = "date"

    return cumulative_returns


def calculate_growth_index(
    returns: pd.DataFrame,
    initial_value: float = 100.0,
) -> pd.DataFrame:
    """
    Calculate the value of an initial investment.

    An initial value of 100 makes it easy to compare the
    fund with its benchmark.
    """
    if initial_value <= 0:
        raise ValueError(
            "The initial value must be strictly positive."
        )

    growth_index = initial_value * (1 + returns).cumprod()
    growth_index.index.name = "date"

    return growth_index