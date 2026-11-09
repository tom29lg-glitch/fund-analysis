"""Performance and risk metrics."""

import numpy as np
import pandas as pd


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized geometric return."""
    returns = returns.dropna()

    if returns.empty:
        return np.nan

    total_growth = (1 + returns).prod()

    if total_growth <= 0:
        return -1.0

    return total_growth ** (
        periods_per_year / len(returns)
    ) - 1


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized volatility."""
    returns = returns.dropna()

    if len(returns) < 2:
        return np.nan

    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized Sharpe ratio."""
    returns = returns.dropna()

    periodic_risk_free_rate = (
        (1 + annual_risk_free_rate)
        ** (1 / periods_per_year)
        - 1
    )
    excess_returns = returns - periodic_risk_free_rate
    volatility = excess_returns.std(ddof=1)

    if volatility == 0 or np.isnan(volatility):
        return np.nan

    return (
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )


def sortino_ratio(
    returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized Sortino ratio."""
    returns = returns.dropna()

    periodic_risk_free_rate = (
        (1 + annual_risk_free_rate)
        ** (1 / periods_per_year)
        - 1
    )
    excess_returns = returns - periodic_risk_free_rate
    negative_excess = np.minimum(excess_returns, 0)

    downside_deviation = np.sqrt(
        np.mean(negative_excess**2)
    )

    if downside_deviation == 0:
        return np.nan

    return (
        excess_returns.mean()
        / downside_deviation
        * np.sqrt(periods_per_year)
    )


def maximum_drawdown(returns: pd.Series) -> float:
    """Calculate the maximum loss from a previous peak."""
    returns = returns.dropna()

    wealth = (1 + returns).cumprod()
    previous_peaks = wealth.cummax()
    drawdowns = wealth / previous_peaks - 1

    return drawdowns.min()


def calculate_metrics(
    returns: pd.DataFrame,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Calculate the main metrics for each asset."""
    results = {}

    for column in returns.columns:
        asset_returns = returns[column]

        annual_return = annualized_return(
            asset_returns,
            periods_per_year,
        )
        max_drawdown = maximum_drawdown(asset_returns)

        calmar = (
            annual_return / abs(max_drawdown)
            if max_drawdown < 0
            else np.nan
        )

        results[column] = {
            "annualized_return": annual_return,
            "annualized_volatility": annualized_volatility(
                asset_returns,
                periods_per_year,
            ),
            "sharpe_ratio": sharpe_ratio(
                asset_returns,
                annual_risk_free_rate,
                periods_per_year,
            ),
            "sortino_ratio": sortino_ratio(
                asset_returns,
                annual_risk_free_rate,
                periods_per_year,
            ),
            "maximum_drawdown": max_drawdown,
            "calmar_ratio": calmar,
        }

    return pd.DataFrame(results).T


def calculate_beta(
    fund_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Calculate the fund's beta relative to the benchmark."""
    aligned = pd.concat(
        [fund_returns, benchmark_returns],
        axis=1,
    ).dropna()

    aligned.columns = ["fund", "benchmark"]

    benchmark_variance = aligned["benchmark"].var(ddof=1)

    if benchmark_variance == 0:
        return np.nan

    covariance = aligned["fund"].cov(
        aligned["benchmark"]
    )

    return covariance / benchmark_variance


def calculate_benchmark_metrics(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Calculate the fund's metrics relative to its benchmark.
    """
    required_columns = {"fund", "benchmark"}

    if not required_columns.issubset(returns.columns):
        raise ValueError(
            "The fund and benchmark columns are required."
        )

    aligned = returns[
        ["fund", "benchmark"]
    ].dropna()

    if len(aligned) < 2:
        raise ValueError(
            "Insufficient data to compare the fund "
            "with the benchmark."
        )

    fund = aligned["fund"]
    benchmark = aligned["benchmark"]
    active_returns = fund - benchmark

    active_return_volatility = active_returns.std(ddof=1)

    tracking_error = (
        active_return_volatility
        * np.sqrt(periods_per_year)
    )

    information_ratio = (
        active_returns.mean()
        / active_return_volatility
        * np.sqrt(periods_per_year)
        if active_return_volatility != 0
        else np.nan
    )

    fund_growth = (1 + fund).prod()
    benchmark_growth = (1 + benchmark).prod()

    relative_growth = fund_growth / benchmark_growth

    annualized_active_return = (
        relative_growth
        ** (periods_per_year / len(aligned))
        - 1
    )

    return pd.Series(
        {
            "beta": calculate_beta(fund, benchmark),
            "correlation": fund.corr(benchmark),
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "annualized_active_return": (
                annualized_active_return
            ),
        },
        name="fund_vs_benchmark",
    )